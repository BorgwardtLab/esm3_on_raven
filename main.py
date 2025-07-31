import os
import sys
import time
import torch
import pathlib
import argparse
import contextlib
import pickle as pkl
import atexit, errno

from torch.utils.data import DataLoader, Dataset

from huggingface_hub import login as hf_login

from esm.models.esm3 import ESM3
from esm.utils import encoding


class Mgnify90Dataset(Dataset):

    def __init__(self, shard):
        if isinstance(shard, dict):
            self.seqs = list(shard.values())
            self.ids = list(shard.keys())
        elif isinstance(shard, set):
            self.seqs = list(shard)
            self.ids = list(range(len(self.seqs)))
        elif isinstance(shard, list):
            self.seqs = shard
            self.ids = list(range(len(self.seqs)))
        else:
            raise ValueError(f"Invalid shard type: {type(shard)}")
        self.n_seqs = len(self.seqs)

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        return idx, self.seqs[idx]


def collate_fn(batch):
    indices, seqs = zip(*batch)
    tok_seqs = [
        torch.as_tensor(
            encoding.tokenize_sequence(s, tokenizer, add_special_tokens=True),
            dtype=torch.long,
        )
        for s in seqs
    ]
    max_len = max(t.size(0) for t in tok_seqs)
    tokens = torch.full((len(tok_seqs), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tok_seqs):
        tokens[i, : t.size(0)] = t

    mask = (tokens != pad_id) & (tokens != bos_id) & (tokens != eos_id)
    return torch.tensor(indices, dtype=torch.long), tokens, mask


def _cleanup_lock():
    try:
        os.rmdir(lock_dir)
    except FileNotFoundError:
        pass


def pooling(out, mask, out_dtype=torch.float16):
    """
    Mean-pool over residues-only with fp32 accumulation
    ---
    Why residues-only? Special tokens are noisy
    Why accumulate in fp32? To avoid fp16 overflow
    """
    per_tok_emb = out.embeddings.to(torch.float32)
    mask_f32    = mask.unsqueeze(-1).to(torch.float32)

    summed  = (per_tok_emb * mask_f32).sum(dim=1)
    lengths = mask_f32.sum(dim=1).clamp(min=1.0)
    mean    = summed / lengths

    return mean.to(out_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=str, default="/ptmp/bbana/mgnify90/1_1024")
    parser.add_argument("--model-precision", type=str, default="fp32")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--logging-interval", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed NaN/Inf dump and early exit")
    args = parser.parse_args()

    shards_path = pathlib.Path(args.shards)
    pkl_candidates = (
        [shards_path] if shards_path.is_file() else sorted(shards_path.glob("*.pkl"))
    )

    shard_path = None
    lock_dir = None
    for pkl_path in pkl_candidates:
        if pkl_path.with_suffix(".pt").exists():
            continue # already done
        cand_lock = pkl_path.with_suffix(".lock")
        try:
            os.mkdir(cand_lock) # atomic try‑lock
        except OSError as e:
            if e.errno == errno.EEXIST: # another worker works on it
                continue
            raise
        shard_path = pkl_path
        lock_dir = cand_lock
        break

    if shard_path is None:
        print("No free shards found.")
        sys.exit(2)

    print(f"Shard {shard_path} taken")

    atexit.register(_cleanup_lock)

    shard = pkl.load(open(shard_path, "rb"))

    dataset = Mgnify90Dataset(shard)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_login(token=os.environ["HF_TOKEN"])
    model = ESM3.from_pretrained("esm3-open").to(device).eval()

    if device != "cpu" and args.model_precision == "fp16":
        model = model.to(torch.float16)

    tokenizer = model.tokenizers.sequence
    pad_id, bos_id, eos_id = (
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    embedding_dim = model.encoder.sequence_embed.embedding_dim
    all_embeddings = torch.empty(len(dataset), embedding_dim, dtype=torch.float16)
    
    debug_path = shard_path.with_suffix(".debug.txt") if args.debug else None  # Path for detailed debug dump

    del shard

    start_time = time.perf_counter()

    with torch.no_grad():

        for idx, (ids_batch, token_batch, mask_batch) in enumerate(loader):
            token_batch = token_batch.to(device)
            mask_batch = mask_batch.to(device)

            ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device != "cpu"
                else contextlib.nullcontext()
            )
            with ctx:
                out = model.forward(sequence_tokens=token_batch)

            emb = pooling(out, mask_batch)

            if args.debug:
                bad_sample_mask = (~torch.isfinite(emb)).any(dim=1)  # NaN/Inf detection and detailed dump
                if bad_sample_mask.any():
                    bad_idx_in_batch = bad_sample_mask.nonzero(as_tuple=False)[0].item()
                    sample_id = ids_batch[bad_idx_in_batch].item()

                    emb_bad = emb[bad_idx_in_batch]
                    tok_bad = out.embeddings[bad_idx_in_batch]

                    n_nan_emb = torch.isnan(emb_bad).sum().item()
                    n_inf_emb = torch.isinf(emb_bad).sum().item()
                    dim_emb = emb_bad.numel()

                    n_nan_tok = torch.isnan(tok_bad).sum().item()
                    n_inf_tok = torch.isinf(tok_bad).sum().item()
                    tot_vals = tok_bad.numel()

                    nan_tok_cnt = torch.isnan(tok_bad).any(dim=1).sum().item()
                    inf_tok_cnt = torch.isinf(tok_bad).any(dim=1).sum().item()
                    tot_tok_cnt = tok_bad.size(0)

                    with open(debug_path, "w") as f:
                        f.write(f"{n_nan_emb},{n_inf_emb},{dim_emb}\n")
                        f.write(",".join(map(lambda x: str(float(x)), emb_bad.tolist())) + "\n")
                        f.write("\n")
                        f.write(f"{n_nan_tok},{n_inf_tok},{tot_vals}\n")
                        f.write(f"{nan_tok_cnt},{inf_tok_cnt},{tot_tok_cnt}\n")
                        for row in tok_bad:
                            f.write(",".join(map(lambda x: str(float(x)), row.tolist())) + "\n")

                    print(f"Encountered NaN/Inf in sample {sample_id}; debug info written to {debug_path}")
                    sys.exit(1)

            all_embeddings[ids_batch] = emb.cpu()

            if idx % args.logging_interval == 0 and idx > 0:
                elapsed = time.perf_counter() - start_time
                tput = (args.logging_interval * args.batch_size)/elapsed
                print(f"{idx} / {len(loader)} | tput={tput:.2f} seq/s")
                start_time = time.perf_counter()

    mask_good = torch.isfinite(all_embeddings).all(dim=1)
    num_bad   = (~mask_good).sum().item()

    if num_bad > 0:
        print(f"{shard_path}: "
              f"discarding {num_bad} / {all_embeddings.size(0)} embeddings containing NaN/Inf")
        all_embeddings = all_embeddings[mask_good]
    else:
        print(f"{shard_path}: all good")

    torch.save(all_embeddings, shard_path.with_suffix(".pt"))

    torch.cuda.empty_cache(); torch.cuda.ipc_collect() # clean up the memory, otherwise the next run will fail
    del model, loader, all_embeddings
