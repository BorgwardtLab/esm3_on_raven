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
        self.seqs = list(shard.values())
        self.ids = list(shard.keys())
        self.n_seqs = len(self.seqs)

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        return idx, self.seqs[idx]


def collate_fn(batch):
    indices, seqs = zip(*batch)
    tok_seqs = [
        torch.tensor(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="/ptmp/bbana/mgnify90/1_1024",
        help="Directory containing shard .pkl files",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--logging-interval", type=int, default=1000)
    args = parser.parse_args()

    shards_dir = pathlib.Path(args.shards_dir)

    shard_path = None
    lock_dir = None
    for pkl_path in sorted(shards_dir.glob("*.pkl")):
        if pkl_path.with_suffix(".pt").exists():
            continue                            # already done
        cand_lock = pkl_path.with_suffix(".lock")
        try:
            os.mkdir(cand_lock)                # atomic try‑lock
        except OSError as e:
            if e.errno == errno.EEXIST:        # someone else owns it
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

    if device != "cpu":
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

    del shard

    start_time = time.perf_counter()

    with torch.no_grad():

        for idx, (ids_batch, token_batch, mask_batch) in enumerate(loader):
            token_batch = token_batch.to(device)
            mask_batch = mask_batch.to(device)

            avg_plddt = torch.ones(
                token_batch.shape, dtype=torch.float16, device=device
            )
            per_res_plddt = torch.zeros(
                token_batch.shape, dtype=torch.float16, device=device
            )

            ctx = (
                torch.autocast("cuda", dtype=torch.float16)
                if device != "cpu"
                else contextlib.nullcontext()
            )
            with ctx:
                out = model.forward(
                    sequence_tokens=token_batch,
                    average_plddt=avg_plddt,
                    per_res_plddt=per_res_plddt,
                )

            per_tok_emb = out.embeddings.to(torch.float16)

            summed = (per_tok_emb * mask_batch.unsqueeze(-1)).sum(dim=1)
            lengths = mask_batch.sum(dim=1, keepdim=True).to(torch.float16)
            mean_emb = summed / lengths.clamp(min=1)

            all_embeddings[ids_batch] = mean_emb.cpu()

            if idx % args.logging_interval == 0 and idx > 0:
                elapsed = time.perf_counter() - start_time
                tput = (args.logging_interval * args.batch_size)/elapsed
                print(f"{idx} / {len(loader)} | tput={tput:.2f} seq/s")
                start_time = time.perf_counter()

    torch.save(all_embeddings, shard_path.with_suffix(".pt"))

    # clean up the memory, otherwise the next run will fail
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    del token_batch, mask_batch, avg_plddt, per_res_plddt
    del out, per_tok_emb, summed, lengths, mean_emb
    del model