# Distributed ESM-3 embedding generation on the Raven cluster 

The dataset is split into **shards** (`*.pkl`) inside the directory referenced by `$SHARDS` (default: `/ptmp/bbana/mgnify90/1_1024`). Each shard is processed **exactly once**:
* `main.py` creates an atomic *lock* directory `<shard>.lock` when it starts.  
* When finished it saves the embedding tensor `<shard>.pt` and removes the lock.
* Use `$BATCH_SIZE` to adjust the batch size used by ESM3.

Because of this locking, you can launch as many independent Slurm jobs as you have GPUs—every job will pick the next free shard and you'll never compute the same shard twice.

## Usage

Submit **`run_main.sbatch`** with the desired number of shards to process sequentially in one job:

```bash
# run up to 4 shards in a single GPU job
sbatch --export=SHARDS="/ptmp/bbana/mgnify90/1_1024",SHARDS_PER_JOB=4,BATCH_SIZE=5 run_main.sbatch
```

Launch the same command multiple times (or turn it into an array job) to keep the queue full:

```bash
N=10; for i in $(seq "$N"); do sbatch --export=SHARDS="/ptmp/bbana/mgnify90/1_1024",SHARDS_PER_JOB=4,BATCH_SIZE=5 run_main.sbatch; done
```

## Installation guide

This short guide covers environment creation, dependency installation, and model-weight access for running **ESM-3 small** on the MPCDF Raven GPUs. The repository provides code for generating embeddings in a distributed way for protein sequences saved across multiple shards.

---

**Log in**

```bash
ssh <mpcdf_username>@raven0{1,2,3,4}i.mpcdf.mpg.de
```

**Create & activate a Conda environment**

```bash
mamba create -n esm3 python=3.11
mamba activate esm3
```

**Load required system modules**

```bash
module load cuda/12.6   # GPU toolkit
module load ninja       # faster C/C++ builds
module load gcc/11      # compiler for PyTorch/flash‑attn
```

**Install Python dependencies**

```bash
pip install esm3
MAX_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3 pip install flash-attn --no-build-isolation
```

**Download the model weights**

The **ESM-3 small** checkpoint is hosted on Hugging Face and requires a valid token.

```bash
export HF_TOKEN=<your_token>   # set once per shell
# the model URL is https://huggingface.co/EvolutionaryScale/esm3
```

---

**Notes**

* On an NVIDIA **A100 40 GB**, the maximum safe batch size for sequences **<1024 tokens** is **5**.
* ⚠️ flash-attn can spawn too many compile threads, that's why it's recommended to explictly limit the number of jobs during installation - `MAX_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3`. If not limited, a complete stall of the system might happen. ⚠️