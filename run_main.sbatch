#!/bin/bash -l
#SBATCH -o ./logs/esm.%j.out
#SBATCH -e ./logs/esm.%j.err
#SBATCH -D /u/bbana/esm3
#SBATCH -J esm
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --mail-type=none
#SBATCH --mail-user=username@biochem.mpg.de
#SBATCH --time=1-00:00:00

source /u/bbana/.bashrc
mamba activate esm3

DIR=${SHARDS:-"/ptmp/bbana/mgnify90/1_1024"}
COUNT=${SHARDS_PER_JOB:-4}
BATCH_SIZE=${BATCH_SIZE:-5}

for ((n=0; n<COUNT; n++)); do
    echo "[$((n+1))/$COUNT] requesting a shard"
    python -u main.py --shards-dir "$DIR" --batch-size "$BATCH_SIZE"
    rc=$?
    if [[ $rc -eq 2 ]]; then
        echo "No shards left.  Exiting early."
        break
    fi
done