#!/bin/bash

#SBATCH --job-name=vmap_embs    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=20G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=vm.out   # output file name
#SBATCH --error=vm.err    # error file name
#SBATCH --array 0-1

# source="mag"
sources=("bho" "mag")
source=${sources[$SLURM_ARRAY_TASK_ID]}
target="hin"

dims=100

SRC="embeddings/monolingual/"$source".dims_$dims.vec"
TRG="embeddings/monolingual/"$target".dims_$dims.vec"

SRC_MAPPED="embeddings/mapped/"$source".dims_$dims.vec"
TRG_MAPPED="embeddings/mapped/"$target".dims_$dims.vec"

python3 vecmap/map_embeddings.py --cuda --identical $SRC $TRG $SRC_MAPPED $TRG_MAPPED