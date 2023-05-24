#!/bin/bash

#SBATCH --job-name=hrlm5_con_embs    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=3               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --partition=cpu_homogen          # Name of the partition
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=60G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/c_m5_embs_hrl_%a.out   # output file name
#SBATCH --error=logs/c_m5_embs_hrl_%a.err    # error file name
#SBATCH --array=0-1              # job array 

HOME="../../"

# Activate conda environment
source /home/$USER/.bashrc
source activate lrlm



LANGS=("hin" "mar" "nep")
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}

epnames=("l3cube" "l3cube" "Shushant")
epname=${ep_names[$SLURM_ARRAY_TASK_ID]}

modelnames=("l3cube-pune/hindi-bert-scratch" "l3cube-pune/marathi-bert-scratch" "Shushant/nepaliBERT")
modelname=${modelnames[$SLURM_ARRAY_TASK_ID]}

mkdir -p "contextual_embeddings/$lang"

corpus_path="${HOME}/data/monolingual/all/$lang.txt"
tokenizer_path=$modelname
model_path=$modelname
output_path="contextual_embeddings/$lang/lm_pt_$epname.min,maxcontexts_5,15.vec"


python3 get_context_embeddings.py \
    --corpus_path $corpus_path \
    --tokenizer_path $tokenizer_path \
    --model_path $model_path \
    --output_path $output_path \
    --num_words 200000 \
    --min_contexts 5 \
    --max_contexts 15 \
    --batch_size 64 \



