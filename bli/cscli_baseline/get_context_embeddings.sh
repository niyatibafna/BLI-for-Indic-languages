#!/bin/bash

#SBATCH --job-name=lrl_con_embs    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --partition=cpu_homogen          # Name of the partition
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=60G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/c_embs_%a.out   # output file name
#SBATCH --error=logs/c_embs_%a.err    # error file name
#SBATCH --array=0-1              # job array 

HOME="../../"

# Activate conda environment
source activate lrlm



LANGS=("bho" "mag" "hin" "mar" "nep")
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}

batchsize=64
if [ $lang == "bho" ]
then
    batchsize=16
fi

mkdir -p "contextual_embeddings/$lang"

corpus_path="${HOME}/data/monolingual/all/$lang.txt"
tokenizer_path="$HOME/training_outputs/tokenizers/$lang/lm_mono.$lang.batchsize_$batchsize.vocabsize_30522.epochs_40.json"
model_path="$HOME/training_outputs/models/$lang/lm_mono.$lang.batchsize_$batchsize.vocabsize_30522.epochs_40"
output_path="contextual_embeddings/$lang/lm_(lm_mono.$lang.batchsize_$batchsize.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec"


python3 get_context_embeddings.py \
    --corpus_path $corpus_path \
    --tokenizer_path $tokenizer_path \
    --model_path $model_path \
    --output_path $output_path \
    --num_words 100000 \
    --min_contexts 2 \
    --max_contexts 15 \
    --batch_size 32 \



