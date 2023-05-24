#!/bin/bash

#SBATCH --job-name=run_cscbli    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --partition=cpu_homogen          # Name of the partition
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=60G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/cscbli_%a.out   # output file name
#SBATCH --error=logs/cscbli_%a.err    # error file name
#SBATCH --array=1              # job array 

HOME="../../"

# Activate conda environment
source /home/$USER/.bashrc
source activate lrlm

LANGS=("bho" "mag")
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}
lang="mag"
echo $lang

# ssemb="$HOME/bli/bli_with_vecmap/embeddings/mapped/$lang.dims_300.vec"
# stemb="$HOME/bli/bli_with_vecmap/embeddings/mapped/hin.dims_300.vec"

# csemb_paths=("$HOME/bli/cscli_baseline/contextual_embeddings/$lang/lm_(lm_mono.$lang.batchsize_16.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec" "$HOME/bli/cscli_baseline/contextual_embeddings/$lang/lm_(lm_mono.$lang.batchsize_64.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec")
# csemb=${csemb_paths[$SLURM_ARRAY_TASK_ID]}
# ctemb="$HOME/bli/cscli_baseline/contextual_embeddings/hin/lm_pt_.min,maxcontexts_5,15.vec"

ssemb="embeddings_row_aligned/static/$lang.dims_300.vec"
stemb="embeddings_row_aligned/static/hin.dims_300.vec"

csemb="embeddings_row_aligned/contextual/$lang.dims_768.vec"
ctemb="embeddings_row_aligned/contextual/hin.dims_768.vec"

save_path="outputs/$lang/"

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0 python CSCBLI/unsupervised/train.py --src_lang $lang --tgt_lang hin \
  --static_src_emb_path $ssemb --static_tgt_emb_path $stemb\
  --context_src_emb_path $csemb --context_tgt_emb_path $ctemb\
   --save_path $save_path