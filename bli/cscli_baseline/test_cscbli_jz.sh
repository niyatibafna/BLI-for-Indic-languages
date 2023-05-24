#!/bin/bash

#SBATCH --job-name=test_cscbli    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1          # Name of the partition
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/testcsbli_%a.out   # output file name
#SBATCH --error=logs/testcsbli_%a.err    # error file name
#SBATCH --array=0-1              # job array 
#SBATCH --account ncm@v100

HOME="../../"

# Activate conda environment
source /home/$USER/.bashrc
source activate lrlm

# TODO change source target direction, hin should be target, use target2hin dicts

LANGS=("nep" "mar")
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}

# ctemb_paths=("$HOME/bli/cscli_baseline/contextual_embeddings/$lang/lm_(lm_mono.$lang.batchsize_16.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec" "$HOME/bli/cscli_baseline/contextual_embeddings/$lang/lm_(lm_mono.$lang.batchsize_64.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec")

# ssemb="$HOME/bli/bli_with_vecmap/embeddings/mapped/hin.dims_300.vec"
# stemb="$HOME/bli/bli_with_vecmap/embeddings/mapped/$lang.dims_300.vec"

model_path="outputs/$lang/_best"
dict_path="../../bli/get_eval_lexicon/lexicons/target2hin/"$lang"2hin.json"
src_lang=$lang
tgt_lang="hin"
dict_outputpath="$HOME/bli/lexicons/$lang/CSCBLI_unsup.$lang.ft_300.ct_768.csls.json"

CUDA_VISIBLE_DEVICES=0 python CSCBLI/unsupervised/test.py  --model_path $model_path \
        --dict_path $dict_path  --mode v1 \
        --src_lang $src_lang --tgt_lang $tgt_lang --lambda_w1 0.05 \
	--output_path $dict_outputpath
