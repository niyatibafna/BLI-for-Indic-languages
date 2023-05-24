#!/bin/bash

#SBATCH --job-name=rb    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=cpu_homogen          # Name of the partition
##SBATCH --partition=gpu          # Name of the partition
##SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=40G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/rb_%a.out   # output file name
#SBATCH --error=logs/rb_%a.err    # error file name
#SBATCH --array 0-1

echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}


module purge
module load gnu8 mpich
# module load cuda/10.2.89


# Set your conda environment
source /home/$USER/.bashrc
source activate lrlm

LANGS=("mag" "bho") # "mai" "awa" "bra"

lang=${LANGS[$SLURM_ARRAY_TASK_ID]}

batch_size=100
iterations=3
threshold=0.5
HF_MODEL_NAME="google/muril-base-cased"
OUTDIR="../lexicons/"$lang
PARAMS_DIR="../rulebook_params/"$lang

mkdir -p $OUTDIR
mkdir -p $PARAMS_DIR

TARGET_FILE_PATH="../../data/monolingual/all/"$lang".txt"

echo $OUTDIR
echo $PARAMS_DIR
echo $TARGET_FILE_PATH

python3 rulebook.py --batch_size $batch_size --iterations $iterations --threshold $threshold \
--lang $lang --hf_model_name $HF_MODEL_NAME --OUTDIR $OUTDIR --TARGET_FILE_PATH $TARGET_FILE_PATH \
--PARAMS_DIR $PARAMS_DIR
