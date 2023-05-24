#!/bin/bash

#SBATCH --job-name=preproc    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=cpu_homogen          # Name of the partition
##SBATCH --partition=gpu          # Name of the partition
##SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=40G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=100:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=pre2.out   # output file name
#SBATCH --error=pre2.err    # error file name


echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}


module purge
module load gnu8 mpich
# module load cuda/10.2.89


# Set your conda environment
source /home/$USER/.bashrc
source activate lrlm

python3 datareader.py