#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 23:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=pp
#SBATCH --array=0-9


module purge
module load anaconda3/2021.11
conda activate /scratch/gpfs/bingjiay/envs/deepmd215/

idx=$SLURM_ARRAY_TASK_ID
python process.py $idx _SS_
rm pto${idx}.lammpstrj