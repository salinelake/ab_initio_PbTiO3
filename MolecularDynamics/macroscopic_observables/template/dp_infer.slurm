#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 5:00:0
#SBATCH --mem 64G 
#SBATCH --job-name=infer

# module purge
module load anaconda3
conda activate dp-plumed-mpi

python dp_infer.py