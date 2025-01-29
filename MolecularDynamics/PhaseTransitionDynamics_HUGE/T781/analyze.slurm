#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 23:0:0
#SBATCH --mem 32G 
#SBATCH --job-name=pp

module purge
module load anaconda3/2020.11
# module load intel/2021.1
# module load intel-mpi/intel/2021.1.1
# module load cudatoolkit/11.3
# module load cudnn/cuda-11.x/8.2.0
# conda activate dp-test
conda activate deepmd221

python analyze.py
