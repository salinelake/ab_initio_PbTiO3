#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 24:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=restart

module load anaconda3/2020.7
conda activate /home/bingjiay/data.gpfs/envs/dp-plumed-mpi


lmp -v TEMP REPLACE0 -in restart.lammps > pto-restart.log
