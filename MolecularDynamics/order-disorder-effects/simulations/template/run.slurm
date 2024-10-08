#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 23:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=nptL15

module purge
module load anaconda3
conda activate dp-plumed-mpi

ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DPModel/train/final_model/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DWModel/train/final_model/dipole-compress.pb dipole.pb

lmp -v TEMP REPLACE0 -in in.lammps > pto.log