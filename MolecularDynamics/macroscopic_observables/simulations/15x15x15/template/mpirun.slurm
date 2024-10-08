#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 23:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:2
#SBATCH --job-name=print


module load anaconda3
conda activate dp-plumed-mpi
export OMPI_MCA_opal_cuda_support=true
 
ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DPModel/train/final_model/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DWModel/train/final_model/dipole-compress.pb dipole.pb

mpirun -n 2 lmp -v TEMP REPLACE0 -in in.lammps > pto.log
