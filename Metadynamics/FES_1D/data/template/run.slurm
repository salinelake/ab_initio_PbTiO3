#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 80:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=fse

module load anaconda3/2020.7
conda activate dp-plumed-mpi

unlink frozen_model.pb
unlink dipole.pb
ln -s  /home/cfs_m1027/Landau_PTO/PTO_MolecularDynamics/DPModel/train/final_model/model-compress.pb frozen_model.pb
ln -s  /home/cfs_m1027/Landau_PTO/PTO_MolecularDynamics/DWModel/train/final_model/dipole-compress.pb dipole.pb

lmp -v TEMP REPLACE0 -in in.lammps > pto.log
