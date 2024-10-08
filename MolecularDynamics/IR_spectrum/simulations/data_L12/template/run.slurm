#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 70:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=irL15

source /tigress/pinchenx/utilities/activate_dpdev_della.sh
 

ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DPModel/train/final_model/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/data.gpfs/ab_initio_PbTiO3/DWModel/train/final_model/dipole-compress.pb dipole.pb


seed=$RANDOM
echo $seed > seed.txt
mpirun -n 1 lmp -v TEMP REPLACE0 -v SEED $seed -in in.lammps > pto.log

# sbatch process.slurm
