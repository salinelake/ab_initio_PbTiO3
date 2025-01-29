#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 23:00:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:4
#SBATCH --job-name=ferroswitch
##SBATCH --constraint=gpu80

source /tigress/pinchenx/utilities/activate_dpdev_della.sh
 

unlink frozen_model.pb
unlink dipole.pb

# ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2/frozen_model.pb frozen_model.pb
ln -s  /tigress/pinchenx/DPModels/PTO-MODEL_DEV/m0/model-compress.pb frozen_model.pb
ln -s  /tigress/pinchenx/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb dipole.pb


# lmp -v TEMP 781 -in in.lammps > pto.log &
seed=$RANDOM
echo "seed is $seed"
mpirun -n 4 lmp -v TEMP 781 -v SEED $seed -in in.lammps > pto.log


sbatch analyze.slurm