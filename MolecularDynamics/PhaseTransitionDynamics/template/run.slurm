#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 23:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=irL12

source /tigress/pinchenx/utilities/activate_dpdev_della.sh
 

unlink frozen_model.pb
unlink dipole.pb

# ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2/frozen_model.pb frozen_model.pb
ln -s  /home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb dipole.pb


seed=$RANDOM
echo $seed > seed.txt
mpirun -n 1 lmp -v TEMP REPLACE0 -v SEED $seed -in in.lammps > pto.log

# sbatch process.slurm
