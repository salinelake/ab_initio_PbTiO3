#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 23:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:2
#SBATCH --job-name=print


module purge
module load anaconda3
conda activate dp-plumed-mpi

##openmp=4.1
# source /scratch/gpfs/yixiaoc/anaconda3/bin/activate dpdev
# export OMPI_MCA_opal_cuda_support=true

unlink frozen_model.pb
unlink dipole.pb
# ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2/frozen_model.pb frozen_model.pb
ln -s  /home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb dipole.pb


mpirun -n 2 lmp -v TEMP REPLACE0 -in in.lammps > pto.log
