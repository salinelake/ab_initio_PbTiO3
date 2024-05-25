#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=nptL15

# module purge
module load anaconda3
conda activate dp-plumed-mpi

##openmp=4.1
# source /scratch/gpfs/yixiaoc/anaconda3/bin/activate dpdev
# export OMPI_MCA_opal_cuda_support=true
unlink frozen_model.pb
unlink dipole.pb
# ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2/frozen_model.pb frozen_model.pb
ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2_small/model-compress.pb frozen_model.pb
ln -s  /home/pinchenx/ferro/DeepWannier/_TRAIN/dp-plumed-global/dipole-compress.pb dipole.pb

lmp -v TEMP REPLACE0 -in in.lammps > pto.log