#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 24:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=fse

module load anaconda3/2020.7
conda activate /home/bingjiay/data.gpfs/envs/dp-plumed-mpi

##openmp=4.1
# source /scratch/gpfs/yixiaoc/anaconda3/bin/activate dpdev
# export OMPI_MCA_opal_cuda_support=true

unlink frozen_model.pb
unlink dipole.pb
# ln -s  /home/pinchenx/ferro/DPGEN/_RE_TRAIN/iter9/v2/frozen_model.pb frozen_model.pb
ln -s  /home/bingjiay/data.gpfs/pinchen/frozen_models/dp-compress.pb frozen_model.pb
ln -s  /home/bingjiay/data.gpfs/pinchen/frozen_models/dipole-compress.pb dipole.pb

lmp -v TEMP 820 -in in.lammps > pto.log
