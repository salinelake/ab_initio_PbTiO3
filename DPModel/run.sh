#!/bin/bash -l
#SBATCH -N 4
#SBATCH -n 20
#SBATCH -t 1:0:0
##SBATCH --mem 128G 
##SBATCH --gres=gpu:1
#SBATCH --job-name=test

hostname
module load intel-mkl/2017.4/5/64
module load intel/17.0/64/17.0.5.239
module load intel-mpi/intel/2017.5/64
module load cudatoolkit/9.1
module load cudnn/cuda-9.1/7.1.2
# source /home/linfengz/SCR/softwares/tf_venv/bin/activate
source /tigress/pinchenx/utilities/activate_qe.sh
# source env.sh

conda activate /tigress/pinchenx/conda_envs/dp115gen092
nohup dpgen run param.json machine.json 1> out.log 2>&1 &
nohup dpgen run param.json cori-nersc.json 1> out.log 2>&1 &
