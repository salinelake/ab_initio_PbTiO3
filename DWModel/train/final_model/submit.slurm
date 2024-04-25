#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 6:00:00
#SBATCH --mem=32G 
#SBATCH --partition=all 
#SBATCH --gres=gpu:1


# module load anaconda
# conda activate /tigress/pinchenx/conda_envs/deepmd133
# source /home/pinchenx/tigress/utilities/activate_dpgen.sh
# source /home/yixiaoc/SCR/anaconda3/bin/activate dp-dev
# conda activate dp-dev
conda activate dp-plumed-mpi

test $? -ne 0 && exit 1

if [ ! -f tag_0_finished ] ;then
  { if [ ! -f model.ckpt.index ]; then dp train input_v2.json; else dp train input_v2.json --restart model.ckpt; fi }  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_0_finished; fi 
fi

if [ ! -f tag_1_finished ] ;then
  dp freeze  -o dipole.pb 1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_1_finished; fi 
fi
