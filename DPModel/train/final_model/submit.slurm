#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 23:00:00
#SBATCH --mem=32G 
#SBATCH --partition=all 
#SBATCH --gres=gpu:1

module load anaconda3
conda activate dp-plumed-mpi

test $? -ne 0 && exit 1

if [ ! -f tag_0_finished ] ;then
  { if [ ! -f model.ckpt.index ]; then dp train input.json; else dp train input.json --restart model.ckpt; fi }  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_0_finished; fi 
fi


if [ ! -f tag_1_finished ] ;then
  dp freeze  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_1_finished; fi 
fi


