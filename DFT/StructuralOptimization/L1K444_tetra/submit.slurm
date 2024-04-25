#!/bin/bash -l
#SBATCH --nodes 4
#SBATCH --tasks-per-node=20
#SBATCH --time=01:00:00
#SBATCH --job-name='qe' 
source /home/pinchenx/utilities/activate_qe.sh
module load intel intel-mpi
srun pw.x -in relax-sym.in >> relax-sym.out
