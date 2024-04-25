#!/bin/bash -l
#SBATCH --nodes 4
#SBATCH --tasks-per-node=10
##SBATCH --mem-per-cpu=3G       # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="qe-scf" 


module load intel intel-mpi intel-mkl
source ~/utilities/activate_qe.sh

srun pw.x -input relax-sym.in > relax-sym.out

# srun $PH -input ph.in > ph.out

# rm -rf ./pwscf*
