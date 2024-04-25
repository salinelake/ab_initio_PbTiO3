#!/bin/bash -l
#SBATCH --nodes 2
#SBATCH --tasks-per-node=10
##SBATCH --mem-per-cpu=3G       # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="qe-scf" 
#SBATCH --array=0-10


module load intel intel-mpi intel-mkl
PW=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/pw.x
PH=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/ph.x
PP=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/pp.x

cd ./$SLURM_ARRAY_TASK_ID

srun $PW -input qe.in > qe.out
# srun $PW -input berry.in > berry.out
srun $PP -input pp.in > pp.out

rm -rf ./pwscf*
