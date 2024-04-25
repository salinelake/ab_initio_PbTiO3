#!/bin/bash -l
#SBATCH --nodes 8
#SBATCH --tasks-per-node=10
##SBATCH --mem=128G       # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="qe-scf" 


module load anaconda intel intel-mpi intel-mkl
# PW=/tigress/bingjiay/softwares/QE_6.4.1/q-e-qe-6.4.1/bin/pw.x
PW=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/pw.x
W90=/home/pinchenx/data.gpfs/softwares/wannier90-3.1.0/wannier90.x
PWW90=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/pw2wannier90.x
# PH=/tigress/bingjiay/softwares/QE_6.7/qe-6.7/bin/ph.x

# generate si.xml, si.save, scf.out
srun $PW -input PbTiO3.scf > scf.out  

## complete k-mesh, generate nscf.out, si.wfc
srun $PW -input PbTiO3.nscf > nscf.out 

# # generate .nnkp as the input of QE pw2wannier90
srun $W90 -pp PbTiO3

# produce the matrices .mmn, .amn, .eig…
srun $PWW90 < PbTiO3.pw2wan > pw2wan.out

#minimize the spread, calculate wannier function
srun $W90 PbTiO3

# echo "set term png" >> PbTiO3_band.gnu
# echo "set output 'band_structure.png'" >> PbTiO3_band.gnu
# echo "replot" >> PbTiO3_band.gnu
# gnuplot PbTiO3_band.gnu
# rm *_band.*

# srun $PW -input BP.in > BP.out
# srun $PH -input ph.in > ph.out

# conda activate /tigress/pinchenx/conda_envs/deepmd_gpu
# python postprocessing.py >> log.txt

# rm -rf ./pwscf*
