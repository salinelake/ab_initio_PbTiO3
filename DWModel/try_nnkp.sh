module load intel-mkl
cd $1
/home/pinchenx/data.gpfs/softwares/wannier90-3.1.0/wannier90.x -pp PTO
if test $? -eq 0; then 
    true
else
    echo $1 >> /home/pinchenx/ferro/DeepWannier/iter-nnkp_failed.log
fi