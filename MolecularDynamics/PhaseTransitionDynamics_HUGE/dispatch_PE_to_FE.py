import ase
import ase.io
import os
import numpy as np
# from init_conf import *
meV2KJpmol = 1/10.364
temp_list   = [778, 780, 781,]  
sc          = [512,16,16]
ncell       = sc[0]*sc[1]*sc[2]

os.system("echo '#batch submit' > run_batch.sh")
for temp in temp_list:
    folder = 'T{}'.format(temp)
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    ### lammps
    os.system("cp ./template/in.lammps {}/".format(folder))
    os.system("cp ./template/analyze.py {}/".format(folder))
    os.system("cp ./template/analyze.slurm {}/".format(folder))
    ## initial configuration
    conf_file = 'L{}x{}x{}_cubic.lmp'.format(sc[0],sc[1],sc[2])
    os.system("cp ./template/{} {}".format(conf_file, os.path.join(folder,'conf.lmp')))
    ## slurm
    os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
    ## submission
    os.system("echo 'cd {}' >> run_batch.sh".format(folder))
    os.system("echo 'sbatch run.slurm' >> run_batch.sh")
    os.system("echo 'cd ..' >> run_batch.sh")


