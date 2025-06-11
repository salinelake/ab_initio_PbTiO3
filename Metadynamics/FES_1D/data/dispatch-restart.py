import ase
import ase.io
import os

kb = 8.617333e-5  #eV/K
unit_system = ase.io.read('./template/cubic.lmp',format='lammps-data',style='atomic')
temp_list = [815,820,825,830]

# repeat_list = [3,2]
# barrier_list = [0.3, 0.6]
# repeat_list = [4]
# barrier_list = [0.2]
repeat_list = [5]
barrier_list = [0.1]

os.system("echo '#batch restart' > restart.sh")
for repeat, barrier in zip(repeat_list, barrier_list):
    for temp in temp_list:
        supersize = repeat*3
        ncell = supersize ** 3
        natoms = ncell * 5
        folder = '{}K{}x{}x{}'.format(temp,supersize,supersize,supersize)
        biasf = int(barrier * natoms / 1000 /kb/temp)
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        ## copy lammps input
        os.system("cp ./restart_template/restart.lammps {}/".format(folder))
        ## copy slurm submission scripts
        if repeat == 5:
            os.system("sed 's/REPLACE0/{}/' ./restart_template/restart_long.slurm > {}".format(
                temp,os.path.join(folder,'restart.slurm')))
        else:
            os.system("sed 's/REPLACE0/{}/' ./restart_template/restart.slurm > {}".format(
                temp,os.path.join(folder,'restart.slurm')))
        ## copy plumed scripts
        os.system("sed 's/REPLACE_ncell/{}/' ./restart_template/plumed-restart.dat > {}".format(
            ncell,os.path.join(folder,'plumed-restart.dat')))
        os.system("sed 's/REPLACE_bias/{}/' {} > {}".format(
            biasf,os.path.join(folder,'plumed-restart.dat'),os.path.join(folder,'plumed-restart.dat.tmp')))
        os.system("sed 's/REPLACE_temp/{}/' {} > {}".format(
            temp,os.path.join(folder,'plumed-restart.dat.tmp'),os.path.join(folder,'plumed-restart.dat')))
        os.system("rm {}".format(os.path.join(folder,'plumed-restart.dat.tmp')))
        ## setup shell scripts for batch submission
        os.system("echo 'cd {}' >> restart.sh".format(folder))
        os.system("echo 'sbatch restart.slurm' >> restart.sh")
        os.system("echo 'cd ..' >> restart.sh")

    
