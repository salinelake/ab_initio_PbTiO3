import ase
import ase.io
import os

supersize = 15
ss = supersize
# temp_list =   [300, 400, 500, 600,700, 750, 780, 790, 800, 810, 821, 821,  830, 840, 850, 860, 900, 1000, 1100, 1200]
# phase_list =  ['t', 't', 't', 't','t', 't', 't', 't', 't', 't', 't', 'c',  'c', 'c', 'c', 'c', 'c', 'c',   'c', 'c']

temp_list =   [ 823 ]
phase_list =  [ 't' ]
seed_list = [0,1]
os.system("echo '#batch submit' > run_batch.sh")
# os.system("echo '#batch submit' > run_infer.sh")
for temp,phase in zip(temp_list,phase_list):
    for seed in seed_list:
        folder = 'T{}S{}'.format(temp, seed )
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        os.system("cp ./template/npt.lammps {}".format(os.path.join(folder, 'in.lammps') ))
        conf = 'cubic' if phase=='c' else 'tetra'
        os.system("cp ./template/L{}x{}x{}_{}/conf.lmp {}".format(ss,ss,ss,conf,os.path.join(folder,'conf.lmp')))
        os.system("cp ./template/L{}x{}x{}_{}/type.raw {}".format(ss,ss,ss,conf,os.path.join(folder,'type.raw')))
        # os.system("cp ./template/L15x15x15_{}/type.raw {}".format(conf,os.path.join(folder,'type.raw')))
        # os.system("cp ./template/process.py {}".format( os.path.join(folder,'process.py')))
        # os.system("sed 's/_SS_/{}/' ./template/process.slurm > {}".format(ss,os.path.join(folder,'process.slurm')))
        # os.system("sed 's/REPLACE0/{}/' ./template/mpirun.slurm > {}".format(temp,os.path.join(folder,'mpirun.slurm')))
        os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
        os.system("sed 's/REPLACE_ncell/{}/' ./template/plumed.dat > {}".format(supersize**3,os.path.join(folder,'plumed.dat')))
        
        os.system("echo 'cd {}' >> run_batch.sh".format(folder))
        os.system("echo 'sbatch run.slurm' >> run_batch.sh")
        os.system("echo 'cd ..' >> run_batch.sh")

        
for temp,phase in zip(temp_list,phase_list):
    for seed in seed_list:
        folder = 'T{}S{}_boundary'.format(temp, seed )
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        os.system("cp ./template/langiven.lammps {}".format(os.path.join(folder, 'in.lammps') ))
        conf = 'cubic' if phase=='c' else 'tetra'
        os.system("cp ./template/L{}x{}x{}_{}/conf.lmp {}".format(ss,ss,ss,conf,os.path.join(folder,'conf.lmp')))
        os.system("cp ./template/L{}x{}x{}_{}/type.raw {}".format(ss,ss,ss,conf,os.path.join(folder,'type.raw')))
        # os.system("cp ./template/L15x15x15_{}/type.raw {}".format(conf,os.path.join(folder,'type.raw')))
        # os.system("cp ./template/process.py {}".format( os.path.join(folder,'process.py')))
        # os.system("sed 's/_SS_/{}/' ./template/process.slurm > {}".format(ss,os.path.join(folder,'process.slurm')))
        # os.system("sed 's/REPLACE0/{}/' ./template/mpirun.slurm > {}".format(temp,os.path.join(folder,'mpirun.slurm')))
        os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
        os.system("sed 's/REPLACE_ncell/{}/' ./template/plumed.dat > {}".format(supersize**3,os.path.join(folder,'plumed.dat')))
        
        os.system("echo 'cd {}' >> run_batch.sh".format(folder))
        os.system("echo 'sbatch run.slurm' >> run_batch.sh")
        os.system("echo 'cd ..' >> run_batch.sh")