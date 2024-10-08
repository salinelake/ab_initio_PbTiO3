import ase
import ase.io
import os

supersize = 15
# temp_list =   [300, 400, 500, 600,700, 750, 780, 790, 800, 810,   830, 840, 850, 860, 900, 1000, 1100, 1200]
# phase_list =  ['t', 't', 't', 't','t', 't', 't', 't', 't', 't',   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
temp_list =   [821]
phase_list =  ['c']

os.system("echo '#batch submit' > run_batch.sh")
# os.system("echo '#batch submit' > run_infer.sh")
for temp,phase in zip(temp_list,phase_list):
    folder = 'T{}_ss{}'.format(temp,supersize)
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    os.system("cp ./template/in.lammps {}/".format(folder))
    conf = 'cubic' if phase=='c' else 'tetra'
    os.system("cp ./template/L15x15x15_{}/conf.lmp {}".format(conf,os.path.join(folder,'conf.lmp')))
    os.system("cp ./template/L15x15x15_{}/type.raw {}".format(conf,os.path.join(folder,'type.raw')))
    # os.system("cp ./template/dp_infer.py {}".format(os.path.join(folder,'dp_infer.py')))
    # os.system("cp ./template/dp_infer.slurm {}".format(os.path.join(folder,'dp_infer.slurm')))
    os.system("sed 's/REPLACE0/{}/' ./template/mpirun.slurm > {}".format(temp,os.path.join(folder,'mpirun.slurm')))
    os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
    os.system("sed 's/REPLACE_ncell/{}/' ./template/plumed.dat > {}".format(supersize**3,os.path.join(folder,'plumed.dat')))
    
    os.system("echo 'cd {}' >> run_batch.sh".format(folder))
    os.system("echo 'sbatch run.slurm' >> run_batch.sh")
    os.system("echo 'cd ..' >> run_batch.sh")

    # os.system("echo 'cd {}' >> run_infer.sh".format(folder))
    # os.system("echo 'sbatch dp_infer.slurm' >> run_infer.sh")
    # os.system("echo 'cd ..' >> run_infer.sh")
    
