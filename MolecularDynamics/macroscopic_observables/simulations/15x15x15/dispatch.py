import ase
import ase.io
import os

supersize = 15
# temp_list =   [300, 400, 500, 600,700, 750,780, 790,800, 810, 815, 820,  825, 830, 840, 850, 860, 870, 900, 1000]
# phase_list =  ['t', 't', 't', 't','t', 't','t', 't','t', 't', 't', 't',  'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
temp_list =   [785, 790,795,800,      825,830,835,840]
phase_list =  ['c', 'c', 'c', 'c',   't','t','t','t']
# temp_list =   [821,822,823,824,     821,822,823,824]
# phase_list =  ['t', 't', 't', 't',  'c', 'c','c','c']
os.system("echo '#batch submit' > run_batch.sh")
# os.system("echo '#batch submit' > run_infer.sh")
for temp,phase in zip(temp_list,phase_list):
    folder = 'T{}_ss{}_{}'.format(temp,supersize,phase)
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    os.system("cp ./template/in.lammps {}/".format(folder))
    os.system("cp ./template/type.raw {}/".format(folder))
    conf = 'cubic.lmp' if phase=='c' else 'tetra.lmp'
    os.system("cp ./template/{} {}".format(conf,os.path.join(folder,'conf.lmp')))
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
    
