import ase
import ase.io
import os

kb = 8.617333e-5  #eV/K
unit_system = ase.io.read('./template/cubic.lmp',format='lammps-data',style='atomic')
temp_list = [815,820,825,830]
repeat_list = [5,4,3,2]
barrier_list = [0.1, 0.2, 0.3, 0.6]
os.system("echo '# batch submit' > submit.sh" )
os.system("echo '# plumed sumhill' > sumhills.sh" )

for repeat, barrier in zip(repeat_list, barrier_list):
    for temp in temp_list:
        supersize = repeat*3
        ncell = supersize ** 3
        natoms = ncell * 5
        folder = '{}K{}x{}x{}'.format(temp,supersize,supersize,supersize)
        biasf = int(barrier * natoms / 1000 /kb/temp)
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        os.system("cp ./template/in.lammps {}/".format(folder))
        if repeat ==2:
            os.system("sed 's/REPLACE0/{}/' ./template/run_short.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
        elif repeat > 4:
            os.system("sed 's/REPLACE0/{}/' ./template/run_long.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
        else:
            os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))
        os.system("sed 's/REPLACE_ncell/{}/' ./template/plumed.dat > {}".format(ncell,os.path.join(folder,'plumed.dat')))
        os.system("sed 's/REPLACE_bias/{}/' {} > {}".format(biasf,os.path.join(folder,'plumed.dat'),os.path.join(folder,'plumed.dat.tmp')))
        os.system("sed 's/REPLACE_temp/{}/' {} > {}".format(temp,os.path.join(folder,'plumed.dat.tmp'),os.path.join(folder,'plumed.dat')))
        os.system("rm {}".format(os.path.join(folder,'plumed.dat.tmp')))
        os.system("echo 'cd {}' >> submit.sh".format(folder))
        os.system("echo 'sbatch run.slurm' >> submit.sh")
        os.system("echo 'cd ..' >> submit.sh")
        os.system("echo 'cd {}' >> sumhills.sh".format(folder))
        os.system("echo 'plumed sum_hills --hills HILLS --min 0 --max 3.0 --bin 300 --stride 1000' >> sumhills.sh")
        os.system("echo 'cd ..' >> sumhills.sh")

        system = unit_system.repeat(repeat)
        ase.io.write(os.path.join(folder,'cubic.lmp'),system,format='lammps-data')
        atypes = system.get_atomic_numbers()
        with open(os.path.join(folder,'type.raw'),'w') as file:
            for t in atypes:
                file.write(str(t-1)+'\n')
    
