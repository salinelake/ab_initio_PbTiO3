import ase
import ase.io
import os

unit_system = ase.io.read('./template/cubic.lmp',format='lammps-data',style='atomic')
# system = ase.io.read('pto.lammpstrj',format='lammps-dump-text')
meV2Kelvin = 1/1000 * 11604.5 


# temp_list = [700, 800, 900, 1000]
# barrier_list = [6,6,6,6] # estimated, unit: meV
temp_list = [300, 600, 820]
barrier_list = [8,6,6] # estimated, unit: meV
bscale = 5
repeat = 3
supersize = repeat * 3
ncell = supersize**3
natoms = ncell * 5
os.system("echo '# batch submit' > submit.sh" )
os.system("echo '# plumed sumhill' > sumhills.sh" )
for temp, barrier in zip(temp_list,barrier_list):
    folder = '{}K{}x{}x{}'.format(temp,supersize,supersize,supersize)
    biasf = int(bscale * barrier * natoms * meV2Kelvin / temp)
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    os.system("cp ./template/in.lammps {}/".format(folder))
    os.system("sed 's/REPLACE0/{}/' ./template/run.slurm > {}".format(temp,os.path.join(folder,'run.slurm')))

    # os.system("sed 's/REPLACE_ncell/{}/' ./template/plumed.dat > {}".format(ncell,os.path.join(folder,'plumed.dat')))
    os.system("sed 's/REPLACE_bias/{}/' ./template/plumed_wider.dat > {}".format(biasf, os.path.join(folder,'plumed.dat.tmp')))
    os.system("sed 's/REPLACE_temp/{}/' {} > {}".format(temp,os.path.join(folder,'plumed.dat.tmp'),os.path.join(folder,'plumed.dat')))
    os.system("rm {}".format(os.path.join(folder,'plumed.dat.tmp')))
    
    os.system("echo 'cd {}' >> submit.sh".format(folder))
    os.system("echo 'sbatch run.slurm' >> submit.sh")
    os.system("echo 'cd ..' >> submit.sh")
    os.system("echo 'cd {}' >> sumhills.sh".format(folder))
    os.system("echo 'plumed sum_hills --hills HILLS --min -0.02,-0.02,-0.02 --max 0.035.0,035.0.065 --bin 100,100,200 --stride 1000' >> sumhills.sh")
    os.system("echo 'cd ..' >> sumhills.sh")

    system = unit_system.repeat(repeat)
    ase.io.write(os.path.join(folder,'cubic.lmp'),system,format='lammps-data')
    atypes = system.get_atomic_numbers()
    with open(os.path.join(folder,'type.raw'),'w') as file:
        for t in atypes:
            file.write(str(t-1)+'\n')
    
