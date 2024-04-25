import numpy as np
import os
import ase
import ase.io
from ase import Atoms
# from ase.calculators.emt import EMT
# from ase.calculators.espresso import Espresso
# from ase.calculators.vasp import Vasp


pseudopotentials = {'Pb': 'Pb_ONCV_PBE-1.2.upf',
                                        'Ti': 'Ti_ONCV_PBE-1.2.upf',
                                        'O': 'O_ONCV_PBE-1.2.upf'}
# pseudopotentials = {'Pb': 'Pb_ONCV_PBE_sr.upf',
#                     'Ti': 'Ti_ONCV_PBE_sr.upf',
#                     'O': 'O_ONCV_PBE_sr.upf'}

input_scf = {
                        'calculation':'scf',
                        'outdir':'./',
                        'pseudo_dir':'/home/pinchenx/data.gpfs/softwares/QuantumEspresso/q-e-qe-6.4.1/pseudo',
                        # 'pseudo_dir':'/home/pinchenx/data.gpfs/softwares/QuantumEspresso/ONCVPSP/abinit',

                        # 'lberry': True,     ## perform berry phase calculatoin
                        # 'gdir':3,            ## direction of the k-point strings in reciprocal space. Allowed values: 1, 2, 3
                        # 'nppstr':7,            ## number of k-points to be calculated along each symmetry-reduced string
                        'system':{
                            'ecutwfc': 80,
                            # 'ecutrho': 560,
                            'input_dft': 'SCAN',
                            # 'nbnd'            : 25,
                            'occupations'     : 'fixed',
                         },
                        'electrons':{
                            # 'mixing_beta': 0.3,
                            'conv_thr'    :1e-8,
                            #    'startingwfc': 'atomic+random',
                            #    'startingpot': 'atomic',
                         },
} 
input_berry = {
                        'calculation':'nscf',
                        'outdir':'./',
                        'pseudo_dir':'/home/pinchenx/data.gpfs/softwares/QuantumEspresso/q-e-qe-6.4.1/pseudo',
                        'lberry': True,     ## perform berry phase calculatoin
                        'gdir':3,            ## direction of the k-point strings in reciprocal space. Allowed values: 1, 2, 3
                        'nppstr':7,            ## number of k-points to be calculated along each symmetry-reduced string
                        'system':{
                            'ecutwfc': 80,
                            # 'ecutrho': 560,
                            'input_dft': 'SCAN',
                            # 'nbnd'            : 22,    ## number of electrons/2
                            'occupations'     : 'fixed',
                         },
                        'electrons':{
                            'mixing_beta': 0.3,
                            # 'conv_thr'    :1e-5,
                            #    'startingwfc': 'atomic+random',
                            #    'startingpot': 'atomic',
                         },
} 


# l = 4.016
# d_list = 0.0025 * (np.arange(21)-10)
# d_list = 0.0025 * np.arange(10)
# dO=0.025

os.system("echo '#batch submit' > batch_submit.sh")
l_list = np.arange(11)*0.02 + 3.82

for ecutwfc in [80]:
    kpts = 6
    input_scf['system']['ecutwfc'] = ecutwfc
    for idx,l in enumerate(l_list):
        molecule = Atoms(
            symbols=['Pb', 'Ti', 'O','O','O'], 
            scaled_positions=[(0., 0., 0.), (0.5, 0.5, 0.5), (0.5, 0.5, 0),(0, 0.5, 0.5),(0.5, 0, 0.5)],
            cell=[l,l,l],
            pbc=[1,1,1],)

        prefix = 'cubic_kpts{}_kcut{}'.format(kpts, int(ecutwfc) )
        folder = './'+prefix
        if not os.path.exists(folder):
            os.mkdir(folder)
        out_dir = folder + '/{}/'.format(idx)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        ase.io.write( out_dir + 'qe.in', molecule, format='espresso-in',
            input_data=input_scf, 
            pseudopotentials=pseudopotentials,
            tstress=True, tprnfor=True,    # kwargs added to parameters
            kpts=[kpts,kpts,kpts],
            koffset=[0,0,0])

        ase.io.write( out_dir + 'berry.in', molecule, format='espresso-in',
            input_data=input_berry, 
            pseudopotentials=pseudopotentials,
            kpts=[kpts,kpts,kpts],
            koffset=[0,0,0])

    os.system("sed 's/REPLACE1/{}/g' submit.base > submit.slurm".format(idx))
    os.system('mv submit.slurm {}/'.format(folder))
    os.system("echo cd {} >> batch_submit.sh".format(folder))
    os.system("echo sbatch submit.slurm >> batch_submit.sh")
    os.system("echo cd .. >> batch_submit.sh")