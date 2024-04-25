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
input_scf = {
                        'calculation':'vc-relax',
                        'outdir':'./',
                        'pseudo_dir':'/home/pinchenx/data.gpfs/softwares/QuantumEspresso/pseudos',
                        'etot_conv_thr' : 1e-6,
                        'forc_conv_thr' : 1e-5,
                        'disk_io': 'None',
                        'system':{
                            'ecutwfc': 150,
                            'nosym': True,
                            'input_dft': 'SCAN',
                            'occupations'     : 'fixed',
                         },
                        'electrons':{
                           'conv_thr':1e-8,
                         },
} 
  
kpts = 4
molecule = Atoms(
    symbols=['Pb', 'Ti', 'O','O','O'], 
    scaled_positions=[(0., 0., 0.), (0.5, 0.5, 0.5), (0.5, 0.5, 0.),(0.0, 0.5, 0.5),(0.5, 0.00, 0.5)],
    cell=[3.95,3.95,3.95],
    pbc=[1,1,1],)

ase.io.write('relax.in', molecule, format='espresso-in',
    input_data=input_scf, 
    pseudopotentials=pseudopotentials,
    kpts=[kpts,kpts,kpts],
    koffset=[0,0,0])
 