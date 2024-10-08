import os
import numpy as np
import dpdata
import sys
import ase 
import ase.io
from fse.systems import perovskite
from deepmd.infer import DeepDipole # use it to load your trained model

fidx=int(sys.argv[1])
ss=int(sys.argv[2])
folder = '.'
model = DeepDipole('/home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb')
pto_factory = perovskite(ABO3=['He','Li','H'], born_charges=[3.7140, 5.4879, -3.3551, -2.9234])
print('Loading')
ref_atom = ase.io.read(os.path.join(folder,'conf.lmp' ),style='atomic', format = 'lammps-data' )
ase_atoms = ase.io.read(os.path.join(folder,'pto{:d}.lammpstrj'.format(fidx)),format = 'lammps-dump-text',index=':')
print('Finish Loading')
lattice = pto_factory.get_effective_lattice([ss,ss,ss], ref_atom, central_element='Li')
print('Found lattice')
print('system {} loaded -- {} frames'.format(folder, len(ase_atoms)))
atypes = ref_atom.get_atomic_numbers()-1
dipole_list = []
for idx in range(len(ase_atoms)):
    dipole = pto_factory.get_lattice_dipole( model, atypes, ase_atoms[idx], lattice )
    dipole_list.append(dipole)
dipole_list = np.array(dipole_list)
np.save(os.path.join(folder, 'dipole{:d}.npy'.format(fidx)), dipole_list)
 