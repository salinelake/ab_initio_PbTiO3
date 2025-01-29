import numpy as np
import ase
import ase.io
import os
from matplotlib import pyplot as plt
from fse.systems import perovskite as perovskite
from fse.utils import remap
from deepmd.infer import DeepDipole # use it to load your trained model
dp_model = DeepDipole('/tigress/pinchenx/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb')
## make output folder
folder = './'
output_folder = os.path.join(folder, 'figures')
if os.path.exists(output_folder) is False:
    os.mkdir(output_folder)
## read reference configuration
pto_factory = perovskite(['Pb','Ti','O'])  ## O->H Pb->He Ti->Li
supercell = [512,16,16]
ref = ase.io.read('conf.lmp',  format='lammps-data', style='atomic')
syms_new = remap(ref.get_atomic_numbers(), [1,2,3],[8,82,22])
ref.set_atomic_numbers(syms_new)
lattice = pto_factory.get_effective_lattice(supercell, ref, central_element='Ti')
atypes = remap(ref.get_atomic_numbers(), [8,82,22],[0,1,2])

## read trajectories
for start_time in [0, 50, 100, 150]:
    f = open('pto{:d}ps.lammpstrj'.format(start_time))
    lines = f.readlines()
    nframes = 51
    natoms = 512*16*16*5
    frame_llines = len(lines)//nframes
    for idx in range(nframes-1):
        frame_lines = lines[idx*frame_llines:(idx+1)*frame_llines]
        cell_x = frame_lines[5].split()
        cell_x = float(cell_x[1]) - float(cell_x[0])
        cell_y = frame_lines[6].split()
        cell_y = float(cell_y[1]) - float(cell_y[0])
        cell_z = frame_lines[7].split()
        cell_z = float(cell_z[1]) - float(cell_z[0])
        syms = ['O']*natoms
        positions = np.zeros((natoms, 3))
        for line in frame_lines[9:]:
            atom_id, sym, x, y, z = line.split()
            if sym == '1':
                syms[int(atom_id)-1] = 'O'
            elif sym == '2':
                syms[int(atom_id)-1] = 'Pb'
            elif sym == '3':
                syms[int(atom_id)-1] = 'Ti'
            else:
                raise ValueError('Unknown symbol: ', sym)
            positions[int(atom_id)-1] = np.array([float(x), float(y), float(z)])
        atoms = ase.Atoms(symbols=syms, positions=positions, cell=(cell_x, cell_y, cell_z),pbc=True)
        print('loaded frame', start_time+idx)
        print(atoms)
        lattice_dipole = pto_factory.get_lattice_dipole( dp_model, atypes, atoms, lattice)
        np.save(os.path.join(output_folder,'dipole{}.npy'.format(start_time+idx)), lattice_dipole)
        spinz_xz = lattice_dipole[:,:,:,-1].mean(1)        
        spinz_xy = lattice_dipole[:,:,:,-1].mean(2)
        spinz_yz = lattice_dipole[:,:,:,-1].mean(0)


        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8,6))
        ax1.imshow(spinz_xz.transpose())
        ax1.set_title('(XZ)',loc='left')
        ax2.imshow(spinz_yz.transpose())
        ax2.set_title('(YZ)',loc='left')
        ax3.imshow(spinz_xy.transpose())
        ax3.set_title('(XY)',loc='left')
        plt.tight_layout()
        fig.savefig(os.path.join(output_folder,'f{}.png'.format(start_time+idx)))
        plt.close(fig)

