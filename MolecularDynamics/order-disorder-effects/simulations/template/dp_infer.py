import os
import numpy as np
import dpdata
from deepmd.infer import DeepDipole # use it to load your trained model


model = DeepDipole('/home/pinchenx/ferro/DeepWannier/_TRAIN/dp-plumed-global/dipole-compress.pb')

# ss_list = [6,12,18]
supersize = 6
ncell = supersize**3
folder = './'
colvar_path = os.path.join(folder, 'COLVAR')
if os.path.exists(colvar_path):
    print('system {} already processed'.format(folder))
else:
    systems = dpdata.System(os.path.join(folder,'pto.lammpstrj'),fmt = 'lammps/dump')
    print('system {} loaded'.format(folder))
    nframe = systems['cells'].shape[0]
    dipole_list = []
    for i in range(nframe):
        if i%100==0:
            print('processing {}-th frame'.format(i))
        dipole_list.append(model.eval(systems['coords'][i], systems['cells'][i], systems['atom_types']))
    box = systems['cells'][:,[0,1,2],[0,1,2]]
    volume = box[:,0]*box[:,1]*box[:,2]
    dipole = np.concatenate(dipole_list).sum(-2)  ##(nframe,3)
    dipole_abs = ((dipole**2).sum(-1))**0.5
    cv1 = dipole_abs / ncell
    polar_d = dipole_abs / volume
    time = np.arange(cv1.shape[0])*400*0.0005
    colvar = np.concatenate([time[:,None],dipole,cv1[:,None],polar_d[:,None], box, volume[:,None]],axis=-1)
    np.savetxt(colvar_path,colvar,fmt='%.6f',
        header='#! FIELDS time dp.x[eA] dp.y[eA] dp.z[eA] |dp|/ncell[eA] dp_density[e/A^2]  cell_x[A] cell_y[A] cell_z[A] vol[A^3]')

    