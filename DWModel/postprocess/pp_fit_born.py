from fse.systems import perovskite, PTO
import os
import ase
import ase.io
from numpy.linalg import lstsq, norm
import numpy as np
import torch as th
from matplotlib import pyplot as plt

pto_factory = perovskite()
def get_all_traj(data_dir, scf_outname, wan_outname):
    scf_traj = []
    wan_traj = []
    for folder in os.listdir(data_dir):
        data_folder = os.path.join(data_dir, folder)
        scf_path = os.path.join(data_folder, scf_outname)
        wan_path = os.path.join(data_folder, wan_outname)
        try:
            scf_frame = ase.io.read(scf_path,format='espresso-out',index=-1)
            _wan_frame = ase.io.read(wan_path,format='wout')
            wan_frame = PTO(_wan_frame)            
            scf_traj.append(scf_frame)
            wan_traj.append(wan_frame)
        except:
            pass
            # print('{} not loaded'.format(data_folder))
    if len(scf_traj) != len(wan_traj):
        raise ValueError('#scf_data={} != #wan_data={}'.format(len(scf_traj),len(wan_traj)))
    return scf_traj, wan_traj

def fit_born(atoms_list, target_dipole_set):
    target = th.from_numpy(target_dipole_set).type(th.float)
    chg_A = th.tensor([4.0],requires_grad=True)
    chg_B = th.tensor([4.0])
    chg_Ol = th.tensor([-5.5],requires_grad=True)
    chg_Ot = th.tensor([-2.5],requires_grad=True)
    born_charges = [chg_A, chg_B, chg_Ol, chg_Ot]
    optimizer = th.optim.Adam(born_charges, lr=0.05)
    for iters in range(1000):
        loss = 0
        for atoms, target in zip(atoms_list, target):
            lattice = pto_factory.get_effective_lattice([3,3,3],atoms,'Ti')
            pred_local_dipole = pto_factory._get_lattice_dipole_born_torch(atoms, lattice, born_charges)
            pred_global_dipole = pred_local_dipole.reshape(-1,3).sum(0)
            loss += ((pred_global_dipole - target)**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        chg_B = - chg_Ol - 2 * chg_Ot - chg_A
        print('step: {}, Born charges: {}, loss:{}'.format(iters, born_charges, (loss/nframes)**0.5))
    return born_charges

if __name__ == "__main__":
    # data_set = [
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/PTO.init/02.md/cubic/300K/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/PTO.init/02.md/tetra/300K/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/PTO.init/02.md/cubic/600K/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/PTO.init/02.md/tetra/600K/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000000/data.001/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000000/data.004/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000001/data.003/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000003/data.004/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000002/data.004/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000004/data.004/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000005/data.005/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000006/data.005/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000007/data.003/wannier",
    #             "/home/pinchenx/ferro_scratch/PTO/DeepWannier/dataset/iter.000008/data.005/wannier",
    #             ]
    # atoms_list = []
    # wan_list = []
    # for data_dir in data_set:
    #     print('loading wannier data from ', data_dir)
    #     scf_traj, wan_traj = get_all_traj(data_dir, 'scf.out', 'PTO.wout')
    #     print('loaded {} configurations'.format(len(wan_traj)))
    #     atoms_list += scf_traj
    #     wan_list += wan_traj
    # nframes = len(wan_list)
    # target_dipole_set = [wan.get_global_dipole_moment().flatten() for wan in wan_list]
    # target_dipole_set = np.array(target_dipole_set).reshape(-1,3)
    # # born_charges = fit_born(atoms_list, target_dipole_set)
    # born_charges = [3.7140, 5.4879, -3.3551, -2.9234]
    # pred_dipole_set =[]
    # for atoms in atoms_list:
    #     lattice = pto_factory.get_effective_lattice([3,3,3],atoms,'Ti')
    #     pred_local_dipole = pto_factory.get_lattice_dipole_born(atoms, lattice, born_charges)
    #     pred_dipole_set.append(pred_local_dipole.reshape(-1,3).sum(0))
    # pred_dipole_set = np.array(pred_dipole_set).reshape(-1,3)
    ## plot
    if os.path.exists('./born/pred_dipole.npy'):
        pred_dipole_set = np.load('./born/pred_dipole.npy')
        target_dipole_set = np.load('./born/target_dipole.npy')
    else:
        np.save('./born/pred_dipole.npy', pred_dipole_set)
        np.save('./born/target_dipole.npy', target_dipole_set)
    D_born = pred_dipole_set
    D = target_dipole_set

    std=((((D_born-D)**2).sum()/D.shape[0])**0.5)

    ###
    norm_D  = norm(D,axis=-1)
    diff_D = norm(D_born-D,axis=-1)
    filter = (norm_D < 5) | (norm_D>95)
    print('total data:', norm_D.shape[0])
    print('std:',std)
    print('extreme data:', norm_D[filter].shape[0])
    print('error>2std:', (diff_D > std*2).sum())

    # print(((diff_D[filter]**2).sum()/norm_D[filter].shape[0])**0.5)

    ####
    fig, ax = plt.subplots(2,2)
    ax[0,0].scatter(norm_D, diff_D,s=np.ones_like(norm_D)*1)
    ax[0,1].scatter(D[:,0], D_born[:,0]-D[:,0])
    ax[1,0].scatter(D[:,1], D_born[:,1]-D[:,1])
    ax[1,1].scatter(D[:,2], D_born[:,2]-D[:,2])
    plt.tight_layout()
    plt.savefig('./born/born_predicted_.png', dpi=300)
