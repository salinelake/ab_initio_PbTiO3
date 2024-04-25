import os
import dpdata
import numpy as np
import ase
import ase.io
from pto import PTO
import warnings
## has been tested with dpdata

def ase_fromdir(data_dir, outname):
    traj = []
    for folder in os.listdir(data_dir):
        fp_data = os.path.join(data_dir, folder)
        fp_data = os.path.join(fp_data, outname)
        flag = True
        index = 0
        while flag:
            try:
                frame = ase.io.read(fp_data,format='espresso-out',index=index)
                traj.append(frame)
                index += 1
            except:
                flag = False
    return traj

# def wan_fromdir(data_dir, outname):
#     traj = []
#     for folder in os.listdir(data_dir):
#         fp_data = os.path.join(data_dir, folder)
#         fp_data = os.path.join(fp_data, outname)
#         try:
#             frame = ase.io.read(fp_data,format='wout')
#             wan_frame = PTO(frame)
#             traj.append(wan_frame)
#         except:
#             warnings.warn('{} not loaded'.format(fp_data))
#     return traj

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

def get_atomic_dipole(traj):
    return [frame.get_dipole_moment().flatten() for frame in traj]

def get_global_dipole(traj):
    return [frame.get_global_dipole_moment() for frame in traj]

def get_atomic_wc(traj, wc_atype=None):
    atomic_wc_3D = [frame.get_rwcs_displacement(atom_type=wc_atype, sort = True) for frame in traj]
    atomic_wc = [wc_3D.flatten() for wc_3D in atomic_wc_3D]
    global_wc = [wc_3D.sum(0) for wc_3D in atomic_wc_3D]
    return atomic_wc, global_wc

def get_energy(traj):
    return [frame.get_potential_energy() for frame in traj]

def get_cell(traj):
    return [np.array(frame.get_cell()).flatten() for frame in traj]

def get_coords(traj):
    return [frame.get_positions().flatten() for frame in traj]

def get_forces(traj):
    return [frame.get_forces().flatten() for frame in traj]

def get_type_map(traj, type_map):
    symbols = traj[0].get_chemical_symbols()
    atom_type = np.zeros_like(symbols,dtype=int)
    for itype, s in enumerate(type_map):
        atom_type[np.array(symbols) == s] += itype
    return atom_type

def get_virial(traj):
    virial = []
    for frame in traj:
        s = - frame.get_stress()  ## eV/A, sign opposite to ase convention
        stress = np.array([
            [s[0],s[5],s[4]],
            [s[5],s[1],s[3]],
            [s[4],s[3],s[2]]
            ])
        virial.append(stress.flatten() * np.linalg.det(frame.get_cell()))
    return virial


def collect_wc(data_dir, scf_outname, wan_outname, outdir, dp_version=2, type_map=['O','Pb','Ti'], wc_atype=None):
    print('loading wannier data from ', data_dir)
    scf_traj, wan_traj = get_all_traj(data_dir, scf_outname, wan_outname)
    nframes = len(wan_traj)
    print('#data = ', nframes)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ###### process scf data first
    ## energy
    energy = np.array(get_energy(scf_traj))
    np.savetxt(os.path.join(outdir,'energy.raw'), energy.reshape(nframes,-1))
    ## supercell
    box = np.array(get_cell(scf_traj))
    np.savetxt(os.path.join(outdir,'box.raw'), box.reshape(nframes,-1))
    ## coordinates
    coord = np.array(get_coords(wan_traj))
    np.savetxt(os.path.join(outdir,'coord.raw'), coord.reshape(nframes,-1))
    ## atom mapping
    atom_type  = get_type_map(wan_traj, type_map)
    np.savetxt(os.path.join(outdir,'type.raw'), np.array(atom_type).reshape(-1,1),fmt = '%d')
    np.savetxt(os.path.join(outdir,'type_map.raw'), np.array(type_map).reshape(-1,1),fmt = '%s')
    ## export to deepmd npy format
    dp_traj = dpdata.LabeledSystem(outdir,fmt='deepmd/raw')
    dp_traj.to_deepmd_raw(outdir)
    dp_traj.to_deepmd_npy(outdir)

    ###### process wannier data
    ## atomic dipole
    atomic_wc, global_wc = get_atomic_wc(wan_traj, wc_atype)
    atomic_wc = np.array(atomic_wc)
    global_wc = np.array(global_wc)
    if dp_version == 1:
        ## export to dataset for DPMD 1.*
        np.savetxt(os.path.join(outdir,'dipole.raw'), atomic_wc.reshape(nframes,-1))
        np.savetxt(os.path.join(outdir,'global_dipole.raw'), global_wc.reshape(nframes,-1))
        np.save(os.path.join(outdir,'set.000/dipole.npy'), 
            atomic_wc.reshape(nframes,-1).astype('float32'))

    else:
        ## export to dataset for DPMD 2.*
        np.savetxt(os.path.join(outdir,'atomic_dipole.raw'), atomic_wc.reshape(nframes,-1))
        np.savetxt(os.path.join(outdir,'dipole.raw'), global_wc.reshape(nframes,-1))
        np.save(os.path.join(outdir,'set.000/atomic_dipole.npy'),
            atomic_wc.reshape(nframes,-1).astype('float32'))
        np.save(os.path.join(outdir,'set.000/dipole.npy'),
            global_wc.reshape(nframes,-1).astype('float32'))
    return

if __name__ == "__main__":
        folder = '/home/pinchenx/ferro/DeepWannier/PTO.test/T900'
        data_dir = os.path.join(folder,'wannier')
        outdir = os.path.join(folder,'deep_test')
        collect_wc(data_dir, 'scf.out', 'PTO.wout', outdir, wc_atype=['O','Pb'])

