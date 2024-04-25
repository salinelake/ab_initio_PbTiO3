import os
import random
import string
import numpy as np
import dpdata
import ase
import ase.io
s=string.ascii_lowercase + string.digits


################################
dataset_path = [ 
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/cubic/300K/deepmd",
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/cubic/600K/deepmd",
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/cubic/900K/deepmd",
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/tetra/300K/deepmd",
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/tetra/600K/deepmd",
    # "/home/pinchenx/ferro/DPGEN/PTO.init/02.md/tetra/900K/deepmd",
    "/home/pinchenx/ferro/DPGEN/iter.000000/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000000/02.fp/data.004",
    "/home/pinchenx/ferro/DPGEN/iter.000001/02.fp/data.003",
    "/home/pinchenx/ferro/DPGEN/iter.000001/02.fp/data.002",
    "/home/pinchenx/ferro/DPGEN/iter.000001/02.fp/data.005",
    "/home/pinchenx/ferro/DPGEN/iter.000001/02.fp/data.000",
    "/home/pinchenx/ferro/DPGEN/iter.000002/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000002/02.fp/data.004",
    "/home/pinchenx/ferro/DPGEN/iter.000003/02.fp/data.003",
    "/home/pinchenx/ferro/DPGEN/iter.000003/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000003/02.fp/data.000",
    "/home/pinchenx/ferro/DPGEN/iter.000003/02.fp/data.004",
    "/home/pinchenx/ferro/DPGEN/iter.000004/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000004/02.fp/data.004",
    "/home/pinchenx/ferro/DPGEN/iter.000005/02.fp/data.003",
    "/home/pinchenx/ferro/DPGEN/iter.000005/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000005/02.fp/data.002",
    "/home/pinchenx/ferro/DPGEN/iter.000005/02.fp/data.005",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.003",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.001",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.002",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.005",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.000",
    "/home/pinchenx/ferro/DPGEN/iter.000006/02.fp/data.004",
    "/home/pinchenx/ferro/DPGEN/iter.000007/02.fp/data.003",
    "/home/pinchenx/ferro/DPGEN/iter.000007/02.fp/data.000",
    "/home/pinchenx/ferro/DPGEN/iter.000008/02.fp/data.002",
    "/home/pinchenx/ferro/DPGEN/iter.000008/02.fp/data.005",
    ]
################################
# QE options
################################
def complete_win_in(dir_win, conf):
  '''
  append cell and coordinates information to the input of Wannier90.
  Args:
    dir_win: directory of the input of Wannier90
    conf(ase.atoms): The configuration.
  '''
  symbols = conf.get_chemical_symbols()
  coords = conf.get_positions().tolist()
  # coords = conf.get_scaled_positions().tolist()
  with open(dir_win,"a") as file:
    file.write("begin unit_cell_cart\n")
    file.write("Ang\n")
    for cell_vec in conf.get_cell().tolist():
      file.write(' '.join([str(x) for x in cell_vec]))
      file.write('\n')
    file.write("end unit_cell_cart\n")

    file.write("begin atoms_cart\n")
    for symbol, coord in zip(symbols, coords):
      file.write(symbol+' ')
      file.write(' '.join([str(x) for x in coord]))
      file.write('\n')
    file.write("end atoms_cart\n")

PPs = {
  'O': 'O_ONCV_PBE-1.2.upf',
  'Pb':'Pb_ONCV_PBE-1.2.upf',
  'Ti':'Ti_ONCV_PBE-1.2.upf'}

kmesh = '/home/pinchenx/data.gpfs/softwares/wannier90-3.1.0/utility/kmesh.pl'
input_scf = {
  'calculation':'scf',
  'outdir':'./out',
  'pseudo_dir':'/home/pinchenx/data.gpfs/softwares/QuantumEspresso/pseudos',
  'prefix':'pto',
  # 'disk_io': 'none',
  # 'pseudo_dir':'/global/homes/p/pinchenx/cfs/pseudos',
  'system':{
    'ecutwfc': 150,
    'input_dft': 'SCAN',
    'nosym':  True
    },
  'electrons':{
    'conv_thr': 1e-7,
    },
} 
kpts_scf = [1,1,1]
input_nscf = input_scf.copy()
input_nscf['calculation'] = 'nscf'
kpts_nscf = [2,2,2]


for data_folder in dataset_path:
  multi_systems = dpdata.LabeledSystem(data_folder,fmt = 'deepmd/raw')
  nframes = multi_systems['cells'].shape[0]
  ## make directory
  ss = data_folder.split('/')
  if 'init' in data_folder:
    out_dir_base = '/'.join(ss[-5:-1])
  else:
    out_dir_base = '/'.join(ss[-3::2])
  out_dir_base = os.path.join(out_dir_base, 'wannier')
  if not os.path.exists(out_dir_base):
    os.makedirs(out_dir_base, exist_ok=True)
  for idx, system in enumerate(multi_systems):
    ## load atomic system from deepmd dataset
    atom_system = ase.Atoms(
            symbols = np.array(system['atom_names'])[system['atom_types']], 
            positions = system['coords'][0],
            cell=system['cells'][0],
            pbc=[1,1,1])
    ## update temporary out path
    rd_name = ''.join(random.sample(s,12))
    folder = '/tmp/{}'.format(rd_name)
    input_scf['outdir'] = folder
    input_nscf['outdir'] = folder
    ################################
    out_dir = os.path.join(out_dir_base,'task.{:06d}'.format(idx))
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    ## generate scf input
    ase.io.write(os.path.join(out_dir,'scf.in'), atom_system, format='espresso-in',
      input_data=input_scf, pseudopotentials=PPs, kpts=kpts_scf)

    ## generate nscf input, handled the Kpoints
    ase.io.write('nscf.in.base', atom_system, format='espresso-in',
      input_data=input_nscf, pseudopotentials=PPs)
    os.system("sed '/K_POINTS/d' nscf.in.base > nscf.in")
    os.system("{} {} {} {} >> nscf.in".format(kmesh,kpts_nscf[0],kpts_nscf[1],kpts_nscf[2]))
    os.system("mv nscf.in {}".format(out_dir))
    os.system("rm nscf.in.base")

    ## generate wannier90 inputs
    os.system("sed 's/REPLACE0/{}/' ./template/PTO.pw2wan >> PTO.pw2wan".format(rd_name))
    os.system("mv PTO.pw2wan {}".format(out_dir))
    os.system("cp ./template/PTO.win {}".format(out_dir))
    complete_win_in(os.path.join(out_dir, 'PTO.win'), atom_system)
    os.system("sh try_nnkp.sh {}".format(out_dir))

    ## generate slurm scripts
    os.system("cp ./template/submit.slurm {}".format(out_dir))

