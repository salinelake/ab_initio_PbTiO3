import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 12

font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 15,
        }
inset_font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 10,
        }
text_font = {'family': 'serif',
    'color':  'k',
    'weight': 'normal',
    'size': 12,
    }
inset_label_size=12
kb = 8.617333e-5 #eV K-1
e2C = 1.60217662e-19
e2uC = e2C * 1e6
Atcm=1e-8
barA32eV = 6.24150913e-7
calm2eV = 0.0000433641
avogadro = 6.02214086e23
eV2J = 1.60218e-19
epsilon0=8.8541878128e-12 / e2C / 1e10 # e/AV
kjmol2mev=10.364
kjmol2eV =  1.0364e-2  ## eV
dt=0.0005
def get_plumed_paras(folder):
    conf_path = os.path.join(folder,'plumed.dat')
    with open(conf_path,'r') as f:
        for line in f.readlines():
            if 'BIASFACTOR' in line:
                bias = float(line[:-1].split('=')[-1])
            if 'FUNC' in line:
                ncell = int(line[:-1].split('/')[-1])
            if 'PACE' in line:
                pace = int(line[:-1].split('=')[-1])
    return bias, ncell, pace

def get_offset(folder):
    ### get c(t)
    colvar_path  = os.path.join(folder,'COLVAR')
    colvar = np.genfromtxt(colvar_path, skip_footer=1)
    with open(colvar_path,'r') as f:
        info = f.readline()
        info = info.split()
        for idx, item in enumerate(info):
            if item == 'metad.rct':
                idx_rct = idx-2
    rct = colvar[:,idx_rct]
    ### get bias factor gamma
    bias, ncell, pace = get_plumed_paras(folder)
    ## offset for free energy is  gamma/(gamma-1)*c(t)
    return bias/(bias-1)*rct

def get_final_fes(folder,window=[0,5]):
    counter=0
    bias, ncell, pace = get_plumed_paras(folder)
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    while os.path.exists(fes_dir):
        counter +=1
        fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    # T =  counter*dt
    counter -= 1
    print('loading fes_{}.dat from {}'.format(counter, folder))
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    fes = np.loadtxt(fes_dir)
    filter = (fes[:,0]>=window[0])&(fes[:,0]<=window[1])
    offset = get_offset(folder)
    x = fes[:,0][filter]
    F = (fes[:,1][filter] + offset[-1])/ncell*kjmol2mev
    return x, F, counter-1

def get_all_fes(folder, window=[0,5]):
    counter=0
    bias, ncell, pace = get_plumed_paras(folder)
    offset = get_offset(folder)
    fes_list = []   
    offset_list = []
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    while os.path.exists(fes_dir):
        fes = np.loadtxt(fes_dir)
        filter = (fes[:,0]>=window[0])&(fes[:,0]<=window[1])
        fes_list.append(fes[:,1][filter])
        counter +=1
        fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    x = fes[:,0][filter]
    fes_list = fes_list[:-1]  ## the last one is duplicated
    for i in range(counter-1):
        idx = int(np.floor(offset.shape[0]/counter))*(i+1)
        fes_list[i] = (fes_list[i])/ncell*kjmol2mev
        offset_list.append(offset[idx]/ncell*kjmol2mev)
    return x,fes_list, offset_list



    
if __name__=='__main__':
    test_dir = '/home/pinchenx/ferro/DPMD/final-metad1D_press/600K9x9x9'
    print(get_plumed_paras(test_dir))
    print(get_offset(test_dir))
    print(get_final_fes(test_dir))

