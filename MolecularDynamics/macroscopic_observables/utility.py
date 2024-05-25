import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.linestyle'] = 'dashed'
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


def block_analysis(x, nblock):
    nframe = x.shape[0]
    if nframe < nblock:
        raise ValueError('nframe <= nblock')
    block_size = int(x.shape[0] / nblock)
    x_mean = x.mean()
    x_block = [x[i*block_size:(i+1)*block_size].mean() for i in range(nblock)] 
    block_mean = np.array(x_block).mean()
    block_error = np.array(x_block).std() / (nblock -1)**0.5
    return block_mean, block_error

def polarizability(x, temp,N):
    beta = 1/kb/temp 
    variance = (x**2).mean() - (x.mean())**2
    return beta * N * variance  #eA^2/eV  

def simultaneous_read(folder, throw=400,phase='t',restrain=False, dp_phase_boundary=1):
    thermo_path =  os.path.join(folder,'pto.log') 
    colvar_path = os.path.join(folder,'COLVAR')
    ## get rid of the relaxation trajectory
    skip_header = 0
    skip_flag = 2
    with open(thermo_path) as file:
        while skip_flag > 0:
            skip_header += 1
            x = file.readline().split()
            if len(x) > 0 and x[0] == 'Step':
                skip_flag -= 1
    thermo = np.genfromtxt(thermo_path, skip_header=skip_header+throw,skip_footer=50,invalid_raise=True)
    colvar = np.genfromtxt(colvar_path, skip_header=1+throw,skip_footer=1,invalid_raise=True)
    ## syncing
    nf1 = thermo.shape[0]
    nf2 = colvar.shape[0]
    nframe = np.minimum(nf1,nf2)
    thermo = thermo[:nframe]
    colvar = colvar[:nframe]
    dp_abs = colvar[:,4]  ## dipole per cell
    ##  make sure thermo and colvar are sync
    for i in range(10):
        test_idx = np.random.randint(nframe)
        assert np.abs(colvar[test_idx,-4]*1000 - thermo[test_idx,6]) < 1
    #  purify the phase
    if restrain:
        if phase == 't':
            phase_filter = dp_abs > (np.ones_like(dp_abs)*dp_phase_boundary)
        else:
            phase_filter = dp_abs <= (np.ones_like(dp_abs)*dp_phase_boundary)
        phase_percentile = phase_filter.astype(int).sum() / nframe
        print('total_frame={}, phase_percentile:{:.2f}%'.format(nframe,phase_percentile*100))
        thermo = thermo[phase_filter]
        colvar = colvar[phase_filter]
        nframe = thermo.shape[0]

    ##
    cell = thermo[:,-3:]
    dipole = colvar[:,1:4]
    c_idx = np.argmax(cell,1)
    cell_reversed = (c_idx != 0).astype(np.float).sum()/ nframe
    print('cell_reversed:{:.2f}%'.format(cell_reversed*100))
    a_idx = (c_idx+1)%3
    b_idx = (c_idx+2)%3

    # dpx=dipole[:,0]
    # dpy=dipole[:,1]
    # dpz=dipole[:,2]
    # dpa=dipole[np.arange(nframe),a_idx]
    # dpb=dipole[np.arange(nframe),b_idx]
    # dpc=np.abs(dipole[np.arange(nframe),c_idx])
    
    thermo = {
        ## thermodynamics
        "step":thermo[:,0],
        "time":thermo[:,0]*0.0005, #ps
        "PotEng":thermo[:,2], #eV
        "TotEng":thermo[:,4], #eV
        ## geometries
        "Volume":thermo[:,6],   #A^3
        "Cell": cell,
        "c":cell[np.arange(nframe),c_idx],
        "b":cell[np.arange(nframe),b_idx],
        "a":cell[np.arange(nframe),a_idx],
        ### dipole amplitude
        "dpx":dipole[:,0],
        "dpy":dipole[:,1],
        "dpz":dipole[:,2],
        "dpa":dipole[np.arange(nframe),a_idx],
        "dpb":dipole[np.arange(nframe),b_idx],
        "dpc":np.abs(dipole[np.arange(nframe),c_idx]),
        "dp_abs":colvar[:,4],  ## dipole per cell
        ### susceptibility
        # "chi_x":((dpx*dpx/vol).mean() - dpx.mean()*(dpx/vol).mean())/kb/temp/ epsilon0,
        # "chi_y":((dpy*dpy/vol).mean() - dpy.mean()*(dpy/vol).mean())/kb/temp/ epsilon0,
        # "chi_z":((dpz*dpz/vol).mean() - dpz.mean()*(dpz/vol).mean())/kb/temp/ epsilon0,
        # "chi_a":((dpa*dpa/vol).mean() - dpa.mean()*(dpa/vol).mean())/kb/temp/ epsilon0,
        # "chi_b":((dpb*dpb/vol).mean() - dpb.mean()*(dpb/vol).mean())/kb/temp/ epsilon0,
        # "chi_c":((dpc*dpc/vol).mean() - dpc.mean()*(dpc/vol).mean())/kb/temp/ epsilon0,
    }

    return thermo


def compute_susceptibility(_thermo, temp):
    '''
    thermo["dpx"]: numpy array, x component of the total dipole
    thermo["dpy"]: numpy array, y component of the total dipole
    thermo["dpz"]: numpy array, z component of the total dipole
    '''
    thermo = _thermo.copy()
    vol = thermo["Volume"]
    thermo["chi_x"] = ((thermo["dpx"]**2/vol).mean() - thermo["dpx"].mean()*(thermo["dpx"]/vol).mean())/kb/temp/ epsilon0
    thermo["chi_y"] = ((thermo["dpy"]**2/vol).mean() - thermo["dpy"].mean()*(thermo["dpy"]/vol).mean())/kb/temp/ epsilon0
    thermo["chi_z"] = ((thermo["dpz"]**2/vol).mean() - thermo["dpz"].mean()*(thermo["dpz"]/vol).mean())/kb/temp/ epsilon0
    thermo["chi_a"] = ((thermo["dpa"]**2/vol).mean() - thermo["dpa"].mean()*(thermo["dpa"]/vol).mean())/kb/temp/ epsilon0
    thermo["chi_b"] = ((thermo["dpb"]**2/vol).mean() - thermo["dpb"].mean()*(thermo["dpb"]/vol).mean())/kb/temp/ epsilon0
    thermo["chi_c"] = ((thermo["dpc"]**2/vol).mean() - thermo["dpc"].mean()*(thermo["dpc"]/vol).mean())/kb/temp/ epsilon0
    return thermo