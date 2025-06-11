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
# mpl.rcParams['text.usetex'] = True

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
    'size': 15,
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

# def get_fes(folder):
#     counter=0
#     fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
#     while os.path.exists(fes_dir):
#         counter += 1
#         fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
#     fes = np.loadtxt(os.path.join(folder,'fes_{}.dat'.format(counter-1)))
#     return fes

# def get_offset(folder):
#     colvar_path  = os.path.join(folder,'COLVAR')
#     colvar = np.genfromtxt(colvar_path, skip_footer=1)
#     offset = colvar[-1,12]
#     return offset
def get_plumed_paras(folder):
    conf_path = os.path.join(folder,'plumed.dat')
    with open(conf_path,'r') as f:
        for line in f.readlines():
            if 'BIASFACTOR' in line:
                bias = float(line[:-1].split('=')[-1])
            # if 'FUNC' in line:
            #     ncell = int(line[:-1].split('/')[-1])
            if 'PACE' in line:
                pace = int(line[:-1].split('=')[-1])
    return bias, pace
    
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
    bias, pace = get_plumed_paras(folder)
    ## offset for free energy is  gamma/(gamma-1)*c(t)
    return bias/(bias-1)*rct


def get_all_fes(folder, window=[0,1], ncell=1):
    counter=0
    bias, pace = get_plumed_paras(folder)
    offset = get_offset(folder)
    fes_list = []   
    offset_list = []
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    while os.path.exists(fes_dir):
        fes = np.loadtxt(fes_dir)
        filter = (fes[:,0]>=window[0])&(fes[:,0]<=window[1])
        fes_list.append(fes[:,3][filter])
        counter +=1
        fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    polar = fes[:,:3][filter]


    fes_list = fes_list[:-1]  ## the last one is duplicated
    for i in range(counter-1):
        idx = int(np.floor(offset.shape[0]/counter))*(i+1)
        fes_list[i] = (fes_list[i])/ncell*kjmol2mev
        offset_list.append(offset[idx]/ncell*kjmol2mev)
    return polar,fes_list, offset_list

def get_final_fes(folder, ncell=1):
    counter=0
    bias, pace = get_plumed_paras(folder)
    offset = get_offset(folder)
    fes_list = []   
    offset_list = []
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    while os.path.exists(fes_dir):
        counter +=1
        fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter-1))
    fes = np.loadtxt(fes_dir)
    # filter = (fes[:,0]>=window[0])&(fes[:,0]<=window[1])
    F = (fes[:,3] - offset[-1])/ncell*kjmol2mev
    polar = fes[:,:3]
    return polar,F


def get_projected_fes_100(polar,fes):
    proj = (polar[:,0] == 0)
    ndim = int(proj.sum()**0.5)
    assert ndim**2 == int(proj.sum()), 'projected data can not be transformed to a square matrix'
    px, py, pz = [ polar[:,ii][proj].reshape(ndim,ndim) for ii in range(3)]
    F = fes[proj].reshape(ndim,ndim)
    proj_reg = (pz>=py)
    F[~proj_reg] *= 0
    F = F + F.transpose() - np.diag(np.diag(F))
    return px,py,pz,F

def get_projected_fes_110(polar,fes ):
    proj_110 = (polar[:,0] == polar[:,1])
    proj_011 = (polar[:,1] == polar[:,2])
    ndim = int(proj_110.sum()**0.5)
    assert ndim**2 == int(proj_110.sum()), 'projected data can not be transformed to a square matrix'
    px, py, pz = [ polar[:,ii][proj_110].reshape(ndim,ndim) for ii in range(3)]
    F = fes[proj_110].reshape(ndim,ndim)
    proj_reg = (pz>=py)
    F[~proj_reg] *= 0

    F_aux = fes[proj_011].reshape(ndim,ndim)
    proj_reg = (pz>px)
    F_aux[~proj_reg] *= 0
    F = F + F_aux.transpose() 
    return px,py,pz,F

# def get_complete_fes(folder):
#     fes = get_fes(folder)
#     px = fes[:,0]
#     proj = (px == 0) 
#     ndim = int(proj.sum()**0.5)
#     px = fes[:,0][proj].reshape(ndim,ndim)
#     py = fes[:,1][proj].reshape(ndim,ndim)
#     pz = fes[:,2][proj].reshape(ndim,ndim)
#     F = fes[:,3][proj].reshape(ndim,ndim) - get_offset(folder)
#     F = F * kjmol2mev / ncell
#     proj_reg = (pz>=py)
#     F[~proj_reg] *= 0
#     F = F + F.transpose() - np.diag(np.diag(F))
#     return px,py,pz,F




'''   
Landau Model
'''
def _get_landau_vars(px,py,pz):
    x0 = np.ones_like(px)
    x1 = 0.5 * (px**2 + py**2 + pz**2)
    x2 = (px**2 + py**2 + pz**2)**2
    x3 = (px**4 + py**4 + pz**4)
    x4 = (px**2 + py**2 + pz**2)**3
    x5 = (px**6 + py**6 + pz**6)
    x6 = px**4*(py**2+pz**2) + py**4*(pz**2+px**2) + pz**4*(px**2+py**2)
    M = np.array([x0,x1,x2,x3,x4,x5,x6]).reshape(7,-1).transpose()
    # M = np.array([x0,x1,x2,x3,x4,x5]).reshape(6,-1).transpose()
    return M

def get_landau_vars(px,py,pz):
    X = [np.ones_like(px)]
    X.append(px**2 + py**2 + pz**2)  # a1
    X.append(px**4 + py**4 + pz**4)  # a11
    X.append(px**2*py**2 + px**2*pz**2 + py**2*pz**2) #a12
    X.append(px**6 + py**6 + pz**6)  #a111
    X.append(px**4*(py**2+pz**2) + py**4*(pz**2+px**2) + pz**4*(px**2+py**2)) # a112
    X.append(px**2*py**2*pz**2)  #a123
    # X.append(px**8 + py**8 + pz**8)  #a1111
    # X.append(px**6*(py**2+pz**2) + py**6*(pz**2+px**2) + pz**6*(px**2+py**2)) # a1112
    # X.append(px**4*py**4 + px**4*pz**4 + py**4*pz**4) #a1122
    # X.append(px**4*py**2*pz**2+px**2*py**4*pz**2+px**2*py**2*pz**4)  #a1123
    M = np.array(X).reshape(len(X),-1).transpose()
    return M

def plot_ld_coefs(temp_list, coefs,ax,markerstyle='o'):
    mpl.rcParams['lines.marker'] = markerstyle
    ax.plot(temp_list, coefs[:,1], label=r'$\alpha_1\mathcal{P}_{cut}^2$')
    ax.plot(temp_list, coefs[:,2], label=r'$\alpha_{11}\mathcal{P}_{cut}^4$')
    ax.plot(temp_list, coefs[:,3]/4, label=r'$\frac{1}{4}\alpha_{12}\mathcal{P}_{cut}^4$')
    ax.plot(temp_list, coefs[:,4], label=r'$\alpha_{111}\mathcal{P}_{cut}^6$')
    ax.plot(temp_list, coefs[:,5]/3, label=r'$\frac{1}{3}\alpha_{112}\mathcal{P}_{cut}^6$')
    ax.plot(temp_list, coefs[:,6]/12, label=r'$\frac{1}{12}\alpha_{123}\mathcal{P}_{cut}^6$')
    try:
        ax.plot(temp_list, coefs[:,7], label=r'$\alpha_{1111}\mathcal{P}_{cut}^8$')
        ax.plot(temp_list, coefs[:,8]/3, label=r'$\frac{1}{3}\alpha_{1112}\mathcal{P}_{cut}^8$')
        ax.plot(temp_list, coefs[:,9]/3, label=r'$\frac{1}{3}\alpha_{1122}\mathcal{P}_{cut}^8$')
        ax.plot(temp_list, coefs[:,10]/6, label=r'$\frac{1}{6}\alpha_{1123}\mathcal{P}_{cut}^8$')
    except:
        pass
    mpl.rcParams['lines.marker'] = 'o'
    return ax

def get_temp_mat(temp: np.array, order: int) -> np.ndarray:
    assert len(temp.shape)==1
    nt = temp.shape[0]
    mat = np.zeros((nt,order+1))
    for i, t in enumerate(temp):
        for j in range(order+1):
            mat[i,j] = t**j
    return mat

def process_data(temp, folder, polar_cutoff=0.06 ,energy_cutoff_ratio=0.8, ncell=None):
    if os.path.exists('./data_storage/fes3D{}K.npy'.format(temp)):
        polar = np.load('./data_storage/polar.npy')
        fes = np.load('./data_storage/fes3D{}K.npy'.format(temp))
    else:
        polar, fes =  get_final_fes(folder, ncell=ncell)
        np.save('./data_storage/polar.npy', polar)
        np.save('./data_storage/fes3D{}K.npy'.format(temp), fes)
    funit =  kb* temp * 1000  # kbT/meV
    polar = polar / polar_cutoff
    ngrid = polar[:,0].shape[0]
    n = int(np.round(ngrid**(1/3)))
    assert n**3 == ngrid, 'projected data can not be transformed to a square matrix'
    #### get a sparser p grid
    px = polar[:,0].reshape(n,n,n)[::2,::2,::2]
    py = polar[:,1].reshape(n,n,n)[::2,::2,::2]
    pz = polar[:,2].reshape(n,n,n)[::2,::2,::2]
    fes = fes.reshape(n,n,n)[::2,::2,::2]
    fcutoff = (fes.max()-fes.min()) * energy_cutoff_ratio + fes.min()
    #### fit well-explored region only
    proj = (px>=0)&(py>=0)&(pz>=0)&(py>=px)&(pz>=py) & (fes < fcutoff)
    px = px[proj]
    py = py[proj]
    pz = pz[proj]
    print('min(F)={:.3f}kT,max(F)={:.3f}kT, kT={:.3f}meV'.format(fes.min()/funit,fes.max()/funit, funit))
    print('number of fitted data={}, fcut={:.3f}kT'.format(proj.sum(), fcutoff/funit))
    fes = fes[proj]
    return px, py, pz, fes

