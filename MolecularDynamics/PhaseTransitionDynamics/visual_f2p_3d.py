import os
import numpy as np
import ase, ase.io
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d, convolve1d
from matplotlib import cm
import matplotlib as mpl
from time import time as gettime
import torch as th

mpl.rcParams['lines.linestyle'] = 'solid'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.linewidth'] = 2.5

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['xtick.minor.width'] = 2.5

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 1

mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.linewidth'] = 2  

mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 20

def moving_average_half_gaussian(a, sigma=25, axis=0, truncate=3.0):
    fsize = int(truncate * np.ceil(sigma))
    weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
    throw = fsize//2 + 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    ret = convolve1d(a, weights, axis=axis, origin=1 )
    return ret[throw:-throw,...]
 

mfile = './T823S0/dipole2.npy'
############## Load structure data
ss=15
dt = 0.0005 * 20
############# Load magnetic data
moments = - np.load(mfile)  # (nframes, ss,ss,ss, 3)
#####   
smooth = 20
cg = True
if cg:
    moments_cg = moving_average_half_gaussian(moments, sigma=smooth, axis=0)
else:
    moments_cg = moments.copy()
nframes = moments_cg.shape[0]

mycmap = cm.twilight_shifted
print( moments.shape )
print( moments_cg.shape )

# x,y = np.meshgrid(np.linspace(0,ss-1,ss),np.linspace(0,ss-1,ss))

#################   Phase transition dynamics
tt_list = [12.5, 14.5, 16.5, 17.0 ,17.25, 17.5 ]
toffset= 200
##### setup figure
nrow = 2
ncol = len(tt_list)//nrow
# fig, _ax = plt.subplots(nrow,ncol, figsize=( 4*ncol,4.5*nrow),subplot_kw=dict(projection="3d"))
# ax = _ax.flatten()

fig, axs = plt.subplots(ncols=ncol, nrows=nrow+1, figsize=( 4.5*ncol,4*(nrow+1)),subplot_kw=dict(projection="3d"))
gs = axs[0, 0].get_gridspec()
# remove the axes
for _ax in axs[0, :]:
    _ax.remove()
# axbig = fig.add_subplot(gs[0, :])
axbig =fig.add_axes([0.1,.75,.8,.23])
ax = axs[1:].flatten()
############################## evolution of Pz ############################## 
Pz = moments[...,-1].mean(1).mean(1).mean(1)
Px =moments[...,0].mean(1).mean(1).mean(1)
Py =moments[...,1].mean(1).mean(1).mean(1)
timevec =np.arange(Pz.size) * dt + toffset
axbig.plot(timevec[timevec<30+toffset], Pz[timevec<30+toffset], linewidth=3,label='z-component')
axbig.plot(timevec[timevec<30+toffset], Px[timevec<30+toffset], alpha=0.6,label='x-component')
axbig.plot(timevec[timevec<30+toffset], Py[timevec<30+toffset], alpha=0.6,label='y-component')
# axbig.axhline(0, color='black', alpha=0.5, linestyle='dashed')
axbig.axhline(1.64, color='black', alpha=0.5, linestyle='dashed')
axbig.legend(frameon=True,framealpha=1,)
axbig.set_xlabel(r'$t$ [ps]')
axbig.set_ylabel(r'$\overline{p}$ [eA]')
############################## evolution of dipole ############################## 
paxis =2
for ii, tt in enumerate(tt_list):
    if cg:
        idx = int(tt / dt - (3 * smooth /2 + smooth) )
    else:
        idx = int(tt / dt) 
    ### plot Pz first
    order =  moments_cg[idx,:,:,:,paxis].mean()  
    axbig.plot(tt+toffset, order, marker='o',markersize=10,color='purple')
    axbig.annotate('({})'.format(ii+1), (tt-1.3+toffset, order-0.15), 
               xycoords='data', va='center', fontsize=20)    
    axbig.set_xlim(toffset, toffset+30)
    ###
    facevalues = moments_cg[idx,...,paxis]
    vmax = 3
    facecolors = mycmap((facevalues+vmax)/vmax/2)[...,:3]
    ax[ii].voxels(filled=np.ones_like(facevalues).astype(bool), facecolors=facecolors)
    ax[ii].set_axis_off()
    ax[ii].set_title( '({})'.format(ii+1), y = 0.97)

fig.tight_layout()
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
fig.subplots_adjust(left=0.01 , right=0.94, top=0.99, bottom=0.01)
cbar_ax = fig.add_axes([0.945, 0.07, 0.015 , 0.5])
sm = plt.cm.ScalarMappable(norm=norm, cmap=mycmap)
cb = fig.colorbar(sm,  cax=cbar_ax)#, orientation='horizontal')
cbar_ax.set_title(r'$\tilde{p}_i^z$ [eA]', pad=30)
fig.savefig('f2p_3d.png', dpi=300)
plt.close(fig)
 