import os
import numpy as np
import dpdata
import ase 
import ase.io
from fse.systems import perovskite
from utility import *


    
temp_list =   [   400, 500, 600, 700, 750, 800, 820.99, 821.01,  850,  900, 1000  ]
ss = 15
throw = 100
ncell = ss**3
dipole_average = []
fig4, ax4 = plt.subplots(3,2, sharex=True, sharey=True, figsize = (5.2, 6.2))
 
################
ax2_idx=0
ax4_idx=0
for ii, temp in enumerate(temp_list):
    folder = '/home/pinchenx/data.gpfs/Github/ferro_scratch/PTO/DPMD/check_dipole_distribution/data/T{}_ss{}'.format(temp, ss ) 
    pto_factory = perovskite(ABO3=['He','Li','H'], born_charges=[3.7140, 5.4879, -3.3551, -2.9234])
    ######################################## load data ########################################
    data_file = os.path.join(folder, 'dipole.npy')
    if os.path.exists(data_file):
        dipole_list = np.load(data_file)
        print('system {} loaded -- {} frames'.format(folder, dipole_list.shape[0]))
    else:
        from deepmd.infer import DeepDipole # use it to load your trained model
        model = DeepDipole('/home/pinchenx/tigress/DPModels/PTO-MODEL_DEV/m0/dipole-compress.pb')
        ase_atoms = ase.io.read(os.path.join(folder,'pto.lammpstrj'),format = 'lammps-dump-text',index=':')
        lattice = pto_factory.get_effective_lattice([ss,ss,ss], ase_atoms[0], central_element='Li')
        print('Found lattice')
        print('system {} loaded -- {} frames'.format(folder, nframe))
        atypes = ase_atoms[0].get_atomic_numbers()-1
        dipole_list = []
        for idx in range(throw, len(ase_atoms)):
            dipole = pto_factory.get_lattice_dipole( model, atypes, ase_atoms[idx], lattice )
            dipole_list.append(dipole)
        dipole_list = np.array(dipole_list)
        np.save(os.path.join(folder, 'dipole.npy'), dipole_list)
    nframe = dipole_list.shape[0]  ##  (nframe, #Lx,#Ly,#Lz, 3)
 
    ######################### polarization distribution (simple projection ) ####################
    dipole_abs = dipole_list.reshape(-1,3)   ##  (nframe*ncell, 3)
    d100 =  dipole_abs[:,0]
    d010 =  dipole_abs[:,1]
    d001 =  - dipole_abs[:,2]
    d110 = ( dipole_abs * np.array([1,1,0]) ).sum(-1) / 2**0.5
    hist_d, xx, yy = np.histogram2d(d110, d001 , 
            bins=100, range=[[-4,4],[-4,4]], density=True  )
    hist_dd, xx, yy = np.histogram2d(d100, d010 , 
        bins=100, range=[[-4,4],[-4,4]], density=True  )
    extent_d = [xx[0], xx[-1], yy[0], yy[-1]]
 
    #############################  plot simple projection histogram
    if temp in [   400,    820.99,  821.01  ]:
        dipole_avg = np.sqrt((dipole_list**2).sum(-1)).mean()
        im = ax4[ax4_idx,0].imshow(hist_d.T, extent=extent_d, origin='lower', 
            cmap=cmap2, interpolation='spline16', label='T={}K'.format(temp))
        im = ax4[ax4_idx,1].imshow(hist_dd.T, extent=extent_d, origin='lower', 
            cmap=cmap2, interpolation='spline16', label='T={}K'.format(temp))
        if temp < 820 or temp > 822:
            temp_label = 'T={}K'.format(temp)
        elif temp < 821:
            temp_label = 'T=821K, Ferro' 
        else:
            temp_label = 'T=821K, Para' 

        ax4[ax4_idx,0].text(0.05, 0.12,temp_label , transform=ax4[ax4_idx,0].transAxes, fontsize=12, color='black', #fontdict=text_font,
            verticalalignment='top')
        ax4[ax4_idx,1].text(0.05, 0.12, temp_label , transform=ax4[ax4_idx,1].transAxes, fontsize=12, color='black', #fontdict=text_font,
            verticalalignment='top')
        ax4[ax4_idx,0].add_patch(plt.Circle((0, 0), dipole_avg, color='black', fill=False,linestyle='dashed'))
        ax4[ax4_idx,1].add_patch(plt.Circle((0, 0), dipole_avg, color='black', fill=False,linestyle='dashed'))
        dx = -dipole_avg/2
        dy = 3**0.5*dipole_avg/2
        ax4[ax4_idx,0].arrow(x=0,y=0, dx=dx,dy=dy,  linestyle='dashed',
            head_width=0.2, head_length=0.2, length_includes_head=True,   color='black')
        ax4[ax4_idx,0].text(dx-0.1 , dy , r'$\langle |p| \rangle$', ha='right', va='bottom', 
                fontsize=12, color='black' )
        ax4[ax4_idx,0].set_xticks([-2,0,2])
        ax4[ax4_idx,0].set_yticks([-2,0,2])
        ax4[ax4_idx,1].set_xticks([-2,0,2])
        ax4[ax4_idx,1].set_yticks([-2,0,2])

        ax4_idx += 1
 
 
#########################   Direct projection histogram  ################################### 
fig4.subplots_adjust(hspace=0)
ax4[-1,0].set_xlabel(r'$  p_{[110]} $ [eA]')
ax4[-1,1].set_xlabel(r'$  p_{[100]} $ [eA]')

ax4[0,0].set_title(r'$(1\overline{1}0)$  Plane', fontsize=14)
ax4[0,1].set_title(r'$(001)$ Plane', fontsize=14)
ax4[1,0].set_ylabel(r'$  p_{[001]} $ [eA]')
ax4[1,1].set_ylabel(r'$  p_{[010]} $ [eA]')

# plt.setp([a.get_xticklabels() for a in fig2.axes[:-1]], visible=False)

fig4.subplots_adjust(left=0.125 , right=0.875, top=0.95, bottom=0.1)
cbar_ax = fig4.add_axes([0.9, 0.17, 0.02 , 0.7])
# fig2.subplots_adjust(left=0.1 , right=0.95, top=0.99, bottom=0.2)
# cbar_ax = fig2.add_axes([0.2, 0.1, 0.5 , 0.02])

sm = plt.cm.ScalarMappable(cmap=cmap2)#, norm=plt.Normalize(vmin=0, vmax=1))
cb = fig4.colorbar(sm,  cax=cbar_ax)#, orientation='horizontal')
# cb.set_label( label=r'$T$[K]' )
# fig2.tight_layout()
fig4.savefig('DipolePlane.png', dpi=300)
plt.close(fig4)