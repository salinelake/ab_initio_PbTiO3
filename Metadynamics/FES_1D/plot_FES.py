
from utility import *
mpl.rcParams['lines.linewidth'] = 1.5

stride=1000
pace=250
dt=0.0005 #ns
fes_inverval = stride*pace*dt/1000

temp = 820
lb,ub = 0.1, 2.2

datafolder = './data'

# Create 2x1 subplot grid
fig = plt.figure(figsize=(8,6))
gs = fig.add_gridspec(2, 5)  # Changed to 2x3 grid
ax1 = fig.add_subplot(gs[0, 0:3]) # Top left, spans 2 columns
ax2 = fig.add_subplot(gs[1, 0:3]) # Bottom left, spans 2 columns
ax3 = fig.add_subplot(gs[0, 3:])   # Right column full height, 1 column wide
ax4 = fig.add_subplot(gs[1, 3:])   # Right column full height, 1 column wide
# Share x-axis between ax2 and ax1
ax2.sharex(ax1)

#######################################    First FIGURE
print('getting data for T=815K and different L')
temp=815
for ss in [6,9,12,15]:
    folder = os.path.join( datafolder , '{}K{}x{}x{}'.format(temp,ss,ss,ss))
    x,y,counter = get_final_fes(folder, [lb,ub])
    ax1.plot(x, y, markersize=0, label=r'${}K, L={}$'.format(temp, ss ))
ax1.set_xlabel(r'$d_c$ [e$\mathrm{\AA}$]',fontdict=font)
ax1.set_ylabel(r'$G(T,d_c)$ [meV/f.u.]',fontdict=font)
# ax[0].set_ylim(top=0.2)
ax1.legend(fontsize=12,frameon=False )
ax1.set_yticks([0.0,0.5,1.0])
# ax1.text(0.3, 0.4, r'$T={}$K'.format(temp), transform=ax1.transAxes, fontsize=14,verticalalignment='top')

#######################################    Second FIGURE
stride = 1000
pace = 250
fes_inverval = stride*pace*dt/1000
temp_list = [815,820,825,830]
error_list = [0.01, 0.01, 0.01, 0.01]
# clrs = sns.color_palette("husl", len(temp_list))
lb,ub = 0.1, 2.2
for temp,error in zip(temp_list,error_list):
    folder = os.path.join( datafolder , '{}K15x15x15'.format(temp))
    x,y,nfes = get_final_fes(folder, [lb,ub])
    ax2.plot(x,y, markersize=0, 
        linewidth=1.5,
        label=r'${}K, L=15$'.format(temp ))
    ax2.fill_between(x,y-error,y+error,
        interpolate=True,
        alpha=0.3,
        )
ax2.set_xlabel(r'$d_c$ [e$\mathrm{\AA}$]',fontdict=font)
ax2.set_ylabel(r'$G(T,d_c)$ [meV/f.u.]',fontdict=font)
ax2.legend(fontsize=12,frameon=False )
## set yticks to be 0.0,0.3,0.6
ax2.set_yticks([0.0,0.3,0.6])
# ax2.text(0.25, 0.4, r'$L=15$', transform=ax2.transAxes,  fontsize=14,verticalalignment='top')

#######################################    THIRD FIGURE
temp_list = [815, 820,825,830]
ss_list = [6,9,12,15]
dF_list = []
divider = 1.0
for ss in ss_list:
    ncell = ss**3
    natom = ncell * 5
    for temp in temp_list:
        folder = os.path.join( datafolder , '{}K{}x{}x{}'.format(temp,ss,ss,ss))
        colvar = np.genfromtxt(os.path.join(folder,'COLVAR_static'),skip_footer=1)
        time = colvar[:,0]
        dp_abs = colvar[:,4]
        bias =  colvar[:,5]
        offset = colvar[:,7]
        
        weight = np.exp((bias-offset)*kjmol2eV/kb/temp)
        cubic_filter = (dp_abs < divider)*1
        tetra_filter = (dp_abs >= divider)*1
        p_ratio = (tetra_filter*weight).mean() / (cubic_filter*weight).mean()
        dF = - kb * temp * np.log(p_ratio) *1000/ncell  ## F_cubic - F_tetra in meV/natom
        dF_list.append(dF)
        print('-----------------{}---------------'.format(folder) )
        print('sampling time: {} ps, #data={}'.format(time[-1]-time[0], cubic_filter.shape[0]))
        print('at cubic:',cubic_filter.sum()/cubic_filter.shape[0])
        print('prob(cubic)/prob(tetra)={}'.format(p_ratio) )
        print('F_tetra-F_cubic={}meV/atom'.format(dF) )

ndata = len(ss_list)
arr_temp = np.array(temp_list)
arr_dF = np.array(dF_list).reshape(ndata,-1)

for idx, ss in enumerate(ss_list):
    ax3.plot(arr_temp,arr_dF[idx],linestyle='dashed',linewidth=3,markersize=8,label='L={}'.format(ss))
ax3.hlines(0,temp_list[0],temp_list[-1],colors='black',linestyles='dashed')
ax3.set_xlabel(r'$T$ [K]',fontdict=font)
ax3.set_ylabel(r'$\Delta G(T)$ [meV/f.u.]',fontdict=font)
ax3.legend(fontsize=10,frameon=False,loc='upper left')
ax3.set_ylim(-0.55, 0.5)

#######################################    FOURTH FIGURE
temp_list = [820,820,820]
lsize_list = [9,12,15]
vol = np.array(lsize_list)**3
barrier_list = []
kbT = np.array(temp_list) * 8.61733e-5 * 1000
for temp,lsize in zip(temp_list,lsize_list):
    folder = os.path.join( datafolder , '{}K{}x{}x{}'.format(temp,lsize,lsize,lsize))
    x,y,nfes = get_final_fes(folder, [lb,ub])
    # Find max in range 0.6 < x < 1.5
    mask1 = (x > 0.6) & (x < 1.5)
    max_y1 = np.max(y[mask1])
    max_x1 = x[mask1][np.argmax(y[mask1])]
    # Find min in range x > 1.5
    mask2 = x > 1.5
    min_y2 = np.min(y[mask2])
    min_x2 = x[mask2][np.argmin(y[mask2])]
    
    energy_barrier = max_y1 - min_y2
    barrier_list.append(energy_barrier)
ax4.plot(vol,np.array(barrier_list) * vol / kbT, marker='*',markersize=15,linewidth=2, linestyle='dashed', color='black')
ax4.set_xlabel(r'$N$',fontdict=font)
ax4.set_ylabel(r'Barrier [$\mathrm{k_BT}$]',fontdict=font)

##### Final touch
ax1.set_title('(a)',loc='left', fontsize=15)
ax2.set_title('(b)',loc='left', fontsize=15)
ax3.set_title('(c)',loc='left', fontsize=15)
ax4.set_title('(d)',loc='left', fontsize=15)

plt.tight_layout()
plt.savefig('./FES_1D.png',dpi=300)
plt.close()
