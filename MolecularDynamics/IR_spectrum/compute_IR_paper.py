import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm   as cm
from numpy import polyfit
import torch as th
from scipy.optimize import curve_fit
# mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
# mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.linestyle'] = 'dashed'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 2  
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 10
mycmap = cm.plasma
device = 'cuda' if th.cuda.is_available() else 'cpu'

def DHO(x, freq, damp):
    nom = damp * freq * freq
    denom = (freq**2 - x**2)**2 + (damp * freq)**2
    return  nom / denom

def gaussian(x, freq, damp, amp):
    return amp * np.exp(-0.5*((x-freq)/damp)**2 )


fig,ax = plt.subplots( 2,  figsize = (8, 4), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0, hspace=0)
fig.text(0.11, 0.85, '(a)', ha='center', fontsize=12)
fig.text(0.11, 0.46, '(b)', ha='center', fontsize=12)

########
temp_list =   [  300,   500 ,   700,  750,   810,  830,   900, 1000, 1100, 1200  ]
temp_plot =  [  300,   500 ,   700,    810,  830,    1000,  1200  ]
throw = 10000
dt = 0.0005 #ps

exp_data = np.loadtxt('sus_imag.csv',delimiter=',',)
exp_freq = exp_data[:,0]
exp_ir = 4 * np.pi * exp_data[:,0] * exp_data[:,1]
soft_mode = []
soft_damp = []
soft_err = []
for ii, temp in enumerate(temp_list):
    phase = 't' if temp < 821 else 'c'
    ss = 12 if temp < 821 else 15      
    folder = '/home/pinchenx/data.gpfs/ferro_scratch/PTO/IR_spectrum/data_L{}/T{}K{}'.format(ss, temp, phase ) 
    
    datafile = "./data_buffer/IRT{}.npy".format(temp)
    if os.path.exists(datafile):
        acf_x = np.load(datafile)
        acf_t = np.arange(acf_x.size) * dt
    else:
        dipole = []
        plumed_file = np.genfromtxt(os.path.join( folder, 'COLVAR'), skip_footer=1, )
        dipole = plumed_file[:,1:4]
        dipole = th.tensor(dipole, dtype=th.float, device=device)
        print('system {} loaded data -- {}  '.format(folder, dipole.shape))
        #######
        dpdt = (dipole[1:] - dipole[:-1]) / dt
        dpdt = dpdt[throw:, :] 
        del dipole
        duration = 10 ##ps
        ncorr = 1
        N = int(duration / dt)
        acf_t = [0]
        acf_x = [(dpdt[::ncorr]**2).sum(-1).mean().cpu().numpy()  ]
        for idx in range(1, N):
            acf_t.append(idx * dt)
            acf_x.append( (dpdt[idx::ncorr] * dpdt[:-idx:ncorr]).sum(-1).mean().cpu().numpy()   )
        acf_t = np.array(acf_t)
        acf_x = np.array(acf_x)
        np.save(datafile, acf_x)
        del dpdt


    # print(acf_x.sum() )
    acf_fourier = np.fft.fft(acf_x)    ## 2pi ij/N
    # print(acf_fourier)
    acf_freal = acf_fourier.real
    acf_freal *= 1 / temp  * 1e-6 #! TODO
    nframes = acf_fourier.size
    freq_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
    wavenumber = freq_fourier / 6 / np.pi * 100
    if temp in temp_plot:
        if temp < 821:
            if temp == 300:
                ax[0].plot(wavenumber[wavenumber<800], acf_freal[wavenumber<800], label='T={}K'.format(temp), zorder=10)
            else:
                ax[0].plot(wavenumber[wavenumber<800], acf_freal[wavenumber<800], label='T={}K'.format(temp) )
        else:
            ax[1].plot(wavenumber[wavenumber<800], acf_freal[wavenumber<800], label='T={}K'.format(temp))
    ### get first peak frequency
    center = np.argmax(acf_freal[wavenumber<100])
    center_height = acf_freal[wavenumber<100].max()
    fitting_region = (acf_freal > (center_height / 2)) & (wavenumber<100)
    data_x = wavenumber[fitting_region]
    data_y = acf_freal[fitting_region]
    try:
        popt, pcov = curve_fit(gaussian, data_x, data_y, p0=[wavenumber[center], 10, 2])
        perr = pcov[0,0]**0.5
    except:
        popt = [wavenumber[center],0]
        perr = wavenumber[1]-wavenumber[0]
    # soft_mode.append(wavenumber[np.argmax(acf_freal[wavenumber<100])])
    soft_mode.append(popt[0])
    soft_damp.append(popt[1])
    soft_err.append(perr)
    print('temp :', temp)
    print('soft mode: ', soft_mode[-1])
    print('soft damp: ', soft_damp[-1])
    print('soft error: ', soft_err[-1])

### Ramen peak
alpha=0.3
raman_location = [87.5, 148.5, 218.5, 359.5, 505.0, 647]
raman_label    = [r'1$E$', r'1$A_1$', r'2$E$', r'2$A_1$',r'4$E$',r'3$A_1$']
ax[0].axvline( 87.5  , alpha=alpha, color='black', markersize=0, label='Raman')
ax[0].axvline( 148.5 , alpha=alpha, color='black', markersize=0)
ax[0].axvline( 218.5 , alpha=alpha, color='black', markersize=0)
# ax[0].axvline( 289.0 , alpha=alpha, color='black', markersize=0)
ax[0].axvline( 359.5 , alpha=alpha, color='black', markersize=0)
ax[0].axvline( 505.0 , alpha=alpha, color='black', markersize=0)
# ax[0].axvline( 627.0 , alpha=alpha, color='black', markersize=0)
ax[0].axvline( 647.0 , alpha=alpha, color='black', markersize=0)
axtx = ax[0].twiny()
ax[0].set_xlim(-30,800)
ax[1].set_xlim(-30,800)
axtx.set_xlim(-30,800)
axtx.set_xticks( raman_location, labels=raman_label )
axtx.tick_params(axis='x', width=0 )
 
###  soft mode 
temp_list = np.array(temp_list)
soft_mode = np.array(soft_mode)**2

###
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel(r'$\omega$[cm${}^{-1}$]', fontsize=14)
# ax[1].set_ylabel(r'$\alpha(\omega)n(\omega)$ [arb. unit]')
fig.supylabel(r'$\alpha(\omega)n(\omega)$ [arb. unit]', fontsize=14)

fig.tight_layout()
fig.savefig('IR.png', dpi=300)
######################################################################################
################### SOft MOde ################################################
 

####Fontana, raman, ferro
exp1_T = [19.57715, 99.42824,199.37687,300.41964,399.93862,433.37403,465.18690,477.42796,489.12195,495.52669,]
exp1_w = [7.85383, 7.53866, 7.13125, 6.53540, 5.62962, 4.89226, 4.34329, 3.90008, 3.55109, 3.22850]
exp1_T = np.array(exp1_T) - 495
exp1_w = np.array(exp1_w) * 1000
## Shirane  Phys. Rev. B 2, 155
exp2_T = [783.13253, 1054.21687]
exp2_w = [2.98644, 5.99376]
exp2_T = np.array(exp2_T) - 763
exp2_w = np.array(exp2_w) * 10/1.24   # meV to cm^-1
exp2_w = exp2_w**2
### Tomeno   PHYSICAL REVIEW B 86, 134306 (2012)
exp3_T = [793.97590, 873.49398,973.49398,1073.49398,1173.49398,]
exp3_w = [ 2.43056, 3.93182, 4.92041,5.95179 , 7.96748]
exp3_T = np.array(exp3_T) - 763
exp3_w =  np.array(exp3_w) * 10/1.24 
exp3_w = exp3_w**2
#### Hlinka  PHYSICAL REVIEW B 87, 064101 (2013)
exp4_T = [1.22224, 12.77878,18.74993, 21.73268,31.80321,41.49757,51.94143,62.38473,71.70915,97.05886,122.41760,172.41045,]
exp4_w = [ 0.28376,  0.33232, 0.37481, 0.38847, 0.44461, 0.49014, 0.54932, 0.60698, 0.65857, 0.76480, 0.89530,  1.20941]
exp4_T = np.array(exp4_T) 
exp4_w = np.array(exp4_w) * 1000
 

####
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
figs,axs = plt.subplots( 1,  figsize = (6, 4))
axs.plot(temp_list[temp_list < 821] - 821, soft_mode[temp_list < 821], markersize=6, color='black')
axs.plot(temp_list[temp_list >= 821] - 821, soft_mode[temp_list >= 821], markersize=6, color='black',label='Molecular Dynamics')
axs.plot(exp1_T, exp1_w, markersize=4, marker='D',  label='Raman, Fontana et al.')
axs.plot(exp2_T, exp2_w, markersize=4,  marker='D',  label='Neutron, Shirane et al.')
axs.plot(exp3_T, exp3_w, markersize=4,  marker='D',  label='Neutron, Tomeno et al.')
axs.plot(exp4_T, exp4_w, markersize=4,  marker='D',  label='Hyper-Raman, Hlinka et al.')

curie_t0 = 792
curie_t = np.arange(100)*2+curie_t0
curie_c = 1.6e5
###
xx = temp_list[temp_list >= 821]
yy = soft_mode[temp_list >= 821]
zz = np.polyfit(xx, yy, 1)
print('w^2={}(T-{})'.format(zz[0], -zz[1]/zz[0]))
###

print(yy**0.5)
axs.legend()
# axs.axvline(0, linestyle='dashed', alpha=alpha, color='black')
axs.set_xlabel(r'$T-T_c$ [K]' )
axs.set_ylabel(r'$\omega_s^2$ [cm${}^{-2}$]' )
axs.set_ylim(0,9000)
figs.tight_layout()
figs.savefig('SoftMode.png', dpi=300)
