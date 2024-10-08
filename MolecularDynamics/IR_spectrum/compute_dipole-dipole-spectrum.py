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
mpl.rcParams['axes.linewidth'] = 1.5 
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 8
mycmap = cm.plasma
device = 'cuda' if th.cuda.is_available() else 'cpu'

def DHO(x, freq, damp):
    nom = damp * freq * freq
    denom = (freq**2 - x**2)**2 + (damp * freq)**2
    return  nom / denom

def gaussian(x, freq, damp, amp):
    return amp * np.exp(-0.5*((x-freq)/damp)**2 )


# fig,ax = plt.subplots( 2,  figsize = (8, 4), sharex=True )
# fig.subplots_adjust(wspace=0, hspace=0)
# fig.text(0.11, 0.85, '(a)', ha='center', fontsize=12)
# fig.text(0.11, 0.46, '(b)', ha='center', fontsize=12)
fig,ax = plt.subplots( 1,  figsize = (4, 3) )

########
temp_list =  [ 810,  830, 900, 1000, 1100, 1200  ]
temp_plot =  [ 810,  830, 900, 1000, 1100, 1200  ]
throw = 10000
dt = 0.0005 #ps

exp_data = np.loadtxt('sus_imag.csv',delimiter=',',)
exp_freq = exp_data[:,0]
exp_ir = 4 * np.pi * exp_data[:,0] * exp_data[:,1]
soft_mode = np.array([23.32674628, 32.44540828, 41.49598341, 51.46851626, 58.19284341])
peak = []
for ii, temp in enumerate(temp_list):
    phase = 't' if temp < 821 else 'c'
    ss = 15      
    folder = '/home/pinchenx/data.gpfs/ferro_scratch/PTO/IR_spectrum/data_L{}/T{}K{}'.format(ss, temp, phase ) 
    print('processing system {} '.format(folder))
    datafile = "./data_buffer/dpcorr_T{}.npy".format(temp)
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
        dipole = dipole[throw:, :] 
        duration = 10 ##ps
        ncorr = 1
        N = int(duration / dt)
        acf_t = [0]
        acf_x = [(dipole[::ncorr]**2).sum(-1).mean().cpu().numpy()  ]
        for idx in range(1, N):
            acf_t.append(idx * dt)
            acf_x.append( (dipole[idx::ncorr] * dipole[:-idx:ncorr]).sum(-1).mean().cpu().numpy()   )
        acf_t = np.array(acf_t)
        acf_x = np.array(acf_x)
        np.save(datafile, acf_x)

    acf_fourier = np.fft.fft(acf_x)    ## 2pi ij/N
    acf_freal = acf_fourier.real
    # acf_freal = acf_fourier / acf_fourier[0]

    acf_freal *= 1e-8
    nframes = acf_fourier.size
    freq_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
    wavenumber = freq_fourier / 6 / np.pi * 100

    wavenumber = wavenumber[wavenumber<200]
    acf_freal = acf_freal[:len(wavenumber)]
    peak.append( wavenumber[np.argmax(acf_freal)] )
    ##
    ax.plot(wavenumber[1:], acf_freal[1:] , label='T={}K'.format(temp) )
    print('smallest freq: {}cm-1'.format(wavenumber[1]))
# ax.axvline( 0 , alpha=0.3, color='black', markersize=0 , linestyle='solid', linewidth=1.5)
## change to log scale
ax.set_yscale('log')
ax.set_xlim(-10, 200)
ax.set_ylim(1e-3, 10**1.5)
ax.set_xlabel(r'$\omega$[cm${}^{-1}$]', fontsize=12)
ax.set_ylabel(r'$S[p^G](\omega)$ [arb. unit]', fontsize=12)
ax.legend(loc='lower left', fontsize=7)
peak = np.array(peak)
## add inset
axins = ax.inset_axes([0.50, 0.58, 0.45, 0.4])
axins.plot(np.array(temp_plot[1:])-821, soft_mode**2, 'o', linestyle='dashed', color='black', markersize=4, label=r'$\omega_s^2$')
axins.plot(np.array(temp_plot)-821, peak**2, 'o', linestyle='dashed', color='purple', markersize=4, label=r'$\omega_p^2$')
axins.set_xlabel(r'$T-T_c$[K]', fontsize=8)
axins.set_xlim(0)
# axins.set_ylabel('$\omega_s^2$ [cm${}^{-2}$]', fontsize=8)
axins.set_ylim(-100,4000)
axins.legend(fontsize=8)
## set label size
for label in (axins.get_xticklabels() + axins.get_yticklabels()):
    label.set_fontsize(8)

 
fig.tight_layout()
fig.savefig('polar-correlation.png', dpi=300)

