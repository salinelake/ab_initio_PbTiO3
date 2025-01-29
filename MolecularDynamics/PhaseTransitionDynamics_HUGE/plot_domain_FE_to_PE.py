from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os
## set global font size, box linewidth
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 3

temp = 823
folder = 'T{}'.format(temp) 

plot_idx = [ 60, 80, 98, 100, 102, 105]
nplots = len(plot_idx)
fig, axs = plt.subplots(nplots, 1,figsize=(40, nplots * 2), sharex=True, sharey=True)
for idx, time in enumerate(plot_idx):
    field = np.load(os.path.join(folder, 'figures/dipole{}.npy'.format(time)))
    section = field[:,:, 8, 2] 
    cax = axs[idx].imshow(section.T, cmap='twilight_shifted', vmin = -4, vmax = 4, aspect='equal', extent=[0, 512, 0, 10])
    axs[idx].set_ylabel('Y', fontsize=24)
    axs[idx].set_title(r'({}) $t$ = {:.1f} ps'.format(idx+1, time), fontsize=24)
axs[-1].set_xlabel('X', fontsize=24)
plt.tight_layout()

## plot colorbar for the last plot 
cbar = fig.colorbar(cax, ax=axs , shrink=0.6, orientation='vertical')
## set colorbar label
cbar.set_label('$p^z_i$ [eA]', loc='center')

fig.savefig('T{}K_domain.png'.format(temp), dpi=300)


fig, ax = plt.subplots(1,2, figsize=(12, 6))

thermo = np.genfromtxt(os.path.join(folder, 'pto.log'), skip_header=1641, invalid_raise=False)[::100]
polar_abs = []
polar_mag = []
polar_xy = []
latt_c = []
for idx in range(120):
    field = np.load(os.path.join(folder, 'figures/dipole{}.npy'.format(idx)))
    polar = np.abs(field[...,2]).mean()
    polar_mag.append(((field**2).sum(-1)**0.5).mean())
    polar_xy.append(((field[...,:2]**2).sum(-1)**0.5).mean())
    polar_abs.append(polar)
    latt_c.append(thermo[idx, -1]/16)
ax[0].plot( np.arange(70), polar_abs[50:120], linewidth=4)
ax[0].plot( np.arange(70), polar_mag[50:120], linewidth=4)
ax[0].plot( np.arange(70), polar_xy[50:120], linewidth=4)
ax[1].plot( np.arange(70), latt_c[50:120], linewidth=4)
# ax.plot(polar_abs, linewidth=4)
ax[0].scatter(plot_idx, [polar_abs[idx] for idx in plot_idx], s=90, color='purple', zorder=10)
## add text to each scatter point
for i in range(nplots):
    ax[0].text(plot_idx[i]+3, polar_abs[plot_idx[i]], '({})'.format(i+1), fontsize=18, color='black', ha='center', va='bottom')
ax[0].set_xlabel(r'$t$ [ps]')
ax[0].set_ylabel(r'$\overline{|p^z_i|}$ [e$\mathrm{\AA}$]')
ax[1].set_xlabel(r'$t$ [ps]')
ax[1].set_ylabel(r'$c$ [$\mathrm{\AA}$]')
plt.tight_layout()
plt.savefig('T{}K_polar_abs.png'.format(temp), dpi=300)