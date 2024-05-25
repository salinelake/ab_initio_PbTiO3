from utility import *
from numpy import polyfit
fig, ax = plt.subplots(2,3,figsize=(16,8))


throw = 400
temp_list =   [300, 400, 500, 600,700, 750,780,790,800, 810, 815, 820, 825, 830, 840, 850, 860, 870, 900, 1000]
ss_list = [9,12,15]
Tc=821.5
P0 = 28000 #bar
thermo_list = []
thermo_ss_list = []
enthalpy_list = []
cp_list = []
data_folder_prefix = '/home/pinchenx/data.gpfs/Github/ferro_scratch/PTO/DPMD/final_susceptibility_press/'

"""
Lattice constant
"""
for ss in ss_list:
    ncell = ss ** 3
    natoms = ncell * 5
    for temp in temp_list:
        phase='t' if temp<Tc else 'c'
        folder = data_folder_prefix+'{}x{}x{}/T{}_ss{}_{}'.format(ss,ss,ss,temp,ss,phase)
        thermo = simultaneous_read(folder, throw)
        thermo_list.append(thermo)
        thermo_ss_list.append(ss)
        ## excessive enthalpy and excessive specific heat.
        enthalpy = (thermo['PotEng'] + P0 * thermo['Volume'] * barA32eV) / natoms - 1.5*kb * temp# eV/atom
        cp = ((enthalpy**2).mean() - (enthalpy.mean())**2)  / (kb*temp**2) * natoms - 1.5*kb #eV/atom/K
        enthalpy_list.append(enthalpy.mean())   # eV/atom
        cp_list.append(cp.mean()) 
        print('folder',folder)
        print('time:{}ps,nframe:{}'.format(thermo['time'][-1],thermo['time'].shape[0]))

  



ndata = len(ss_list)
arr_temp = np.array(temp_list)
arr_cellc = np.array([thermo['c'].mean()/ss for thermo,ss in zip(thermo_list,thermo_ss_list)]).reshape(ndata,-1)
arr_cellb = np.array([thermo['b'].mean()/ss for thermo,ss in zip(thermo_list,thermo_ss_list)]).reshape(ndata,-1)
arr_cella = np.array([thermo['a'].mean()/ss for thermo,ss in zip(thermo_list,thermo_ss_list)]).reshape(ndata,-1)
arr_vol = np.array([thermo['Volume'].mean()/ss**3 for thermo,ss in zip(thermo_list,thermo_ss_list)]).reshape(ndata,-1)
arr_enthalpy = np.array(enthalpy_list).reshape(ndata,-1)
arr_cp = np.array(cp_list).reshape(ndata,-1)

################## PLOTTING##################
'''
subplots(1,2,figsize=(10,4)) :   geometric feature
ax1: lattice constants vs temperature
ax2: tetragonality  vs temperature, also compare to the case without artificial pressure
'''
latt_a_ = np.loadtxt('./exp_data/latt_a_exp.txt')
latt_c_ = np.loadtxt('./exp_data/latt_c_exp.txt')
temp_a = latt_a_[:,0] + 273.15
latt_a = latt_a_[:,1] 
temp_c = latt_c_[:,0]  + 273.15
latt_c = latt_c_[:,1] 

### lattice constants vs temperature
data_idx = -1
ss = ss_list[data_idx]
ax[1,0].plot(temp_a, latt_a, linestyle='dashed', markersize=0, color='black', label='EXP')
ax[1,0].plot(temp_c, latt_c,  linestyle='dashed', markersize=0, color='black',)
ax[1,0].plot(arr_temp[:-1], arr_cellc[data_idx][:-1], label=r'$c$' )
ax[1,0].plot(arr_temp[:-1], arr_cella[data_idx][:-1], label=r'$a$' )
ax[1,0].plot(arr_temp[:-1], arr_cellb[data_idx][:-1], label=r'$b$' )
ax[1,0].plot(arr_temp[:-1], arr_vol[data_idx][:-1]**(1/3), label=r'$(abc)^{1/3}$')


ax[1,0].set_xlabel(r'$T$ [K]',fontdict=font)
ax[1,0].set_ylabel(r'Lattice Const [A]',fontdict=font)
# ax[1,0].set_xlim(300,900)
ax[1,0].legend(fontsize=11,frameon=False,loc='upper right')
### tetragonality  vs temperature
exp_ = np.loadtxt('./exp_data/tetragonality_exp.txt')
temp_exp = exp_[:,0] + 273.15
tetra_exp = exp_[:,1] 


### load lattice constants obtained without artificial pressure
arr_temp_atm =  np.load('temp.npy')
tetra_atm = np.load('tetra-L12.npy')
ax[0,0].plot(temp_exp, tetra_exp, linestyle='dashed', markersize=0, color='black', label='EXP')
for idx,ss in enumerate(ss_list):
    ax[0,0].plot(arr_temp, arr_cellc[idx]/arr_cella[idx], linewidth=2,label=r'$L={}$'.format(ss))
ax[0,0].plot(arr_temp_atm[:-3], tetra_atm[:-3], linewidth=2, label=r'$L=12$, original')
ax[0,0].set_xlabel(r'$T$ [K]',fontdict=font)
ax[0,0].set_ylabel(r'$c/a$',fontdict=font)
ax[0,0].legend(fontsize=12,frameon=False,loc='lower left')
### clean up and save
ax[0,0].set_title('(a)',loc='left', fontsize=15)
ax[1,0].set_title('(b)',loc='left', fontsize=15)
  




"""
HEAT CAPCITY
"""
temp_1 =  [ 700, 750,780, 790, 800, 810, 815, 820, 825, 830, 840, 850, 860, 870, 900]
temp_2 =   [700, 750,780, 790, 800, 810, 815, 820, 821, 822, 824,825, 830, 840, 850, 860, 870, 900]

ss_list = [9,12,15]
Tc=821.5
P0 = 28000 #bar
thermo_list = []
thermo_ss_list = []
temp_list = []
enthalpy_list = []
cp_list = []

for ss in ss_list:
    ncell = ss ** 3
    natoms = ncell * 5
    if ss < 15:
        _temp = temp_1
    else:
        _temp = temp_2
    for temp in _temp:
        temp_list.append(temp)
        phase='t' if temp<Tc else 'c'
        folder = data_folder_prefix +'/{}x{}x{}/T{}_ss{}_{}'.format(ss,ss,ss,temp,ss,phase)
        print('===========folder:{}============'.format(folder))
        ## for thermodynamic property, we shouldn't constraint the global dipole. Because domains doesn't affect specific heat.
        ## i.e. we should include tunneling caused by finite size effect in our computation to mimic 
        ## the experimental measurement where there is always domains. 
        thermo = simultaneous_read(folder, throw, phase,restrain=False)
        thermo_list.append(thermo)
        thermo_ss_list.append(ss)
        ## excessive enthalpy and excessive specific heat.
        enthalpy = (thermo['TotEng'] + P0 * thermo['Volume'] * barA32eV) / natoms - 3 *kb * temp# eV/atom
        cp = ((enthalpy**2).mean() - (enthalpy.mean())**2)  / (kb*temp**2) * natoms -  3*kb #eV/atom/K
        enthalpy_list.append(enthalpy.mean() * 5 * avogadro * eV2J)   # J/mol
        cp_list.append(cp.mean() * 5 * avogadro * eV2J)  # J/mol/K
        print('time:{}ps,kept_frame:{}'.format(thermo['time'][-1],thermo['time'].shape[0]))

# ndata = len(ss_list)
arr_temp = np.array(temp_list)
arr_tetra = np.array([thermo['c'].mean()/thermo['a'].mean() for thermo,ss in zip(thermo_list,thermo_ss_list)])
# arr_cellb = np.array([thermo['b'].mean()/ss for thermo,ss in zip(thermo_list,thermo_ss_list)])
# arr_cella = np.array([thermo['a'].mean()/ss for thermo,ss in zip(thermo_list,thermo_ss_list)])
# arr_vol = np.array([thermo['Volume'].mean()/ss**3 for thermo,ss in zip(thermo_list,thermo_ss_list)])
arr_enthalpy = np.array(enthalpy_list)
arr_cp = np.array(cp_list)

################## PLOTTING##################
'''
subplots(1,2,figsize=(10,4)) :   thermaldynamics feature
ax1: enthalpy vs temperature
ax2: cp  vs temperature 
'''
# temp_exp = np.array([325,375,425,475,525,575,625,675,725,750,760,770,800,850,900,950,1000,1050,1100,1150,1200,1250])
# cp_exp =  np.array( [115,117,120,122,124,126,130,135,144,158,432,140,124,124,124,125,125, 125, 125, 128, 123, 122]) - 3*kb* 5 * avogadro * eV2J
exp1 = np.loadtxt('./exp_data/cp_fz.txt')
temp_exp1 = exp1[:,0] 
cp_exp1 = exp1[:,1] 
exp_enthalpy = np.loadtxt('./exp_data/enthalpy_fz.txt')
temp_enthalpy = exp_enthalpy[:,0] 
exp_enthalpy = exp_enthalpy[:,1] 

exp2 = np.loadtxt('./exp_data/cp_ssr.txt')
temp_exp2 = exp2[:,0] 
cp_exp2 = exp2[:,1] - 3*kb* 5 * avogadro * eV2J

### enthalpy vs temperature
for idx,ss in enumerate(ss_list):
    linewidth = 0 if ss<15 else 2
    filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if current_ss==ss]
    ax[0,1].plot(arr_temp[filter], (arr_enthalpy[filter] - arr_enthalpy[filter].max()) , linewidth=linewidth,label=r'$L={}$'.format(ss))
    ax[1,1].plot(arr_temp[filter], arr_cp[filter]  , linewidth=linewidth, label='$L={}$'.format(ss))
    print(arr_temp[filter])
    print(arr_cp[filter])
    print(arr_tetra[filter] )
ax[0,1].plot(temp_enthalpy[:], exp_enthalpy[:], label='EXP',marker='d', markersize=4)
ax[1,1].plot(temp_exp1[:], cp_exp1[:], label='EXP: FZ',marker='d', markersize=4 )
ax[1,1].plot(temp_exp2[75:], cp_exp2[75:], label='EXP: SSR',marker='s', markersize=4 )
ax[1,1].set_ylim(-10,350)
# ax[1,1].axvline(x=763,  linewidth=2,markersize=0, linestyle='dotted',color='black',label=r"$T_{E}=763$K")
# ax[1,1].axvline(x=821.5,  linewidth=1,markersize=0, linestyle='.',color='b')
# ax[1,1].set_yscale('log')
ax[0,1].set_xlabel(r'$T$ [K]',fontdict=font)
ax[0,1].set_ylabel(r'$H - C_0T$ [J/mol]',fontdict=font)
# ax1.set_ylabel(r'$H$ [eV/atoms]',fontdict=font)
ax[0,1].legend(fontsize=12,frameon=False)
ax[1,1].set_xlabel(r'$T$ [K]',fontdict=font)
ax[1,1].set_ylabel(r'$C_p - C_0$ [J/mol/K]',fontdict=font)
# ax2.set_ylabel(r'$c_p$ [meV$\cdot$K${}^{-1}$/atoms]',fontdict=font)
ax[1,1].legend(fontsize=12,frameon=False)
### clean up and save
ax[0,1].set_title('(c)',loc='left', fontsize=15)
ax[1,1].set_title('(d)',loc='left', fontsize=15)


"""
Polarization and dielectric constant
"""
temp_1 =   [300, 400, 500, 600,700, 750,780,790,800, 810, 815, 820, 825, 830, 840, 850, 860, 870, 900, 1000]
temp_2 =   [300, 400, 500, 600,700, 750,780,790,800, 810, 815, 820, 821, 822, 824, 825, 830, 840, 850, 860, 870, 900, 1000]

ss_list = [9,12,15]
Tc=821.5
P0 = 28000 #bar
thermo_list = []
thermo_ss_list = []
arr_temp = []
polarizatioLong = []
pyrocurrent = []
suscept_long = []
suscept_trans = []

for ss in ss_list:
    ncell = ss ** 3
    natoms = ncell * 5
    if ss < 15:
        _temp = temp_1
    else:
        _temp = temp_2
    for temp in _temp:
        arr_temp.append(temp)
        phase= 't' if temp<Tc else 'c'
        folder = data_folder_prefix+'/{}x{}x{}/T{}_ss{}_{}'.format(ss,ss,ss,temp,ss,phase)
        ## for polarization we should constraint the global dipole. Because the susceptibility is define 
        ## on the absolute value of |P_i|, not the vector (this is an analogy, essentially it is the response function). 
        ## If not, consider there are many domains, the polarization will be 0! then dP/dE=0.
        ## To actually calculate the response function, the most convenient is constraining the global dipole. So we limit the polarizatoin in only one 
        ## section instead of the symmetric two. So automatically we counted \sum|P_i| instead of \sum P_i
        thermo = simultaneous_read(folder, throw, phase,restrain=True)
        thermo = compute_susceptibility(thermo, temp)
        thermo_list.append(thermo)
        thermo_ss_list.append(ss)
        
        if phase == 't':
            polarization = (thermo['dpc']/thermo['Volume']) * e2uC / Atcm**2  ## uC/cm^2
            polarizatioLong.append(polarization.mean())
            suscept_long.append(thermo['chi_c'])
            suscept_trans.append((thermo['chi_a']+thermo['chi_b'])/2)
        else:
            polarization = (thermo['dpz']/thermo['Volume']) * e2uC / Atcm**2  ## uC/cm^2
            polarizatioLong.append(polarization.mean())
            suscept = (thermo['chi_z'] + thermo['chi_x']+thermo['chi_y'])/3
            suscept_long.append(suscept)
            suscept_trans.append(suscept)
        enthalpy_kT = (thermo['TotEng'] + P0 * thermo['Volume'] * barA32eV) / kb / temp # no unit
        pyrocurrent.append(
            ((enthalpy_kT*polarization).mean() - polarization.mean()*enthalpy_kT.mean()) / temp
            )

        print('folder',folder)
        print('time:{}ps,nframe:{}'.format(thermo['time'][-1],thermo['time'].shape[0]))
        print('chi_x={:.0f},chi_y={:.0f},chi_z={:.0f}'.format(thermo['chi_x'],thermo['chi_y'],thermo['chi_z']))
        print('suscept_long=',suscept_long[-1])
        print('pyro coef=',pyrocurrent[-1])


arr_temp = np.array(arr_temp)
arr_tetra = np.array([thermo['c'].mean()/thermo['a'].mean() for thermo,ss in zip(thermo_list,thermo_ss_list)])
arr_polarizatioLong = np.array(polarizatioLong)
arr_pyrocurrent = - np.array(pyrocurrent)
arr_suscept_long = np.array(suscept_long)
arr_suscept_trans = np.array(suscept_trans)

################## PLOTTING##################
'''
subplots(1,2,figsize=(10,4)) :   polarization feature
ax1: polarization vs temperature
ax2: susceptibility  vs temperature 
'''
### spontaneous polarization vs temperature
for idx,ss in enumerate(ss_list):
    filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if current_ss==ss ]
    ax[0,2].plot(arr_temp[filter], arr_polarizatioLong[filter], label=r'$L={}$'.format(ss))
    print(arr_polarizatioLong[filter])
ax[0,2].set_xlabel(r'$T$ [K]',fontdict=font)
ax[0,2].set_ylabel(r'$\mathcal{P}$ [$\mu$C/cm${}^2$]',fontdict=font)
ax[0,2].legend(fontsize=12,frameon=False,loc='upper right')
### pyrocurrent
inset_loc = [0.2, 0.2, 0.4, 0.4]
ins = ax[0,2].inset_axes(inset_loc)
ss= ss_list[-1]
filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if current_ss==ss and arr_temp[idx]<822]
ins.plot(arr_temp[filter], np.log10(arr_pyrocurrent[filter]), color='green',label=r'$L={}$'.format(ss))
# ins.set_yscale('log')
ins.set_xlabel(r'$T$ [K]',fontdict=inset_font)
ins.set_ylabel(r'$\log|dP/dT|$ [$\mu$C/cm${}^2$/K]',fontdict=inset_font)
ins.tick_params(axis="y", labelsize=inset_label_size)
ins.tick_params(axis="x", labelsize=inset_label_size)
ins.legend(fontsize=12,frameon=False )
### susceptibility  vs temperature
# expdata = np.loadtxt('./exp_data/dielectric_exp.txt')
cool_diel = np.loadtxt('./exp_data/dielectric_cooling.txt')
heat_diel = np.loadtxt('./exp_data/dielectric_heating.txt')

# temp_exp = expdata[:,0] 
# suscept_exp = expdata[:,1] -1 
cool_temp = cool_diel[:,0] + 273.15
cool_suscept = cool_diel[:,1] -1 
heat_temp = heat_diel[:,0] + 273.15
heat_suscept = heat_diel[:,1] -1 
for idx,ss in enumerate(ss_list):
    filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if current_ss==ss and arr_temp[idx]>500]
    linewidth = 2 if ss==ss_list[-1] else 0
    ax[1,2].plot(arr_temp[filter], arr_suscept_long[filter], linewidth=linewidth,label=r'$L={}$'.format(ss))
ax[1,2].plot(cool_temp[8:], cool_suscept[8:],label=r'EXP ($T\nearrow$)',marker='s')
ax[1,2].plot(heat_temp[8:], heat_suscept[8:],label=r'EXP ($T\searrow$)',marker='d')

    # ax2.plot(arr_temp[filter], arr_suscept_trans[filter], marker='x',linewidth=2,label=r'Transverse, $L={}$'.format(ss))
######## curie weiss
inset_loc = [0.15, 0.55, 0.32, 0.35]
ins = ax[1,2].inset_axes(inset_loc)
ss= ss_list[-1]
filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if (current_ss==ss and arr_temp[idx]>=822 and arr_temp[idx] < 950)]
cw_temp = arr_temp[filter]
cw_suscept = arr_suscept_long[filter]
coef = polyfit(1/cw_suscept, cw_temp ,deg=1 )
print(coef)
try_temp = np.array([500,1000])
## the right half of X^-1
ins.plot(cw_temp, 1000/cw_suscept, linestyle='-',color='green',linewidth=0 )
ins.plot(try_temp, 1000/coef[0]*(try_temp-coef[1]), linestyle='-',color='green',linewidth=1.5,markersize=0 )
ins.annotate(r'$C={:.1f}\times 10^5$K'.format(coef[0]/10**5), xy=(0.05,0.85),xycoords='axes fraction', fontsize=11)
ins.annotate(r'$T_\theta={:.0f}$K'.format(coef[1]), xy=(0.05,0.65),xycoords='axes fraction', fontsize=11)
ins.set_xlabel(r'$T$ [K]',fontdict=inset_font)
ins.set_ylabel(r'$\chi_l^{-1}$ [$10^{-3}$]',fontdict=inset_font)
ins.set_xlim(800,920)
ins.set_ylim(0.0,1.0)
ins.tick_params(axis="y", labelsize=inset_label_size)
ins.tick_params(axis="x", labelsize=inset_label_size)
# ax2p = ax2.twinx()
# ss= ss_list[-1]
# filter = [idx for idx, current_ss in enumerate(thermo_ss_list) if current_ss==ss and arr_temp[idx]>700]
# ax2p.plot(arr_temp[filter], 1/arr_suscept_long[filter]*1000, linestyle='dotted',color='green',linewidth=2,markersize=0,label=r'$L={}$'.format(ss))
# ax2p.set_ylabel(r'$\chi_l^{-1}$ [$10^{-3}$]',fontdict=font)
# ax2p.legend(fontsize=12,frameon=False,loc='lower right')
#########
# ax2.set_yscale('log')
ax[1,2].set_xlabel(r'$T$ [K]',fontdict=font)
ax[1,2].set_ylabel(r'$\chi_l$',fontdict=font)
ax[1,2].legend(fontsize=12,frameon=False,loc='upper right')
ax[1,2].set_xlim(450,1100)
### clean up and save
ax[0,2].set_title('(e)',loc='left', fontsize=15)
ax[1,2].set_title('(f)',loc='left', fontsize=15)

plt.tight_layout()
plt.savefig('paper-MD.png',dpi=300)
plt.close()   


