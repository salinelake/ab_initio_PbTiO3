import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os

kjmol2mev=10.364
folder = './'
pace=250
stride=1000
dt=pace*stride*0.0005 / 1000 #ns
repeat=2
supersize=repeat*3
ncell = supersize**3
natoms= ncell * 5
folder = '.' 
fig1,ax1=plt.subplots()
fig2,ax2=plt.subplots()

counter=1
fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))
while os.path.exists(fes_dir):
    fes = np.loadtxt(fes_dir)
    filter = (fes[:,0]>=0.1)&(fes[:,0]<=2.2)
    x = fes[:,0][filter]
    F = fes[:,1][filter] /ncell*kjmol2mev
    if counter > 12:
        ax1.plot(x,F, label='t={}ns'.format((counter+1)*dt))
        ax2.plot(x,F - F.min(), label='t={}ns'.format((counter+1)*dt))
    counter += 10
    fes_dir = os.path.join(folder,'fes_{}.dat'.format(counter))

ax1.set_xlabel('|dipole| [eA]')
ax1.set_ylabel('F/L3 [meV]')
ax2.set_xlabel('|dipole| [eA]')
ax2.set_ylabel('F/L3 [meV]')
ax1.legend()
ax2.legend()

fig1.savefig(os.path.join(folder,'fes.png'))
fig2.savefig(os.path.join(folder,'fes-0min.png'))

plt.close(fig1)
plt.close(fig2)
