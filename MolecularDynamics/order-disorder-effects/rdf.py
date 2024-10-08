import os
import numpy as np
import sys
import ase 
import ase.io
from ase.geometry.analysis import Analysis
from matplotlib import pyplot as plt
 
ss=15

rmax=5
nbins=500
distance = np.linspace(0,rmax,nbins+1)[:-1]
distance += (distance[1] - distance[0])/2
fig,ax = plt.subplots(1,figsize=(4*1.1,3*1.1))
# for temp, phase in zip([300, 820, 822, 900],['t','t','c','c',]):
for temp, phase in zip([ 900],[ 'c',]):

    folder = '/home/pinchenx/data.gpfs/ferro_scratch/PTO/DPMD/final_susceptibility_press/{}x{}x{}/T{}_ss{}_{}'.format(ss,ss,ss,temp,ss,phase)
    print('Loading')
    ase_atoms = ase.io.read(os.path.join(folder,'pto.lammpstrj'), format = 'lammps-dump-text',index='10:100')
    print('Finish Loading')
    ana = Analysis(ase_atoms)
    rdf = ana.get_rdf(rmax, nbins=nbins, imageIdx=None, elements='He', return_dists=False)
    nframes = len(rdf)
    rdf = sum(rdf) / nframes
    np.save( 'T{}L{}_rdf.npy'.format(temp, ss), rdf)
    # ax.plot(distance[distance > 3], rdf[distance > 3], label='T={}K'.format(temp))
 