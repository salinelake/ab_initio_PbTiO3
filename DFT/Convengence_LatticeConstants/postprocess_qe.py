# import re
import numpy as np
from matplotlib import pyplot as plt
import ase
import ase.io

ry2ev = 13.605693009

def get_energy(file):
    energy = None
    bandgap = None
    with open(file,'r') as lines:
        for ii in lines :
            if '!    total energy' in ii :
                energy = ry2ev *  float(ii.split('=')[1].split()[0])
            # if 'highest occupied' in ii:
            #     level = ii.split(':')[1].split()
            #     bandgap = float(level[1])-float(level[0])
    return energy#, bandgap


def get_bp_system(file):
    '''
    Args: 
        file(str) - file name.
    Returns:
        (dictionary) -
            anames(list): name of all atoms
            cell(np.array): 1D array of cell size
            positions(np.array): (natoms*3) array of atomic positions
            charges(np.array): #valency electrons of all atoms
            pi(float): Ionic Phase
            pe(float): Electronic Phase
            p(float): TOTAL PHASE
    '''
    with open(file, 'r') as f:
        data = [line.strip() for line in f]
    structure = ase.io.read(file)
    natoms = structure.get_global_number_of_atoms()
    cell = np.array(structure.get_cell())
    positions = structure.get_positions()
    for i, line in enumerate(data):
        if 'Charge' in line:
            start = i+2
            charge_info = data[start:start+natoms]
            anames = [x.split()[1] for x in charge_info]
            charges = np.array([x.split()[2] for x in charge_info]).astype(float)
        if 'Ionic Phase' in line:
            # pi = re.findall("[-+]?\d+\.\d+", line)[0]
            pi = float(line.split()[2])
        if 'Electronic Phase' in line:
            # pe = re.findall("[-+]?\d+\.\d+", line)[0]
            pe = float(line.split()[2])
        if 'TOTAL PHASE' in line:
            # p = re.findall("[-+]?\d+\.\d+", line)[0]
            p = float(line.split()[2])
    return {
        'anames': anames,
        'cell': cell,
        'positions': positions,
        'charges': charges,
        'ephase': pe,
        'iphase': pi,
        'tphase': p,
        }



mode = 'l'

if mode == 'l':
    l_list = np.arange(11)*0.02 + 3.82
    berry_el = []  ## mod 2
    # for kpts in [4, 6]:
    for kcut in [60, 80, 100, 120 ]:
        # kcut = 80
        kpts=4
        prefix = 'cubic_kpts{}_kcut{}'.format(kpts, kcut)
        energy = []
        bandgap = []
        for i in range(len(l_list)):
            qe_out = './{}/{}/qe.out'.format(prefix, i)
            bp_out = './{}/{}/berry.out'.format(prefix, i)
            print('processing {}-th trajectory from {}'.format(i,qe_out))
            E = get_energy(qe_out)
            energy.append(E)
            # bandgap.append(Egap)
            # bp_system = get_bp_system(bp_out)
            # pi = bp_system['iphase']
            # pe = bp_system['ephase']
            # p = bp_system['tphase']
            # berry_el.append(pe)
        ## plot energy vs lattice parameter
        print(energy)
        energy = np.array(energy) - np.array(energy).min()
        plt.plot(l_list, energy,label='kpts={},ecutwfc={}Ry'.format(kpts,kcut))
    plt.xlabel('a/A')
    plt.ylabel(r'$E-E_{min}/eV$')
    plt.legend()
    plt.savefig('./l_vs_E_cubic_kpts{}'.format(kpts) )
    # # plot bandgap vs lattice parameter
    # plt.figure()
    # plt.plot(l_list, bandgap,label='ecutwfc={}Ry,ecutrho={}Ry'.format(cutoff,cutoff*charge))
    # plt.xlabel('a/A')
    # plt.ylabel('E/eV')
    # plt.legend()
    # plt.savefig('./l_vs_dE_cubic_kmesh8_koffset0.png')


        # berry_el = np.array(berry_el)
        # print((berry_el[1:] - berry_el[:-1])*5)
        # plt.figure()
        # plt.plot(l_list, berry_el,'o-')
        # plt.xlabel('cell/A')
        # plt.ylabel('berry_el mod 2')
        # plt.savefig('./l_vs_berry_{}.png'.format(prefix))

else:
    d_list = 0.0025 * np.arange(10)
    energy = []
    berry_el = []  ## mod 2
    #### analyze displacement vs energy
    prefix = 'tetra'
    for i in range(len(d_list)):
        qe_out = './KNbO3/{}/{}/qe.out'.format(prefix, i)
        bp_out = './KNbO3/{}/{}/berry.out'.format(prefix, i)
        print('processing {}-th trajectory from {}'.format(i,qe_out))
        qe_system = dpdata.LabeledSystem(qe_out,fmt='qe/pw/scf')   ## collect all coordinates from dump files
        # displace.append(qe_system['coords'][0,0,2]-qe_system['coords'][0,0,0])
        energy.append(qe_system['energies'][0])
        bp_system = get_bp_system(bp_out)
        pi = bp_system['iphase']
        pe = bp_system['ephase']
        p = bp_system['tphase']
        berry_el.append(pe)
    energy = np.array(energy)
    berry_el = np.array(berry_el)
    print((berry_el[1:] - berry_el[:-1])*5)
    plt.plot(d_list, (energy-energy.mean())*1000)
    plt.xlabel('displacement/A')
    plt.ylabel('E/meV')
    plt.savefig('./d_vs_E_{}.png'.format(prefix))

    plt.figure()
    plt.plot(d_list, berry_el,'o-')
    plt.xlabel('displacement/A')
    plt.ylabel('berry_el mod 2')
    plt.savefig('./d_vs_berry_{}.png'.format(prefix))


    # #### analyze lattice constant vs energy
    # for i,l in enumerate(l_list):
    #     qe_out = './KNbO3/lattice_constant/{}/qe.out'.format(i)
    #     print('processing {}-th trajectory from {}'.format(i,qe_out))
    #     qe_system = dpdata.LabeledSystem(qe_out,fmt='qe/pw/scf')   ## collect all coordinates from dump files
    #     lattice.append(qe_system['cells'][0,0,0])
    #     energy.append(qe_system['energies'][0])

    # lattice = np.array(lattice)
    # energy = np.array(energy)

    # plt.plot(lattice, (energy-energy.mean())*1000)
    # plt.xlabel('a/A')
    # plt.ylabel('E/meV')
    # plt.savefig('./l_vs_E.png')