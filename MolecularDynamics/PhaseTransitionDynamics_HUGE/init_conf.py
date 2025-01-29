from fse.systems import perovskite as perovskite
import numpy as np
import ase
import ase.io
pto_factory = perovskite(['Pb','Ti','O'])

def get_region_filter(pos, bounds=[None,None,None,None, 0.0, 2.0]):
    '''
    Parameters:
    bounds: [x_ub, x_lw, y_ub,y_lb, z_ub,z_lb]
    Returns:
    A filter
    '''
    ### find filter 
    region_filter = np.ones(pos.shape[0], dtype=bool)
    if bounds[0] is not None:
        region_filter = region_filter & (pos[:,0] < bounds[0]) 
    if bounds[1] is not None:
        region_filter = region_filter & (pos[:,0] > bounds[1]) 
    if bounds[2] is not None:
        region_filter = region_filter & (pos[:,1] < bounds[2])
    if bounds[3] is not None:
        region_filter = region_filter & (pos[:,1] > bounds[3]) 
    if bounds[4] is not None:
        region_filter = region_filter & (pos[:,2] < bounds[4])
    if bounds[5] is not None:
        region_filter = region_filter & (pos[:,2] > bounds[5]) 
    return region_filter
def create_nucleus(supercell, reverse_region=None, a=3.91,c=4.1 ):
    '''
    lousy way to create a large supercell with reversed domain and extra epitaxial layers
    '''
    sc = supercell
    if reverse_region is None:
        reverse_region = np.zeros(sc,dtype=int)
        reverse_region[:sc[0]//2,:sc[1]//2,:sc[2]//2] += 1
    reverse_region = reverse_region.astype(bool)
    ##  get perfect tetra configuration
    system = pto_factory.create_tetra_domain(sc, a ,c)
    pos = system.get_positions()
    syms = np.array(system.get_chemical_symbols())
    Ti_lattice  = pto_factory.get_effective_lattice(sc, system, central_element='Ti')
    ## locate reverse region in Cartesian coordinates
    Ti_reversed_idx = Ti_lattice[reverse_region]
    reversed_region_upper_bound = pos[Ti_reversed_idx].max(0) + 3
    assert reversed_region_upper_bound.shape == np.arange(3).shape
    reversed_region_lower_bound = pos[Ti_reversed_idx].min(0) - 3
    print('reverse region: Upper bound {}; Lower bound {}'.format(reversed_region_upper_bound, reversed_region_lower_bound))
    ### get filter for the reversed region 
    reversed_filter = get_region_filter(pos, bounds=[
        reversed_region_upper_bound[0], reversed_region_lower_bound[0],
        reversed_region_upper_bound[1], reversed_region_lower_bound[1],
        reversed_region_upper_bound[2], reversed_region_lower_bound[2],
    ] )
    print('reverse #atoms={}'.format(reversed_filter.astype(int).sum() ))
    O_filter = reversed_filter & (syms == 'O')
    Ti_filter = reversed_filter & (syms == 'Ti')
    print('reverse #O={}; reversed #Ti={}'.format(O_filter.astype(int).sum(), Ti_filter.astype(int).sum()))
    ### reverse the domain
    pos[Ti_filter,-1]  -= 0.25
    pos[O_filter,-1]  -= 0.4*2 
    pos[reversed_filter,-1] += 0.56
    system.set_positions(pos)
    # ### create epitaxy layer and paste to the bottom
    # epitaxy_layer1 = pto_factory.create_cubic_domain(supercell=[sc[0],sc[1],1], a=a)
    # system.set_positions(system.get_positions()+np.array([0,0,a]))
    # new_cell = system.get_cell()
    # new_cell[-1] += np.array([0,0,a])
    # system.set_cell(new_cell)
    # system.extend(epitaxy_layer1)
    # ### cubify and move the top Ti-O plane to the bottom
    # pos = system.get_positions()
    # pos_max = pos[:,-1].max()
    # epitaxy_ub = pos_max + 1
    # epitaxy_lb = pos_max - 1
    # epitaxy_filter = get_region_filter(pos, bounds=[ None,None,None,None,epitaxy_ub,epitaxy_lb ])
    # pos[:,-1] += a/2
    # pos[epitaxy_filter,-1]  *= 0
    # system.set_positions(pos)
    return system, Ti_lattice

if __name__ == "__main__":
    pto_factory = perovskite(['Pb','Ti','O'], born_charges=[3.7140, 5.4879, -3.3551, -2.9234])
    sc = [512, 16, 16]
    nc = [256, 16, 16]
    
    reverse_region = np.zeros(sc,dtype=int)
    reverse_region[:nc[0],:,:] += 1

    atoms, latt = create_nucleus(sc, reverse_region)
    ase.io.write('./template/L{}x{}x{}_twin.lmp'.format(sc[0],sc[1],sc[2]),  atoms, format='lammps-data')

