import ase
import ase.io
import os
from ase.geometry import get_distances
import numpy as np
from numpy import linalg
import warnings


def norm(x):
    return linalg.norm(x,axis=-1)

class ferroelectric:
    def __init__(self, system):
        atoms, wcs = self.load_wannier(system)
        self.atoms = atoms
        self.natoms = atoms.get_global_number_of_atoms()
        self.wcs = wcs
        self._mapping = None
        self.rwcs = self.reduced_wannier(wcs)
    
    ############## initialization  ##############
    def tagging_wc(self, atm, wc, cell, pbc):
        '''
        tagging wannier centers
        Args 
            atm (1D/2D np.array, float): cartesian coordinates of atoms
            wc (1D/2D np.array, float): cartesian coordinates of wannier centers
            cell (2D np.array, float)
            pbc (1D np.array, bool)
        Returns
            tag_wc (1D np.array, int) of size (nwcs): the index of the atom that is associated to wannier center
            wc_shifted (2D np.array, float)
        '''
        D, D_len = get_distances(atm, wc, cell, pbc)
        tag_wc =  np.argmin(D_len, axis=0)
        wc_shifted = atm[tag_wc] + D[tag_wc, np.arange(tag_wc.size)]
        return tag_wc, wc_shifted
    
    def load_wannier(self, system):
        '''
        Args:
            system (ase.atoms): ase-Atoms including all atoms and wannier centers(symbol:X)
        Returns:
            atom_system (ase.atoms): ase-Atoms of all atoms.
            wc_system (ase.atoms): ase-Atoms of all wannier centers(WCs). All WCs are moved into 
            the neighborhood of its home atom. wc_systems.get_tags() gives the index of the home atom in 'atom_system'.
        '''
        cell = np.array(system.get_cell())
        pbc = system.get_pbc()
        anames = system.get_chemical_symbols()
        filter_wc = np.array(anames) == np.array(['X']*len(anames))
        if filter_wc.astype(int).sum() == 0:
            raise ValueError('No wannier center is detected!')
        filter_atm = ~filter_wc 

        pos_atm = system.get_positions(wrap=True)[filter_atm]
        pos_wc = system.get_positions(wrap=True)[filter_wc]

        # tag_wc, pos_wc_shifted = self.naive_tagging_wc(pos_atm, pos_wc, cell)
        tag_wc, pos_wc_shifted = self.tagging_wc(pos_atm, pos_wc, cell, pbc)
        atom_system = ase.Atoms(
            tags = np.arange(pos_atm.shape[0]),
            symbols = np.array(anames)[filter_atm], 
            positions = pos_atm,
            charges = system.numbers[filter_atm],
            cell=cell,
            pbc=pbc)
        wc_system = ase.Atoms(
            tags = tag_wc, 
            positions = pos_wc_shifted,
            charges = - 2 * np.ones_like(tag_wc),
            cell=cell,
            pbc=pbc)
        # print(wc_system.symbols)
        # print(wc_system.get_tags())
        return atom_system, wc_system

    def reduced_wannier(self, wc_system):
        '''
        Reduce wannier centers associated to the same home atom to one effective wannier center.
        The charge of reduced wannier centers is accessed through reduced_wc_system.get_initial_charges()
        Args:
            wc_system (ase.atoms)
        Returns:
            reduced_wc_system (ase.atoms)
        '''
        reduced_wcs = []
        charges = []

        tag_wc = wc_system.get_tags()
        pos_wc = wc_system.get_positions()
        tag_set = list(set(tag_wc))
        
        for l in tag_set:
            filter_l = tag_wc == np.array([l]*len(tag_wc))
            charges.append(filter_l.astype(int).sum())
            reduced_wcs.append(pos_wc[filter_l].mean(0))
        
        reduced_wc_system = ase.Atoms(
            tags = tag_set, 
            positions=np.array(reduced_wcs),
            charges = - 2 * np.array(charges),
            cell=np.array(wc_system.get_cell()), 
            pbc=wc_system.get_pbc())

        self._mapping = np.zeros(self.natoms, dtype=int) - 1
        for ii, tag in enumerate(tag_set):
            self._mapping[tag]= ii
        if self._mapping.min() < 0:
            warnings.warn("there is atom without Wannier centroid")
        return reduced_wc_system
    ############# attributes getter #############
    def get_chemical_symbols(self):
        return self.atoms.get_chemical_symbols()
    
    def get_cell(self):
        return self.atoms.get_cell()

    def get_positions(self):
        return self.atoms.get_positions()

    def get_displacement(self, atom_system, wc_system, atom_type=None):
        '''
        Args: 
            wc_system(ase.atoms)
            atom_type(str): chemical symbol, e.g. 'Pb'
        Returns:
            displacement(np.array): the minimized displacement of selected wannier centers away from their homeatom. 
        '''
        homeatom_idx = wc_system.get_tags()
        homeatom_type = np.array(atom_system.get_chemical_symbols())[homeatom_idx]
        if atom_type == None:
            _filter = np.ones(wc_system.get_global_number_of_atoms(),dtype=bool)
        else:
            _filter = homeatom_type == np.array([atom_type]*len(homeatom_type))
        displacement, D_len = get_distances(
            atom_system.get_positions()[homeatom_idx], 
            wc_system.get_positions(), 
            atom_system.get_cell(),
            atom_system.get_pbc()
            )
        nwc = wc_system.get_global_number_of_atoms()
        displacement = displacement[np.arange(nwc),np.arange(nwc)]
        # displacement_ref =  wc_system.get_positions() - atom_system.get_positions()[homeatom_idx]
        return displacement[_filter]

    def get_wcs_displacement(self, atom_type=None):
        '''
        Args: 
            atom_type(str): chemical symbol, e.g. 'Pb'
        Returns:
            displacement(np.array): the minimized displacement of selected wannier centers away from their homeatom. 
        '''
        return self.get_displacement(self.atoms, self.wcs, atom_type)

    def get_wcs_radial_displacement(self, atom_type=None):
        displacement = self.get_wcs_displacement(atom_type)
        return norm(displacement)

    def get_rwcs_displacement(self, atom_type=None, sort = False):
        '''
        Args: 
            atom_type(str): chemical symbol, e.g. 'Pb'
            sort: aligh the tag of reduced wannier center with the labels of home atoms.
        Returns:
            displacement(np.array): the minimized displacement of selected reduced wannier centers away from their homeatom: r(wc)-r(home atom)
        '''
        displacement = self.get_displacement(self.atoms, self.rwcs, atom_type=None)
        symbols = self.atoms.get_chemical_symbols()
        if sort:
            homeatom_idx = self.rwcs.get_tags()
            sorted_disp = np.zeros_like(displacement)
            assert len(set(homeatom_idx)) == self.natoms, 'some atom does not have associated reduced wannier center'
            sorted_disp[homeatom_idx] = displacement
            displacement = sorted_disp
        else:
            symbols = symbols[homeatom_idx]
        if atom_type is None:
            type_list = list(set(symbols))
        elif isinstance(atom_type,str):
            type_list = [atom_type]
        elif isinstance(atom_type,list) and isinstance(atom_type[0],str):
            type_list = atom_type
        type_filter = [(s in type_list) for s in symbols]
        return displacement[type_filter]

class PTO(ferroelectric):
    def __init__(self, system):
        super().__init__(system)
        self.valence_charges = {'Pb': 14, 'Ti': 12, 'O': 6}
        self._platt_list = None
        self._nlatt_list = None
    def _update_cell_list(self, center_type='Ti'):
        '''
        For ABO3 type crystals, locate all A(or B)-centered atomic unit cell and return a list of cell dictionary.
        TODO: make this compatible with any perovskites.
        Args:
            center_type: 'Pb' or 'Ti'
        Returns: 
            cell_list (list): a list of dictionary containing the information of every aotmic unit cell
        '''
        if center_type not in ['Pb','Ti']:
            raise NotImplementedError('Only PbTiO3 is supported')
        vertex_type = 'Pb' if center_type=='Ti' else 'Ti'
        fcenter_type = 'O'
        c_weight = 1
        nv = 8
        v_cutoff = 6
        v_weight = 1/8
        nf = 6 if center_type == 'Ti' else 12
        f_cutoff = 4
        f_weight = 3 / nf

        N = self.natoms
        center_atom_filter = self.get_chemical_symbols() == np.array([center_type] * N)
        vertex_atom_filter = self.get_chemical_symbols() == np.array([vertex_type] * N)
        fcenter_atom_filter = self.get_chemical_symbols() == np.array([fcenter_type] * N)

        center_atom_idx = np.arange(N)[center_atom_filter]
        vertex_atom_idx = np.arange(N)[vertex_atom_filter]
        fcenter_atom_idx = np.arange(N)[fcenter_atom_filter]

        cell_list = []
        for idx in center_atom_idx:
            ## locate the nearest vertex atoms
            vertex_distance = self.atoms.get_distances(idx, vertex_atom_idx, mic=True)
            sort_idx = np.argsort(vertex_distance)[:nv]
            if vertex_distance[sort_idx].max() > v_cutoff:
                warn_text = "max vertex-center distance is {}, cutoff is {}".format(vertex_distance[sort_idx].max(), v_cutoff)
                warnings.warn(warn_text)
                warnings.warn(center_type)
            vertex_shell_idx = vertex_atom_idx[sort_idx]
            ## locate the nearest face center atoms
            fcenter_distance = self.atoms.get_distances(idx, fcenter_atom_idx, mic=True)
            sort_idx = np.argsort(fcenter_distance)[:nf]
            if fcenter_distance[sort_idx].max() > f_cutoff:
                warn_text = "max oxygen-center distance is {}, cutoff is {}".format(fcenter_distance[sort_idx].max(), f_cutoff)
                warnings.warn(warn_text)
                warnings.warn(center_type)
            fcenter_shell_idx = fcenter_atom_idx[sort_idx]
            cell_list.append({
                'cell_type': 'Atomic',
                'center_type': center_type,
                'center_idx': idx,
                'center_charge': self.valence_charges[center_type],
                'center_weight': c_weight,
                
                'vertex_type': vertex_type,
                'vertex_idx': vertex_shell_idx,
                'vertex_charge': np.ones_like(vertex_shell_idx) * self.valence_charges[vertex_type],
                'vertex_weight': np.ones_like(vertex_shell_idx) * v_weight,
                
                'fcenter_type': fcenter_type,
                'fcenter_idx': fcenter_shell_idx,
                'fcenter_charge': np.ones_like(fcenter_shell_idx) *  self.valence_charges[fcenter_type],
                'fcenter_weight': np.ones_like(fcenter_shell_idx) * f_weight,
    
            })
        return cell_list
    
    def _update_wc_cell_list(self, cell_list):
        '''
        Get the list of associated wannier center unit cell from the list of atomic unit cell
        Args:
            cell_list: a list of dictionary containing the information of aotmic unit cell
        Returns: 
            wc_cell_list (list): a list of dictionary containing the information of wannier center unit cell
        '''
        if cell_list is None:
            raise AssertionError('should initialize atomic cell-lattice first.')
        wc_cell_list = []
        wc_charges = self.rwcs.get_initial_charges()
        for cell in cell_list:
            center_idx = self._mapping[cell['center_idx']]
            vertex_idx = self._mapping[cell['vertex_idx']]
            fcenter_idx = self._mapping[cell['fcenter_idx']]
            wc_cell_list.append({
                'cell_type': 'Wannier',
                'center_type': cell['center_type'],
                'center_idx': center_idx,
                'center_charge': wc_charges[center_idx],
                'center_weight': cell['center_weight'],

                'vertex_type': cell['vertex_type'],
                'vertex_idx': vertex_idx,
                'vertex_charge': wc_charges[vertex_idx],
                'vertex_weight': cell['vertex_weight'],
                
                'fcenter_type': cell['fcenter_type'],
                'fcenter_idx': fcenter_idx,
                'fcenter_charge': wc_charges[fcenter_idx],
                'fcenter_weight': cell['fcenter_weight'],
            })
        return wc_cell_list
    
    def get_charge_center(self, cell_dict):
        '''
        Get the averaged charge center of a given cell. 
        Charge shared with neighboring cell will be weighted accordingly
        Args:
            cell_dict (dict):
        Returns
            cell_charge (float): effective charge of a cell
            charge_center (np.array): coordinate of the charge center of a cell
        '''
        if cell_dict['cell_type'] == 'Atomic':
            base_system = self.atoms
        elif cell_dict['cell_type'] == 'Wannier':
            base_system = self.rwcs
        else:
            raise ValueError('cell type not recognized')

        cell = base_system.get_cell()
        pbc = base_system.get_pbc()
        positions = base_system.get_positions()
        c_pos = positions[cell_dict['center_idx']]
        c_charge = np.atleast_1d(cell_dict['center_charge']*cell_dict['center_weight'])
        c_displacement = np.atleast_2d(np.zeros_like(c_pos))

        v_pos = positions[cell_dict['vertex_idx']]
        v_charge = cell_dict['vertex_charge'] * cell_dict['vertex_weight']
        v_displacement, v_len = get_distances(c_pos, v_pos, cell, pbc)
        
        f_pos = positions[cell_dict['fcenter_idx']]
        f_charge = cell_dict['fcenter_charge'] * cell_dict['fcenter_weight']
        f_displacement, f_len = get_distances(c_pos, f_pos, cell, pbc)
        ## get_distances always return 3D array
        d_collection = np.concatenate([c_displacement,  v_displacement[0], f_displacement[0]])
        w_collection = np.concatenate([c_charge,  v_charge, f_charge])

        total_charge = w_collection.sum()
        total_displacement = (d_collection * w_collection[:,None]).sum(0) / total_charge
        return total_charge, c_pos + total_displacement

    def get_platt_list(self):
        if self._platt_list is None:
            raise ValueError("positive lattice list is not avaliable. Execute 'get_dipole_moment' first")
        else:
            return self._platt_list
    
    def get_nlatt_list(self):
        if self._nlatt_list is None:
            raise ValueError("negative lattice list is not avaliable. Execute 'get_dipole_moment' first")
        else:
            return self._nlatt_list        
    def get_dipole_sublattice(self, center_type='Ti'):
        '''
        Args:
            Center_type: 'Pb' or 'Ti'
        Returns:
            ref_system: sublattice of the center atom
            platt_system: sublattice of the positive charge center of each effective cell
            nlatt_system: sublattice of the negative charge center of each effective cell
        '''
        platt_list = self._update_cell_list(center_type)
        self._platt_list = platt_list
        nlatt_list = self._update_wc_cell_list(platt_list)
        self._nlatt_list = nlatt_list
        platt = [self.get_charge_center(pcell) for pcell in platt_list]
        nlatt = [self.get_charge_center(ncell) for ncell in nlatt_list]
        
        box = self.atoms.get_cell() 
        pbc = self.atoms.get_pbc()
        ref_pos = self.atoms.get_positions()

        ref_system = ase.Atoms(
            symbols = [x['center_type'] for x in platt_list],
            positions = [ref_pos[x['center_idx']] for x in platt_list],
            cell=box,
            pbc=pbc)

        platt_system = ase.Atoms(
            symbols = [x['center_type'] for x in platt_list],
            positions = [x[1] for x in platt],
            charges = [x[0] for x in platt],
            cell=box,
            pbc=pbc)

        nlatt_system = ase.Atoms(
            symbols = [x['center_type'] for x in nlatt_list],
            positions = [x[1] for x in nlatt],
            charges = [x[0] for x in nlatt],
            cell=box,
            pbc=pbc)

        return ref_system, platt_system, nlatt_system

    def _compute_dipole(self,platt_system, nlatt_system):
        platt_pos = platt_system.get_positions()
        nlatt_pos = nlatt_system.get_positions()
        dipole = (platt_pos - nlatt_pos) * platt_system.get_initial_charges()[:,None] # eA
        return dipole

    def get_wannier_displacement(self, center_type='Ti'):
        ref_system, platt_system, nlatt_system = self.get_dipole_sublattice(center_type)
        ref_pos = ref_system.get_positions()
        nlatt_pos = nlatt_system.get_positions()
        wannier_displacement = (nlatt_pos - ref_pos)  # A
        return wannier_displacement

    def get_global_wannier_displacement(self, center_type='Ti'):
        wannier_displacement = self.get_wannier_displacement(center_type='Ti')
        return wannier_displacement.sum(0)

    def get_dipole_moment(self, center_type='Ti'):
        '''
        Args:
            Center_type: 'Pb' or 'Ti'
        Returns:
            dipole(2d np.array): effective cell dipole moment. 
        '''
        ref_system, platt_system, nlatt_system = self.get_dipole_sublattice(center_type)
        return self._compute_dipole(platt_system, nlatt_system)

    def get_global_dipole_moment(self, center_type='Ti'):
        dipole = self.get_dipole_moment(center_type)
        return dipole.sum(0)

if __name__ == '__main__':
    # folder = './T1200/wannier/1/'
    folder = 'cubic.300.task.000000.tric.mKpoint/'
    system = ase.io.read(os.path.join(folder,'PTO.wout'))
    pto = PTO(system)
    print(pto.get_cell())
    
    ref_system, platt_system, nlatt_system = pto.get_dipole_sublattice(center_type='Ti')

    # ase.io.write('atoms.lmp',pto.atoms, format='lammps-data')
    # ase.io.write('platt.lmp',platt_system, format='lammps-data')
    # ase.io.write('nlatt.lmp',nlatt_system, format='lammps-data')

    atomic_center = ref_system.get_positions()
    neutral_center = (platt_system.get_positions() + nlatt_system.get_positions())/2
    dipole_vec = platt_system.get_positions() - nlatt_system.get_positions()

    dx = dipole_vec[:,0]
    dy = dipole_vec[:,1]
    dz = dipole_vec[:,2]

    # print(norm(neutral_center-atomic_center))
    # print(norm(dipole_vec))
    print(pto.get_global_dipole_moment())

    # print( atom_system.get_positions()[x['center_idx']] )
    # print( atom_system.get_positions()[x['vertex_idx']] )
    # print( atom_system.get_positions()[x['oxygen_idx']] )

    folder = '/home/pinchenx/ferro/DeepWannier/PTO.init/02.md/cubic/300K/wannier/task.000000/'
    # folder = 'cubic.300.task.000000.tric.mKpoint/'
    system = ase.io.read(os.path.join(folder,'PTO.wout'))
    pto = PTO(system)
    print(pto.get_cell())
    
    ref_system, platt_system, nlatt_system = pto.get_dipole_sublattice(center_type='Ti')

    # ase.io.write('atoms.lmp',pto.atoms, format='lammps-data')
    # ase.io.write('platt.lmp',platt_system, format='lammps-data')
    # ase.io.write('nlatt.lmp',nlatt_system, format='lammps-data')

    atomic_center = ref_system.get_positions()
    neutral_center = (platt_system.get_positions() + nlatt_system.get_positions())/2
    dipole_vec = platt_system.get_positions() - nlatt_system.get_positions()

    dx = dipole_vec[:,0]
    dy = dipole_vec[:,1]
    dz = dipole_vec[:,2]

    # print(norm(neutral_center-atomic_center))
    # print(norm(dipole_vec))
    print(pto.get_global_dipole_moment())