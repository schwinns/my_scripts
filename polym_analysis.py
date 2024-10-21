# Class for a few analyses of polymerization

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import subprocess

class PolymAnalysis():
    def __init__(self, data_file, frmt='DATA', init_file='system.in.init', settings_file='cleanedsystem.in.settings', tpr_file=None, xlink_c='1', xlink_n='7', term_n='8', cl_type='6', oh_type='10', ow_type='11', hw_type='12'):

        self.lmp = '22' # extension for the LAMMPS executable
        self.gmx = 'gmx' # name of the GROMACS executable

        # Save information about original file
        print('Initializing polyamide membrane for analysis...')
        self.filename = data_file
        self.ext = data_file.split('.')[-1]

        self.init_file = init_file
        self.settings_file = settings_file

        # Create an internal MDAnalysis Universe and atoms AtomGroup
        if tpr_file is None:
            self.universe = mda.Universe(data_file, format=frmt)
            self.tpr = None
        else:
            self.universe = mda.Universe(tpr_file, data_file)
            self.tpr = tpr_file
        self.atoms = self.universe.atoms

        if not self.ext in ['xtc', 'trr']:
            f = open(data_file, 'r')
            self.original = f.readlines()
            self.trajectory = None
        else:
            self.trajectory = self.universe.trajectory

        # Save the monomer types and prepare space for log files
        self.monomers = ['MPD', 'TMC']
        self.logs = []

        # Get information about number of atoms and monomers
        if self.ext == 'data' or tpr_file is not None:
            self.atsel = 'type'
        elif self.ext == 'gro':
            self.atsel = 'name'

        xlink_c_ag = self.universe.select_atoms('{} {}'.format(self.atsel, xlink_c))
        xlink_n_ag = self.universe.select_atoms('{} {}'.format(self.atsel, xlink_n), xlink_c=xlink_c_ag)
        term_n_ag = self.universe.select_atoms('{} {}'.format(self.atsel, term_n), xlink_c=xlink_c_ag)

        self.n_atoms = len(self.atoms)
        nMPD = int((len(xlink_n_ag) + len(term_n_ag)) / 2)
        nTMC = int(len(xlink_c_ag) / 3)
        nH2O = int(len(self.universe.select_atoms('{} {}'.format(self.atsel, ow_type))))
        self.n_monomers = [nMPD, nTMC]
        print(f'PA membrane consists of {nMPD} MPD monomers, {nTMC} TMC monomers, {nH2O} waters for a total of {self.n_atoms} atoms')

        # Store input information about atom types
        self.xlink_n_type = xlink_n
        self.xlink_c_type = xlink_c
        self.term_n_type = term_n
        self.cl_type = cl_type
        self.oh_type = oh_type
        self.ow_type = ow_type
        self.hw_type = hw_type

        # Get coordinate information
        self.coords = np.zeros((1,self.n_atoms, 3))
        self.coords[0,:,:] = self.atoms.positions
        self.coords_lims = np.zeros((2,3))
        # self.coords_lims[0,:] = self.atoms.positions.min(axis=1)
        self.coords_lims[:,0] = self.atoms.positions[:,0].min(), self.atoms.positions[:,0].max()
        self.coords_lims[:,1] = self.atoms.positions[:,1].min(), self.atoms.positions[:,1].max()
        self.coords_lims[:,2] = self.atoms.positions[:,2].min(), self.atoms.positions[:,2].max()

        # Save original charges and box size
        self.charges = self.atoms.charges
        self.box = self._get_box()

        # Get header information
        if self.ext == 'data':
            self.masses = self._get_masses(data_file)
            self._get_topology_types(data_file)
            self.density = self.calculate_density(box=True)

        print('Finished loading PA membrane for analysis!\n')


    def __str__(self):
        return f'<PolymAnalysis object with {self.n_atoms} atoms from {self.filename}>'


    def _run(self, commands):
        '''Run commands with subprocess'''
        if not isinstance(commands, list):
            commands = [commands]
        
        for cmd in commands:
            subprocess.run(cmd, shell=True)

    
    def _get_topology_types(self, file):
        '''Get topology numbers and types from a LAMMPS data file'''
        f = open(file, 'r')
        lines = f.readlines()

        for line in lines[1:]: # always skip first line

            kws = ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types']
            for kw in kws:
                if kw in line:
                    setattr(self, f"n_{kw.replace(' ', '_')}", int(line.split()[0])) 


    def _get_box(self):
        '''Get box limits from data file'''

        box = np.zeros((2,3))
        
        if self.ext == 'data':
            f = open(self.filename, 'r')
            lines = f.readlines()
            for line in lines:
                if 'xlo' in line:
                    box[0,0] = float(line.split()[0])
                    box[1,0] = float(line.split()[1])
                elif 'ylo' in line:
                    box[0,1] = float(line.split()[0])
                    box[1,1] = float(line.split()[1])
                elif 'zlo' in line:
                    box[0,2] = float(line.split()[0])
                    box[1,2] = float(line.split()[1])

        elif self.ext in ['gro', 'xtc', 'trr']:
            box[1,:] = self.universe.dimensions[:3]

        return box
    

    def _get_masses(self, file):
        '''Get atom types and masses from a LAMMPS data file'''
        f = open(file, 'r')
        lines = f.readlines()

        # Get indices of where Masses starts
        idx1 = lines.index('Masses\n')

        # Loop through masses and save in dictionary (masses[atom_type] = mass)
        masses = {}
        for line in lines[idx1+2:]:

            if len(line.split()) == 0:
                break
            else:
                atom_type = int(line.split()[0])
                mass = float(line.split()[1])
                masses[atom_type] = mass

        return masses
    
    
    def _reassign_atom_numbers(self, output='renumbered.data'):
        '''Reassign atom indices to be in numerical order'''

        from lammps import PyLammps

        # Use LAMMPS to remove atoms and reset ids
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        L.reset_atom_ids()
        L.reset_mol_ids('all')

        # write the new system to data file and reinitialize object
        L.write_data(output)
        print()
        self.__init__(output)

    
    def _read_init(self, file=None):
        '''Get styles and settings from the init file (as generated by moltemplate)'''

        if file is None: # use init file from instantiation
            file = self.init_file

        f = open(file, 'r')
        lines = f.readlines()

        # create a Settings class to save information
        self.init_settings = Settings()
        
        for line in lines:
            l = line.split()
            # if line is commented out or blank ignore
            if len(l) == 0:
                pass
            elif not l[0].startswith('#'):
                setattr(self.init_settings, l[0], l[1:])


    def _read_settings(self, file=None):
        '''Get pair, bond, angle, etc. coefficients from the settings file (as generated by moltemplate)'''

        if file is None: # use settings file from instantiation
            file = self.settings_file

        f = open(file, 'r')
        lines = f.readlines()

        # create Settings class to save information
        self.coeff_settings = Settings()

        self.coeff_settings.pair_coeff = {}
        self.coeff_settings.bond_coeff = {}
        self.coeff_settings.angle_coeff = {}
        self.coeff_settings.dihedral_coeff = {}
        self.coeff_settings.improper_coeff = {}

        print(f'Reading coefficients from {file}...')
        print('WARNING: can only read the following coefficients:')
        print('\tpair_coeff: [LJ epsilon and sigma]')
        print('\tbond_coeff: [harmonic]')
        print('\tangle_coeff: [harmonic]')
        print('\tdihedral_coeff: [fourier]')
        print('\timproper_coeff: [cvff]')

        for line in lines:
            l = line.split('#')[0].split()

            if len(l) == 0:
                pass
            elif l[0].startswith('#'):
                pass
            elif l[0] == 'pair_coeff':
                type1 = int(l[1])
                type2 = int(l[2])
                style = l[3]
                eps = float(l[4])
                sig = float(l[5])

                if type1 == type2:
                    self.coeff_settings.pair_coeff[type1] = {
                            'style' : style,
                            'epsilon' : eps,
                            'sigma' : sig     # for lj/charmm/coul/long this is the "zero-crossing distance"
                        }
                else:
                    self.coeff_settings.pair_coeff[f'{type1}-{type2}'] = {
                            'style' : style,
                            'epsilon' : eps,
                            'sigma' : sig     # for lj/charmm/coul/long this is the "zero-crossing distance"
                        }
                
            elif l[0] == 'bond_coeff':
                b = int(l[1])
                style = l[2]
                K = float(l[3])
                r = float(l[4])

                self.coeff_settings.bond_coeff[b] = {
                    'style' : style,
                    'K' : K, # kcal/mol/A^2
                    'r0' : r # Angstroms
                }

            elif l[0] == 'angle_coeff':
                ang = int(l[1])
                style = l[2]
                K = float(l[3])
                theta = float(l[4])

                self.coeff_settings.angle_coeff[ang] = {
                    'style' : style,
                    'K' : K, # kcal/mol
                    'theta0' : theta # degrees
                }

            elif l[0] == 'dihedral_coeff':
                dih = int(l[1])
                style = l[2]
                m = int(l[3])
                K = l[4::3]
                n = l[5::3] 
                d = l[6::3]

                self.coeff_settings.dihedral_coeff[dih] = {
                    'style' : style,
                    'm' : m,
                    'K' : np.array([float(i) for i in K]), # kcal/mol
                    'n' : np.array([int(i) for i in n]),
                    'd' : np.array([float(i) for i in d]) # degrees
                }

            elif l[0] == 'improper_coeff':
                imp = int(l[1])
                style = l[2]
                K = float(l[3])
                d = int(l[4])
                n = int(l[5])

                self.coeff_settings.improper_coeff[imp] = {
                    'style' : style,
                    'K' : K, # kcal/mol
                    'd' : d,
                    'n' : n
                }


    def _atomtype_mapping(self, map):
        '''Provide a mapping from integer atom types to alphanumeric atom types'''
        self.type_map = map # TODO: add some checks or functionality here 


    def _guess_elements(self, masses=None):
        '''Guess elements based on masses'''

        if masses is None:
            masses = self.masses

        self.element_map = {}
        for a in masses:
            mass = masses[a]
            self.element_map[a] = {}
            if mass >= 12 and mass <= 12.02:
                self.element_map[a]['name'] = 'C'
                self.element_map[a]['number'] = 6
            elif mass >= 14 and mass <= 14.01:
                self.element_map[a]['name'] = 'N'
                self.element_map[a]['number'] = 7
            elif mass >= 35 and mass <= 35.5:
                self.element_map[a]['name'] = 'Cl'
                self.element_map[a]['number'] = 17
            elif mass >= 15.9 and mass <= 16.1:
                self.element_map[a]['name'] = 'O'
                self.element_map[a]['number'] = 8
            elif mass >= 1 and mass <= 1.008:
                self.element_map[a]['name'] = 'H'
                self.element_map[a]['number'] = 1
            else:
                raise TypeError(f'Element {a} with mass {mass} not implemented... Add it to self._guess_elements')

    def _generate_tpr(self, top, coord, mdp='min.mdp', tpr=None, flags={}, gmx='gmx'):
        '''Use Gromacs gmx grompp to generate a tpr file'''

        if tpr is None:
            tpr = coord.split('.')[0] + '.tpr'
        cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {coord} -o {tpr}']
        
        for f in flags:
            cmd[0] += f' -{f} {flags[f]}'

        self._run(cmd)
        return tpr


    def load_logs(self, log_files):
        '''Load LAMMPS log files into PolymAnalysis class'''

        import lammps_logfile

        if not isinstance(log_files, list):
            log_files = [log_files]

        for log_file in log_files:
            log = lammps_logfile.File(log_file)
            self.logs.append(log)
        
        print('Currently loaded log files:')
        [print(log.logfile.name) for log in self.logs]


    def load_trajectory(self, traj_file):
        '''Load trajectory data into PolymAnalysis class'''

        print('Loading all trajectory coordinates from {} into coords attribute of {}'.format(traj_file, self))
        self.trajectory_file = traj_file

        if self.tpr is None:
            self.universe = mda.Universe(self.filename, traj_file)
        else:
            self.universe = mda.Universe(self.tpr, traj_file)

        self.trajectory = self.universe.trajectory
        self.n_frames = len(self.trajectory)
        self.coords = np.zeros((self.n_frames, self.n_atoms, 3))
        
        t = 0
        for ts in tqdm(self.trajectory):
            self.coords[t,:,:] = self.universe.atoms.positions
            t += 1


    def pack_water_reservoirs(self, output='prehydrate.data', water_file='water.data', box_frac=0.5, seed=12345):
        '''Pack two water reservoirs on either side of the PA slab'''

        from lammps import PyLammps

        # Calculate number of water molecules for each reservoir
        wat_mass = 16 + 1.008*2
        box = self.coords_lims
        s = (box[1,:] - box[0,:])
        s[2] = s[2]*box_frac
        vol = s[0]*s[1]*s[2] / 10**3
        n_waters = int(vol * 6.022 / wat_mass * 10**2) # assuming 1 g/cm^3 water density
        print(f'Adding {n_waters} waters to each reservoir...')

        # Find new box limits
        ub = box[1,2] + s[2]
        lb = box[0,2] - s[2]

        # Use LAMMPS to add water reservoirs to either side and write data file
        L = PyLammps(self.lmp)
        L.file(self.init_file)

        # add topology types if not there already
        if len(self.universe.select_atoms(f'{self.atsel} {self.ow_type}')) == 0:
            L.read_data(self.filename, 'extra/atom/types', 2, 'extra/bond/types', 1, 'extra/angle/types', 1)
            L.mass(int(self.ow_type), 16)
            L.mass(int(self.hw_type), 1.008)
        else:
            L.read_data(self.filename)

        L.file(self.settings_file)

        # convert to fully periodic
        L.change_box('all', 'z', 'final', lb, ub) # add space for reservoirs
        L.change_box('all', 'boundary', 'p p p') # make box fully periodic

        # create regions to fill with waters
        L.region('top block', box[0,0], box[1,0], box[0,1], box[1,1], box[1,2], ub)
        L.region('bott block', box[0,0], box[1,0], box[0,1], box[1,1], lb, box[0,2])
        
        # read molecule template for water (from https://docs.lammps.org/Howto_tip3p.html)
        L.molecule('water', water_file)

        # create water molecules in reservoirs
        L.create_atoms(int(self.ow_type)-1, 'random', n_waters, seed, 'top', 'mol water', seed+1, 'overlap', 1.33)
        L.create_atoms(int(self.ow_type)-1, 'random', n_waters, seed+2, 'bott', 'mol water', seed+3, 'overlap', 1.33)

        # write the new system to data file and reinitialize object
        L.write_data(output)
        self.__init__(output)


    def pack_term_groups(self, output='packed.data', rad=10, n_OH=5, OH_file='OH.data', seed=12345):
        '''Pack free OH groups near unterminated Cl'''

        from lammps import PyLammps

        # Use LAMMPS to add free OH groups
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        Cl_group = self.atoms.select_atoms(f'{self.atsel} {self.cl_type}')
        print(f'Trying to insert {len(Cl_group)*n_OH} OH groups ({len(Cl_group)*n_OH*2} atoms) around {len(Cl_group)} unterminated Cl for a total of {self.n_atoms + len(Cl_group)*n_OH*2} atoms in system...')
        L.molecule('OH', OH_file)

        for i,Cl in enumerate(Cl_group):

            x, y, z = Cl.position

            # create sphere at the Cl with radius rad and place n_OH terminating groups within sphere
            L.region(f'sph{i}', 'sphere', x, y, z, rad, 'side in')

            # create free OH molecules in region
            L.create_atoms(0, 'random', n_OH, seed, f'sph{i}', 'mol OH', seed+1, 'overlap', 1.33)

        # write the new system to data file and reinitialize object
        L.reset_atom_ids()
        L.reset_mol_ids('all')        
        L.write_data(output)
        self.__init__(output)

        print(f'Final number of atoms: {self.n_atoms}')


    def set_artifical_charges(self, atom_group1, atom_group2, charge1=0.5, charge2=-0.5):
        '''Set artificial charges on reacting groups atom_group1 and atom_group2 and 0 charge on all other atoms'''

        if isinstance(atom_group1, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group1)
        else: # else assume input is AtomGroup
            g1 = atom_group1

        if isinstance(atom_group2, str):
            g2 = self.universe.select_atoms(atom_group2)
        else:
            g2 = atom_group2

        others = self.universe.select_atoms('not group g1 and not group g2', g1=g1, g2=g2)

        print('Total charge in system is originally: {:.4f}'.format(self.charges.sum()))

        self.charges[g1.indices] = charge1
        self.charges[g2.indices] = charge2
        self.charges[others.indices] = 0

        print('Total charge in system is now: {:.4f}'.format(self.charges.sum()))


    def write_ndx(self, atom_groups, output='index.ndx', group_names=None):
        '''Write a Gromacs-style index file that associates atom IDs with the corresponding groups (can be used with ndx2group command in LAMMPS to create group)'''

        if not isinstance(atom_groups, list):
            raise TypeError('Input atom_groups as a list of atom_groups')
        
        groups = []
        for atom_group in atom_groups:
            if isinstance(atom_group, str): # if provided selection language, make AtomGroup
                groups.append(self.universe.select_atoms(atom_group))
            else: # else assume input is AtomGroup
                groups.append(atom_group)

        self.groups = groups
        self.ndx = output
        out = open(output, 'w')
        for g, group in enumerate(groups): 
            if group_names is not None: # use names if provided
                name = group_names[g]
            else:
                name = f'Group{g}'

            out.write(f'[ {name} ]\n') # write the group name
            for i, idx in enumerate(group.indices+1): # loop through all indices and go to next line every 12
                if (i+1) % 12 == 0:
                    out.write(f'{idx}\n')
                else:
                    out.write(f'{idx} ')


    def write_GROMACS(self, output='PA', water_itp='HOH.itp'):
        '''Write topology and coordinate information to Gromacs top and gro files'''

        from textwrap import dedent

        self._read_init()
        self._read_settings()
        self._guess_elements()

        #### WRITE TOP FILE ####

        top = open(output + '.top', 'w')

        # [ defaults ]
        if self.init_settings.pair_modify[0] == 'mix' and self.init_settings.pair_modify[1] == 'arithmetic':
            comb_rule = 2
        else:
            raise TypeError(f'pair_modify {self.init_settings.pair_modify} has not been implemented yet')
    
        if self.init_settings.special_bonds[0] == 'amber':
            fudgeLJ = 0.5
            fudgeQQ = 0.8333
        else:
            print('No fudgeLJ and fudgeQQ specified. Setting them to 1.')
            fudgeLJ = 1
            fudgeQQ = 1

        defaults = dedent(f"""\
        [ defaults ]
        ; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
        \t1\t{comb_rule}\tyes\t{fudgeLJ}\t{fudgeQQ}\n
        """)
        top.write(defaults)

        # [ atomtypes ]
        atomtypes = dedent(f"""\
        [ atomtypes ]
        ;type, bondingtype, atomic_number, mass, charge, ptype, sigma, epsilon
        """)
        for a in self.masses:
            atype = self.type_map[a]
            at_num = self.element_map[a]['number']
            mass = self.masses[a]
            sig = self.coeff_settings.pair_coeff[a]['sigma'] / 10 # convert to nm
            eps = self.coeff_settings.pair_coeff[a]['epsilon'] * 4.184 # convert to kJ/mol

            line = f'{atype:4s} {atype:4s}\t{at_num:.0f}\t{mass:.8f}\t{0:.8f}\tA\t{sig:.8e}\t{eps:.8e}\n'
            atomtypes += line

        top.write(atomtypes)

        # [ moleculetype ] for each molecule in the system
        print('Writing topology information for polymer chains...')
        PA_chains = self.universe.select_atoms('not type 11 12')
        n_chains = len(PA_chains.residues)
        count_atoms = 0
        count_molecules = 0
        sorted_atoms = []
        resnames = np.empty(len(self.universe.residues), dtype=object)
        atomnames = np.empty(self.n_atoms, dtype=object)
        for m in tqdm(range(n_chains)):
            mol = self.universe.select_atoms('resid {}'.format(PA_chains.residues[m].resid))
            resnames[count_molecules] = f'PA{m+1}'
            count_molecules += 1

            molecule = dedent(f'''
            [ moleculetype ]
            PA{m+1}\t\t3

            [ atoms ]
            ;num, type, resnum, resname, atomname, cgnr, q, m
            ''')

            # [ atoms ] for each molecule
            types_per_res = {}
            atom_in_molecule = {}
            for a,atom in enumerate(mol.atoms):
                sorted_atoms.append(atom)
                atype = self.type_map[int(atom.type)]
                resnum = m+1
                resname = f'PA{m+1}'
                element = self.element_map[int(atom.type)]['name']
                charge = atom.charge
                mass = atom.mass
                atom_in_molecule[atom.id] = a+1

                # NOTE: TOO MANY ATOMS --> DO NOT NUMBER THEM WITHIN EACH MOLECULE
                # if element in types_per_res:
                #     types_per_res[element] += 1
                #     atomname = f'{element}{types_per_res[element]}'
                # else:
                #     types_per_res[element] = 0
                #     atomname = f'{element}'
                atomname = element

                atomnames[count_atoms] = atomname
                count_atoms += 1
                line = f'{a+1:7d} {atype:4s}\t\t{resnum} {resname:8s} {atomname:9s}\t0\t{charge:.8f}\t{mass:.8f}\n'
                molecule += line

            # [ bonds ] for each molecule
            molecule += dedent(f'''
            [ bonds ]
            ;   ai     aj funct  r               k
            ''')

            for bond in mol.bonds:
                a1 = atom_in_molecule[bond.atoms[0].id]
                a2 = atom_in_molecule[bond.atoms[1].id]
                btype = int(bond.type)
                style = self.coeff_settings.bond_coeff[btype]['style']

                if style == 'harmonic':
                    func = 1
                    K = self.coeff_settings.bond_coeff[btype]['K'] * 2 * 4.184 * 10**2 # convert to kJ/mol/nm^2
                    r = self.coeff_settings.bond_coeff[btype]['r0'] / 10 # convert to nm
                else:
                    raise NotImplementedError(f'Bond style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {func}\t{r:.8e}\t{K:.8e}\n'
                molecule += line


            # [ angles ] for each molecule
            molecule += dedent(f'''
            [ angles ]
            ;   ai     aj     ak     funct  theta    cth
            ''')

            for ang in mol.angles:
                a1 = atom_in_molecule[ang.atoms[0].id]
                a2 = atom_in_molecule[ang.atoms[1].id]
                a3 = atom_in_molecule[ang.atoms[2].id]
                angtype = int(ang.type)
                style = self.coeff_settings.angle_coeff[angtype]['style']

                if style == 'harmonic':
                    func = 1
                    K = self.coeff_settings.angle_coeff[angtype]['K'] * 2 * 4.184
                    theta = self.coeff_settings.angle_coeff[angtype]['theta0']
                else:
                    raise NotImplementedError(f'Angle style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {func}\t{theta:.8e}\t{K:.8e}\n'
                molecule += line
            
            # [ dihedrals ] for each molecule
            molecule += dedent(f'''
            [ dihedrals ]
            ;    i      j      k      l   func
            ''')

            for dih in mol.dihedrals: # add proper dihedrals first

                a1 = atom_in_molecule[dih.atoms[0].id]
                a2 = atom_in_molecule[dih.atoms[1].id]
                a3 = atom_in_molecule[dih.atoms[2].id]
                a4 = atom_in_molecule[dih.atoms[3].id]
                dihtype = int(dih.type)
                style = self.coeff_settings.dihedral_coeff[dihtype]['style']

                if style == 'fourier' and self.coeff_settings.dihedral_coeff[dihtype]['m'] == 1: # if only one fourier term, this is a periodic proper dihedral
                    func = 1
                    K = self.coeff_settings.dihedral_coeff[dihtype]['K'][0] * 4.184
                    n = self.coeff_settings.dihedral_coeff[dihtype]['n'][0]
                    d = self.coeff_settings.dihedral_coeff[dihtype]['d'][0]
                    params = f'{d:.8e}\t{K:.8e}\t{n}'

                elif style == 'fourier': # otherwise it is a Fourier dihedral
                    func = 5
                    K = self.coeff_settings.dihedral_coeff[dihtype]['K'] * 4.184
                    n = self.coeff_settings.dihedral_coeff[dihtype]['n']
                    d = self.coeff_settings.dihedral_coeff[dihtype]['d']
                    C = np.zeros(4)

                    for Ki, ni, di in zip(K,n,d):
                        if ni == 1 and di == 0:
                            C[0] = 2*Ki
                        elif ni == 2 and di == 180:
                            C[1] = 2*Ki
                        elif ni == 3 and di == 0:
                            C[2] = 2*Ki
                        elif ni == 4 and di == 180:
                            C[3] = 2*Ki
                        else:
                            raise TypeError(f'Parameters {[Ki, ni, di]} cannot be converted to Gromacs dihedral style {style}')

                    params = f'{C[0]:.8e}\t{C[1]:.8e}\t{C[2]:.8e}\t{C[3]:.8e}'

                else:
                    raise NotImplementedError(f'Dihedral style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {a4:>7d} {func}\t{params}\n'
                molecule += line

            for imp in mol.impropers: # add improper dihedrals

                a1 = atom_in_molecule[imp.atoms[0].id]
                a2 = atom_in_molecule[imp.atoms[1].id]
                a3 = atom_in_molecule[imp.atoms[2].id]
                a4 = atom_in_molecule[imp.atoms[3].id]
                imptype = int(imp.type)
                style = self.coeff_settings.improper_coeff[imptype]['style']

                if style == 'cvff':
                    func = 4
                    K = self.coeff_settings.improper_coeff[imptype]['K'] * 4.184
                    n = self.coeff_settings.improper_coeff[imptype]['n']
                    d = self.coeff_settings.improper_coeff[imptype]['d']
                else:
                    raise NotImplementedError(f'Improper style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {a4:>7d} {func}\t{d:.8e}\t{K:.8e}\t{n}\n'
                molecule += line

            top.write(molecule)

        # include water itp if waters present
        OW = self.universe.select_atoms('type {}'.format(self.ow_type))
        n_waters = len(OW)
        if n_waters > 0:
            water = f'\n#include "{water_itp}"'
            top.write(water)

        # [ system ] and [ molecules ]
        system = dedent(f'''
        [ system ]
        {self.filename} converted

        [ molecules ]
        ; Compound        nmols
        ''')

        for n in range(count_molecules):
            line = f'PA{n+1}\t1\n'
            system += line

        if n_waters > 0:
            system += f'SOL\t{n_waters}\n'

        top.write(system)
        top.close()
        print(f'Finished writing Gromacs topology file: {output}.top!')

        #### WRITE GRO FILE ####
        print(f'Writing coordinate information...')

        for ow in OW: # now add waters
            wat = ow.residue
            hw1 = wat.atoms[1]
            hw2 = wat.atoms[2]

            if hw1.type != self.hw_type:
                raise TypeError(f'Atom {hw1} not a water hydrogen')
            elif hw2.type != self.hw_type:
                raise TypeError(f'Atom {hw2} not a water hydrogen')
            
            sorted_atoms.append(ow)
            sorted_atoms.append(hw1)
            sorted_atoms.append(hw2)
            
            resname = 'SOL'
            
            resnames[count_molecules] = resname
            atomnames[count_atoms] = 'OW'
            count_atoms += 1

            atomnames[count_atoms] = 'HW1'
            count_atoms += 1

            atomnames[count_atoms] = 'HW2'
            count_atoms += 1
            count_molecules += 1

        sorted_ag = mda.AtomGroup(sorted_atoms)
        self.universe.add_TopologyAttr('resnames')
        self.universe.add_TopologyAttr('names')
        sorted_ag.residues.resnames = resnames
        sorted_ag.atoms.names = atomnames
        sorted_ag.write(output + '.gro')

        print(f'Finished writing Gromacs coordinate file: {output}.gro!\n')


    def remove_atoms(self, atom_group, output='removed_atoms.data'): 
        '''Remove atoms in atom_group from the system'''
        
        from lammps import PyLammps

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            g1 = atom_group

        # create a string with atom ids for group definition (not very clean...)
        ids = str(g1.indices[0] + 1)
        for idx in (g1.indices + 1)[1:]:
            ids += ' ' + str(idx)

        print(f'Deleting {len(g1)} atoms...')

        # Use LAMMPS to remove atoms and reset ids
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        L.group('del', 'id', ids)
        L.delete_atoms('group', 'del')
        L.reset_atom_ids()
        L.reset_mol_ids('all')

        # write the new system to data file and reinitialize object
        L.write_data(output)
        print()
        self.__init__(output)


    def insert_cations_in_membrane(self, ion_name='Na', ion_charge=1, extra_inserted=0, tol=2, output='PA_ions.gro'):
        '''Add ions to the membrane by merging universe with ion universe'''

        from MDAnalysis.analysis import distances

        # locate COOH/COO- oxygens
        c_group = self.universe.select_atoms('type c and not bonded type n')
        deprot_o = []
        prot_o = []
        for c in c_group:
            my_Os = [atom for atom in c.bonded_atoms if atom.type == 'o']
            if len(my_Os) == 2:
                deprot_o.append(my_Os[0]) # add one of the two O's to the AtomGroup
            elif len(my_Os) == 1:
                prot_o.append(my_Os[0]) # add the =O to the AtomGroup
            else:
                raise TypeError(f'{c} is not the carbon in R-COOH or R-COO-')
        
        # calculate excess charge from adding ions
        polymer_charge = len(deprot_o)*-1
        if len(deprot_o) % ion_charge > 0: # if we cannot exactly balance polymer charge with cations
            n_ions = round(len(deprot_o) / ion_charge) + 1
        else:
            n_ions = len(deprot_o)

        print(f'Polymer charge is {polymer_charge}')

        n_ions += extra_inserted # add the extra inserted ions to total added ions count
        counterion_charge = n_ions*ion_charge # calculate charge from added ions
        excess_charge = counterion_charge + polymer_charge # calculate excess charge from polymer and added ions

        print(f'Adding {n_ions} cations, which results in {excess_charge} excess charge in the system')

        # select which O's to place ions near
        print('WARNING: currently the only insertion algorithm is R-COOH groups with at least 2 water within 5 Angstroms')

        prot_idx = []
        for i,O in enumerate(prot_o):
            my_waters = self.universe.select_atoms(f'(type OW) and (sphzone 5 index {O.index})')
            if len(my_waters) > 1:
                prot_idx.append(i)

        if len(prot_idx) < extra_inserted:
            raise TypeError('Not enough R-COOH groups with waters')

        tmp = np.array(prot_o)
        o_group = mda.AtomGroup(deprot_o + tmp[prot_idx].tolist()) # select all deprotonated groups and extra_inserted random protonated groups

        # delete waters and find where to place ions
        ion_pos = []
        for O in o_group:
            my_id = O.index
            my_waters = self.universe.select_atoms(f'resname SOL and sphzone 5 index {my_id}').residues # select waters within 5 Angstroms
            if len(my_waters) > 0:
                pos = my_waters.atoms.center_of_mass() # place ion at the COM of the waters in the zone
                
                # randomly kick the ion around until it is at least 2 Angstroms away from all other atoms
                i = 0
                too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}') - my_waters.atoms
                n_close = len(too_close)
                while n_close > 0:
                    kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
                    kick_vec = kick_vec / np.linalg.norm(kick_vec) 
                    kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms

                    pos += kick_strength*kick_vec
                    too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}') - my_waters.atoms

                    if len(ion_pos) == 0: # if this is the first ion to be added, other ions do not matter
                        n_close = len(too_close)
                    else: # otherwise, make sure we do not place an ion on top of a previously inserted ion
                        dists = distances.distance_array(pos, np.array(ion_pos), box=self.universe.dimensions)
                        n_close = len(too_close) + (dists < 2).sum()
                    
                    i += 1
                    if i == 10_000:
                        print(f'failed on {O} with {n_close} atoms too close')
                        [print(f'\t{a}') for a in too_close]
                        exit()

                
                # print(f'Distance between O and ion: {distances.distance_array(pos,O.position, box=self.universe.dimensions)[0,0]:.4f}')
                print(f'Inserted ion near {O} after {i} kicks at position {pos}')
                ion_pos.append(pos) 
                self.atoms = self.atoms.subtract(my_waters.atoms) # remove the waters to be replaced

        # reassign residue numbers after deleted waters
        dims = self.universe.dimensions # save original dimensions
        n_residues = len(self.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe = mda.Merge(self.atoms)
        self.universe.add_TopologyAttr('resid', list(resids))

        # create an empty universe for the newly placed ions
        n_residues = len(self.atoms.residues)
        ion_u = mda.Universe.empty(n_ions,
                                   n_residues=n_ions,
                                   atom_resindex=list(range(n_ions)),
                                   trajectory=True)
        
        ion_u.add_TopologyAttr('name', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('type', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('resname', [f'{ion_name.upper()}']*n_ions)
        ion_u.add_TopologyAttr('resid', list(np.arange(n_residues+1, n_residues+n_ions+1)))

        ion_u.atoms.positions = np.array(ion_pos[:n_ions])

        # merge with system universe, write to gro file
        new_universe = mda.Merge(self.universe.atoms, ion_u.atoms)
        new_universe.dimensions = dims
        new_universe.atoms.write(output)

        return n_ions, output, excess_charge


    def insert_ions(self, n_anions=0, anion_name=None, anion_charge=-1, n_cations=0, cation_name=None, cation_charge=1, top='PA_ions.top', output='PA_ions.gro', water_sel=18):
        '''Insert ions with gmx genion'''

        if cation_name is None and anion_name is None:
            raise TypeError('No ions given. Please provide an anion, cation, or both.')
        elif cation_name is None:
            cmd = f'echo {water_sel} | {self.gmx} genion -s {self.tpr} -p {top} -o {output} -nn {n_anions} -nname {anion_name} -nq {anion_charge}'
            self._run(cmd)
        elif anion_name is None:
            cmd = f'echo {water_sel} | {self.gmx} genion -s {self.tpr} -p {top} -o {output} -np {n_cations} -pname {cation_name} -pq {cation_charge}'
            self._run(cmd)
        else:
            cmd = f'echo {water_sel} | {self.gmx} genion -s {self.tpr} -p {top} -o {output} -nn {n_anions} -nname {anion_name} -nq {anion_charge} -np {n_cations} -pname {cation_name} -pq {cation_charge}'
            self._run(cmd)

        # reinitialize the class with updated ions
        self._generate_tpr(top, output, tpr=self.tpr, gmx=self.gmx, flags={'maxwarn' : 1})
        self.__init__(output, frmt='GRO', tpr_file=self.tpr, 
                      xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                      cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type)


    def calculate_density(self, atom_group='all', box=False):
        '''Calculate density of a selection of the polymer membrane'''

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            g = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            g = atom_group

        if box:
            xlo, xhi = self.box[:,0]
            ylo, yhi = self.box[:,1]
            zlo, zhi = self.box[:,2]
        else:
            xlo, xhi = g.positions[:,0].min(), g.positions[:,0].max() # NOTE: these limits will be slightly different than self.box
            ylo, yhi = g.positions[:,1].min(), g.positions[:,1].max() #       since it is using the min and max atom coordinates, but
            zlo, zhi = g.positions[:,2].min(), g.positions[:,2].max() #       the difference should only be at most 0.002

        total_mass = g.masses.sum() / 6.022 / 10**23 # [g/mol * mol/# = g]
        s1 = (xhi-xlo) * 10**-8 # [Ang * 10^8 cm/Ang = cm]
        s2 = (yhi-ylo) * 10**-8 # cm
        s3 = (zhi-zlo) * 10**-8 # cm
        vol = s1*s2*s3 # cm^3
        density = total_mass / vol

        if box:
            print('Density of PA membrane: {:.4f} g/cm^3'.format(density))
        else:
            print('Density of {}: {:.4f} g/cm^3'.format(atom_group, density) )

        return density
    

    def atom_distribution(self, percentage=False):
        '''Plot distribution of atoms over the resids'''

        X = []
        Y = []

        for res in self.universe.residues:
            X.append(res.resid)
            Y.append(len(res.atoms))

        X = np.array(X)
        Y = np.array(Y)
        print(f'Resid {Y.argmax()+1} is the largest chain with {Y.max()} atoms')
        if percentage:
            Y = Y / self.n_atoms * 100

        self.atom_distribution = (X, Y)
        print('Atom distribution saved in self.atom_distribution')

        plt.plot(X,Y)
        plt.xlabel('resid')
        plt.ylabel('number of atoms')
        if percentage:
            plt.ylabel('percentage of atoms')
        plt.savefig('atom_distribution.png')
        plt.show()
        plt.close()


    def plot_crosslinked(self):
        '''Plot crosslinking percentage as a function of simulation time'''

        if len(self.logs) == 0:
            raise AttributeError('No log files loaded... Try load_log() first')

        self.possible_bonds = self.n_monomers[0]*2
        print('Using ({} MPD molecules)*(2 possible bonds) = {} total possible bonds for crosslinking percentage calculation'.format(self.n_monomers[0], self.possible_bonds))

        X = []
        Y = []
        old_bonds = 0
        old_steps = 0
        for current_log in self.logs:
        
            for l, log in enumerate(current_log.partial_logs): 

                if 'f_fxrct[1]' in log.keys():
                    steps = log.get('Step')
                    bonds = log.get('f_fxrct[1]')

                    for x in steps:
                        X.append((old_steps + x)*0.001) # multiply each step by time (0.001 ps per step)

                    for y in bonds:
                        Y.append(old_bonds + y)

                    old_bonds += bonds[-1]
                    old_steps += steps[-1]

        self.bonds_formed = np.array(Y)
        self.percent_crosslinked = np.array(Y) / self.possible_bonds * 100

        X = np.array(X)
        Y = np.array(Y)

        fig, ax = plt.subplots(1,1, figsize=(10,10))

        print('Final percent crosslinked:', Y[-1] / self.possible_bonds * 100)
        plt.plot(X, Y / self.possible_bonds * 100)
        plt.xlabel('Time (ps)', fontsize=16)
        plt.ylabel('Percent crosslinked', fontsize=16)
        plt.ylim(0,100)
        plt.xlim(-500,)
        plt.xticks(fontsize=16, rotation=45)
        plt.yticks(fontsize=16)

        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(250))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.xaxis.set_major_formatter('{x:,.0f}')

        plt.savefig('percent_crosslinked.png')
        plt.show()
        plt.close()


    def plot_ts(self, log, y, timestep=0.001, start_log=0, end_log=None, show_legend=False, plot_end=False):
        '''Plot a timeseries of a given thermodynamic quantity'''

        logfile = log

        if end_log is None:
            plogs = logfile.partial_logs[start_log:]
        else:
            plogs = logfile.partial_logs[start_log:end_log+1]

        fig, ax = plt.subplots(1,1)

        n = 0
        for plog in plogs:

            time = plog.get('Step')*timestep

            if type(y) is list:
                
                for i in y:

                    if not i in plog.keys():
                        raise TypeError('{} not available in partial log {}. Try one of {}'.format(i, n, plog.keys()))
            
                    Y = plog.get(i)
                    plt.plot(time, Y, label=i + ': partial log {}'.format(n))
            
            else:

                if not y in plog.keys():
                    raise TypeError('{} not available in partial log {}. Try one of {}'.format(y, n, plog.keys()))
            
                Y = plog.get(y)
                plt.plot(time, Y, label=y + ': partial log {}'.format(n))


            n += 1

        if show_legend:
            plt.legend()

        if plot_end:
            plt.xlim(time[-100], time[-1])
            plt.ylim(Y[-1]*0.9, Y[-1]*1.1)

        plt.xlabel('Time (ps)')
        plt.ylabel(y)
        plt.savefig('{}_vs_time.png'.format(y))
        plt.show()
        plt.close()


    def density_profile(self, atom_group, bin_width=0.5, dim='z', method='atom', frameby=1):
        '''Calculate the partial density across the box'''

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            ag = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            ag = atom_group

        print(f'\nCalculating the partial density profile of {atom_group} in the {dim} dimension...')

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)

        counts = np.zeros(n_bins-1)
        
        if self.trajectory is None:
            for b in tqdm(range(n_bins-1)):
                lb = bins[b]
                ub = bins[b+1]
                bin_atoms = self.universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group ag', ag=ag)
                if method in ['atom', 'atoms', 'all']:
                    counts[b] += len(bin_atoms)
                elif method in ['molecule', 'mol', 'residue', 'res']: 
                    counts[b] += bin_atoms.n_residues
                elif method in ['mass', 'mass density']:
                    box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                    dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                    mass = bin_atoms.masses.sum() / 6.022 / 10**23
                    counts[b] += mass / dV
        else:
            for ts in tqdm(self.trajectory[::frameby]):
                self.box = self._get_box()
                for b in range(n_bins-1):
                    lb = bins[b]
                    ub = bins[b+1]
                    bin_atoms = self.universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group ag', ag=ag)

                    if method in ['atom', 'atoms', 'all', 'number']:
                        counts[b] += len(bin_atoms)
                    elif method in ['molecule', 'mol', 'residue', 'res']: 
                        counts[b] += bin_atoms.n_residues
                    elif method in ['mass', 'mass density']:
                        box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                        dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                        mass = bin_atoms.masses.sum() / 6.022 / 10**23
                        counts[b] += mass / dV

            counts = counts / len(self.trajectory[::frameby])

        return bins, counts

    def crosslinked_profile(self, bin_width=5, dim='z'):
        '''Calculate the partial crosslinking density across the box'''

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)

        self.n_xlinks = np.zeros(n_bins-1)
        for b in tqdm(range(n_bins-1)):
            lb = bins[b]
            ub = bins[b+1]

            self.n_xlinks[b] = len(self.universe.select_atoms('prop {} > {} and prop {} < {} and {} {} and bonded {} {}'.format(dim, lb, dim, ub, self.atsel, self.xlink_c_type, self.atsel, self.xlink_n_type)))

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(bins[:-1], self.n_xlinks)
        plt.xlabel('{} coordinate'.format(dim))
        plt.ylabel('number of crosslinks')
        plt.savefig('crosslinked_profile.png')
        plt.show()
        plt.close()


    def terminated_profile(self, bin_width=5, dim='z', count_type='Cl'):
        '''Calculated the (un)terminated density across the box'''

        if count_type in ['Cl', 'CL', 'cl']:
            print('Counting remaining Cl in system...')
            term_sel = 'type {}'.format(self.cl_type)
            term_label = 'remaining Cl'
        elif count_type in ['O', 'OH', 'o', 'oh']:
            print('Counting hydroxyl groups in system...')
            term_sel = 'type {}'.format(self.oh_type)
            term_label = 'OH groups'
        elif count_type in ['bonded O', 'bonded OH', 'bonded o', 'bonded oh', 'term C', 'terminated C']:
            print('Counting bonded hydroxyl groups in system...')
            term_sel = 'type {} and bonded type {}'.format(self.oh_type, self.xlink_c_type)
            term_label = 'bonded OH groups'
        elif count_type in ['free O', 'free OH', 'free o', 'free oh', 'nonbonded OH']:
            print('Counting free hydroxyl groups in system...')
            term_sel = 'type {} and not bonded type {}'.format(self.oh_type, self.xlink_c_type)
            term_label = 'free OH groups'
        else:
            raise TypeError('count_type = {} not implemented. Try Cl, OH, free OH, or bonded OH'.format(count_type))

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)
        self.term_bins = bins

        self.n_term = np.zeros(n_bins-1)
        for b in tqdm(range(n_bins-1)):
            lb = bins[b]
            ub = bins[b+1]

            self.n_term[b] = len(self.universe.select_atoms('prop {} > {} and prop {} < {} and {}'.format(dim, lb, dim, ub, term_sel)))

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(bins[:-1], self.n_term)
        plt.xlabel('{} coordinate'.format(dim))
        plt.ylabel('number of {}'.format(term_label))
        plt.savefig('terminated_profile_{}.png'.format(term_label))
        plt.show()
        plt.close()


    def rdf(self, atom_group1, atom_group2, range=(0,15), nbins=75, output='rdf.png', **kwargs):
        '''Calculate the RDF from atom_group1 to atom_group2 using MDAnalysis InterRDF'''

        from MDAnalysis.analysis.rdf import InterRDF

        print(f'Calculating the RDF for {atom_group1} and {atom_group2} over {self.universe.trajectory} frames')

        if isinstance(atom_group1, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group1)
        else: # else assume input is AtomGroup
            g1 = atom_group1

        if isinstance(atom_group2, str):
            g2 = self.universe.select_atoms(atom_group2)
        else:
            g2 = atom_group2

        rdf = InterRDF(g1, g2, range=range, nbins=nbins, verbose=True)
        rdf.run(**kwargs)
        self.rdf_results = rdf.results

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(rdf.results.bins, rdf.results.rdf)
        plt.xlabel('r (A)')
        plt.ylabel('g(r)')
        plt.savefig(output)
        plt.show()
        plt.close()


    def plot_membrane(self, PA, water=None, lb=None, ub=None, avg_mem=True, frame=-1, savename=None):
        '''Plot a scatter plot of the membrane'''

        xyz = self.coords

        fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True)
        
        if avg_mem: # Plot average membrane coordinates
            ax[0].scatter(xyz[:,PA,0].mean(axis=0), xyz[:,PA,2].mean(axis=0), c='gray', alpha=0.05)
            ax[1].scatter(xyz[:,PA,1].mean(axis=0), xyz[:,PA,2].mean(axis=0), c='gray', alpha=0.05, label='average membrane coordinates')
        else: # Plot one frame of membrane coordinates
            ax[0].scatter(xyz[frame,PA,0], xyz[frame,PA,2], c='gray', alpha=0.05)
            ax[1].scatter(xyz[frame,PA,1], xyz[frame,PA,2], c='gray', alpha=0.05)

        if water is not None:
            # Plot one frame of water coordinates
            ax[0].scatter(xyz[frame,water,0], xyz[frame,water,2], c='skyblue', alpha=0.01)
            ax[1].scatter(xyz[frame,water,1], xyz[frame,water,2], c='skyblue', alpha=0.01, label='frame {} water coordinates'.format(frame))

        # Plot bulk membrane cutoffs
        if lb is not None:
            ax[0].axhline(lb, c='r', ls='dashed')
            ax[1].axhline(lb, c='r', ls='dashed', label='bulk boundaries')
        if ub is not None:
            ax[0].axhline(ub, c='r', ls='dashed')
            ax[1].axhline(ub, c='r', ls='dashed')

        # formatting
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('z')
        ax[1].set_xlabel('y')
        
        plt.show()
        if savename is not None:
            fig.savefig(savename)

        plt.close()


    def write_data(self, output, frame=-1):
        '''Write new LAMMPS data file with (possibly) new coordinates and charges'''

        # Open output file and get original data file
        out = open(output, 'w')
        lines = self.original
        box = self.box

        # Write header
        header = []
        header.append(lines.pop(0))
        header.append('\n')

        top_types = ['atom', 'bond', 'angle', 'dihedral', 'improper']
        for top_type in top_types:
            line = '{} {}s\n'.format(getattr(self, 'n_{}s'.format(top_type)), top_type)
            header.append(line)

            line = '{} {} types\n'.format(getattr(self, 'n_{}_types'.format(top_type)), top_type)
            header.append(line)
        
        print('WARNING: PolymAnalysis.write_data() only writes cubic boxes')

        header.append("""
        {:.18g} {:.18g} xlo xhi
        {:.18g} {:.18g} ylo yhi
        {:.18g} {:.18g} zlo zhi
        """.format(box[0,0], box[1,0], box[0,1], box[1,1], box[0,2], box[1,2]))

        header.append('\nMasses\n\n')
        for at in self.masses:
            line = '{} {}\n'.format(at, self.masses[at])
            header.append(line)

        print('WARNING: PolymAnalysis.write_data() does not include any parameters (e.g. Pair Coefs, Bond Coefs)')

        # Write the atom coordinates
        print('Writing atom coordinates and charges...')
        header.append('\nAtoms # full\n\n')
        xyz = self.coords[frame,:,:]
        charges = self.charges

        # Get index of where Atoms section begins
        if 'Atoms\n' in lines:
            idx = lines.index('Atoms\n') + 2
        elif 'Atoms # full\n' in lines:
            idx = lines.index('Atoms # full\n') + 2
        else:
            raise SyntaxError('Could not locate Atoms section in {}'.format(self.filename))
        
        for n in tqdm(range(self.n_atoms)):
            atom = lines[idx+n].split()
            a_id = int(atom[0])
            mol_id = int(atom[1])
            atype = int(atom[2])
            charge = charges[a_id-1]
            x, y, z = xyz[a_id-1,:]

            line = '{:.0f} {:.0f} {:.0f} {:.8g} {:.18g} {:.18g} {:.18g}\n'.format(a_id, mol_id, atype, charge, x, y, z)
            header.append(line)
        
        # Write all other information unchanged
        print('WARNING: PolymAnalysis.write_data() does not write new topology information, but does remove topology information for removed atoms')
        print('Writing remaining velocity and topology sections...')
        end_idx = idx + n + 1
        for line in tqdm(lines[end_idx:]):
            header.append(line)
        
        out.writelines(header)


# class to save setting information
class Settings:
    def __init__(self):
        pass


if __name__ == '__main__':

    data = PolymAnalysis('prehydrate.data')

    mapping = {
        1 : 'c',
        2 : 'ca',
        3 : 'ha',
        4 : 'hn',
        5 : 'ho',
        6 : 'cl',
        7 : 'n',
        8 : 'nh',
        9 : 'o',
        10 : 'oh',
        11 : 'OW',
        12 : 'HW'
    }
    
    data._atomtype_mapping(mapping)
    data.write_GROMACS(output='tmp')