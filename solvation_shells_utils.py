# Hold functions and classes to analyze solvation shells

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import pymbar

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.base import Results

from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

import subprocess
from textwrap import dedent
from glob import glob
from tqdm import tqdm
import time

from linear_algebra import *

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run grompp with mdp file on gro file with topology top
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if tpr is None:
        tpr = gro.split('.gro')[0] + '.tpr'
    cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {gro} -o {tpr}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return tpr


def mdrun(tpr, output=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run GROMACS with tpr file
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if output is None:
        output = tpr.split('.tpr')[0]
    cmd = [f'{gmx} mdrun -s {tpr} -deffnm {output}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return output + '.gro'


def write_packmol(C, cation, anion, cation_charge=1, anion_charge=-1, n_waters=1000, water='water.pdb', filename='solution.inp', packmol_options=None):
    '''
    Write packmol input file for standard MD simulation
    
    Parameters
    ----------
    C : float
        Concentration of the solution
    cation : str
        Filename of the cation in solution
    anion : str
        Filename of the anion in solution
    cation_charge : int
        Charge on the cation, default=+1
    anion_charge : int
        Charge on the anion, default=-1
    n_waters : int
        Number of water molecules to pack in system, default=1000
    water : str
        Filename of the water molecule in solution, default='water.pdb'
    filename : str
        Filename for the packmol input file
    packmol_options : dict
        Additional options to put in the packmol input file, default=None uses some presets.
        If specified, should have 'seed', 'tolerance', 'filetype', 'output', 'box'.

    Returns
    -------
    filename : str
        Filename of the packmol input file

    '''

    if packmol_options is None:
        packmol_options = {
            'seed' : 123456,
            'tolerance' : 2.0,
            'filetype' : 'pdb',
            'output' : filename.split('.inp')[0],
            'box' : 32
        }

    # (n_cations / salt molecule) * (N_A salt molecules / mol salt) * (C mol salt / L water) * (L water / rho g water) * (18.02 g / mol) * (mol / N_A molecules) * (n_water molecules)
    n_cations = (-cation_charge/anion_charge) * C / 997 * 18.02 * n_waters
    n_anions = (-anion_charge/cation_charge) * C / 997 * 18.02 * n_waters

    print(f'For a box of {n_waters} waters, would need {n_cations} cations and {n_anions} anions.')

    n_cations = round(n_cations)
    n_anions = round(n_anions)
    print(f'So, adding {n_cations} cations and {n_anions} anions...')

    f = dedent(f'''\
    #
    # A mixture of water and salt
    #

    # All the atoms from diferent molecules will be separated at least {packmol_options['tolerance']}
    # Anstroms at the solution.

    seed {packmol_options['seed']}
    tolerance {packmol_options['tolerance']}
    filetype {packmol_options['filetype']}

    # The name of the output file

    output {packmol_options['output']}

    # {n_waters} water molecules and {n_cations} cations, {n_anions} anions will be put in a box
    # defined by the minimum coordinates x, y and z = 0. 0. 0. and maximum
    # coordinates {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. That is, they will be put in a cube of side
    # {packmol_options['box']} Angstroms. (the keyword "inside cube 0. 0. 0. {packmol_options['box']}.") could be used as well.

    structure {water}
    number {n_waters}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. 
    end structure

    structure {cation}
    number {n_cations}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure

    structure {anion}
    number {n_anions}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure


    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_plumed_metad(options, filename='plumed.dat'):
    '''Write plumed.dat file for metadynamics simulation'''

    f = dedent(f'''\
    water_group: GROUP ATOMS=1-3000:3   # oxygen atom of the water molecules
    n: COORDINATION GROUPA=3001 GROUPB=water_group SWITCH={{Q REF={options['R_0']} BETA={options['a']} LAMBDA=1 R_0={options['R_0']}}}

    t: MATHEVAL ARG=n FUNC=1000-x PERIODIC=NO

    PRINT STRIDE=10 ARG=* FILE=COLVAR
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_plumed_umbrella(options, filename='plumed.dat'):
    '''Write plumed input file for umbrella sampling simulation'''

    f = dedent(f'''\
    water_group: GROUP ATOMS=1-{options['N_WATERS']*3}:3   # oxygen atom of the water molecules
    n: COORDINATION GROUPA={options['N_WATERS']*3+1} GROUPB=water_group SWITCH={{Q REF={options['R_0']} BETA=-21.497624558253246 LAMBDA=1 R_0={options['R_0']}}}
    t: MATHEVAL ARG=n FUNC={options['N_WATERS']}-x PERIODIC=NO

    r: RESTRAINT ARG=t KAPPA={options['KAPPA']} AT={options['AT']} # apply a harmonic restraint at CN=AT with force constant = KAPPA kJ/mol

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_plumed_decoordination(options, filename='plumed.dat'):
    '''Write plumed input file for umbrella sampling simulation biased in total coordination'''

    f = dedent(f'''\
    ion: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['ion_group']}
    not_ion: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['not_ion_group']}
    n: COORDINATION GROUPA=ion GROUPB=not_ion SWITCH={{Q REF={options['R_0']} BETA=-{options['a']} LAMBDA=1 R_0={options['R_0']}}}
    t: MATHEVAL ARG=n FUNC={options['n_group']}-x PERIODIC=NO

    r: RESTRAINT ARG=t KAPPA={options['KAPPA']} AT={options['AT']} # apply a harmonic restraint at CN=AT with force constant = KAPPA kJ/mol

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_sbatch_umbrella(options, filename='submit.job'):
    '''Write SLURM submission script to run an individual umbrella simulation'''

    f = dedent(f'''\
    #!/bin/bash
    #SBATCH -A chm230020p
    #SBATCH -N 1
    #SBATCH --ntasks-per-node={options['ntasks']}
    #SBATCH -t {options['time']}
    #SBATCH -p RM-shared
    #SBATCH -J '{options['job']}_{options['sim_num']}'
    #SBATCH -o '%x.out'
    #SBATCH --mail-type=END
    #SBATCH --mail-user=nasc4134@colorado.edu

    module load gcc
    module load openmpi/4.0.2-gcc8.3.1

    source /jet/home/schwinns/.bashrc
    source /jet/home/schwinns/pkgs/gromacs-plumed/bin/GMXRC

    # run umbrella sampling simulations and analysis
    python run_umbrella_sim.py -N {options['sim_num']} -g {options['gro']} -m {options['mdp']} -p {options['top']} -n {options['ntasks']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def run_plumed(plumed_input, traj, dt=0.002, stride=250, output='COLVAR'):
    '''Run plumed driver on plumed input file for a given trajectory (as an xtc) and read output COLVAR'''
    cmd = f'plumed driver --plumed {plumed_input} --ixtc {traj} --timestep {dt} --trajectory-stride {stride}'
    run(cmd)

    COLVAR = np.loadtxt(output, comments='#')
    return COLVAR


def get_dehydration_energy(bins, fes, cn1, cn2):
    '''
    Calculate the dehydration energy from cn1 to cn2. This function fits a spline to the free energy surface
    and estimates the energies as the spline evaluated at cn1 and cn2. For positive free energy, corresponding to
    how much free energy is needed to strip a coordinated water, cn1 should be the higher energy coordination state.

    Parameters
    ----------
    bins : np.array
        Bins for the free energy surface in coordination number
    fes : np.array
        Free energy surface (kJ/mol)
    cn1 : float
        Coordination number of state 1 to calculate dG = G_1 - G_2
    cn2 : float
        Coordination number of state 2 to calculate dG = G_1 - G_2

    Returns
    -------
    dG : float
        Free energy difference between cn1 and cn2
    
    '''
    
    from scipy.interpolate import UnivariateSpline

    spline = UnivariateSpline(bins, fes, k=4, s=0)
    dG = spline(cn1) - spline(cn2)

    return dG


class vdW_radii:
    def __init__(self):
        self.dict = {
            "H": 1.20,  # Hydrogen
            "He": 1.40,  # Helium
            "Li": 1.82,  # Lithium
            "Be": 1.53,  # Beryllium
            "B": 1.92,  # Boron
            "C": 1.70,  # Carbon
            "N": 1.55,  # Nitrogen
            "O": 1.52,  # Oxygen
            "F": 1.47,  # Fluorine
            "Ne": 1.54,  # Neon
            "Na": 2.27,  # Sodium
            "Mg": 1.73,  # Magnesium
            "Al": 1.84,  # Aluminum
            "Si": 2.10,  # Silicon
            "P": 1.80,  # Phosphorus
            "S": 1.80,  # Sulfur
            "Cl": 1.75,  # Chlorine
            "Ar": 1.88,  # Argon
            "K": 2.75,  # Potassium
            "Ca": 2.31,  # Calcium
            "Sc": 2.11,  # Scandium
            "Ti": 2.00,  # Titanium
            "V": 2.00,  # Vanadium
            "Cr": 2.00,  # Chromium
            "Mn": 2.00,  # Manganese
            "Fe": 2.00,  # Iron
            "Co": 2.00,  # Cobalt
            "Ni": 1.63,  # Nickel
            "Cu": 1.40,  # Copper
            "Zn": 1.39,  # Zinc
            "Ga": 1.87,  # Gallium
            "Ge": 2.11,  # Germanium
            "As": 1.85,  # Arsenic
            "Se": 1.90,  # Selenium
            "Br": 1.85,  # Bromine
            "Kr": 2.02,  # Krypton
            "Rb": 3.03,  # Rubidium
            "Sr": 2.49,  # Strontium
            "Y": 2.30,  # Yttrium
            "Zr": 2.15,  # Zirconium
            "Nb": 2.00,  # Niobium
            "Mo": 2.00,  # Molybdenum
            "Tc": 2.00,  # Technetium
            "Ru": 2.00,  # Ruthenium
            "Rh": 2.00,  # Rhodium
            "Pd": 1.63,  # Palladium
            "Ag": 1.72,  # Silver
            "Cd": 1.58,  # Cadmium
            "In": 1.93,  # Indium
            "Sn": 2.17,  # Tin
            "Sb": 2.06,  # Antimony
            "Te": 2.06,  # Tellurium
            "I": 1.98,  # Iodine
            "Xe": 2.16,  # Xenon
            "Cs": 3.43,  # Cesium
            "Ba": 2.68,  # Barium
            "La": 2.50,  # Lanthanum
            "Ce": 2.48,  # Cerium
            "Pr": 2.47,  # Praseodymium
            "Nd": 2.45,  # Neodymium
            "Pm": 2.43,  # Promethium
            "Sm": 2.42,  # Samarium
            "Eu": 2.40,  # Europium
            "Gd": 2.38,  # Gadolinium
            "Tb": 2.37,  # Terbium
            "Dy": 2.35,  # Dysprosium
            "Ho": 2.33,  # Holmium
            "Er": 2.32,  # Erbium
            "Tm": 2.30,  # Thulium
            "Yb": 2.28,  # Ytterbium
            "Lu": 2.27,  # Lutetium
            "Hf": 2.25,  # Hafnium
            "Ta": 2.20,  # Tantalum
            "W": 2.10,  # Tungsten
            "Re": 2.05,  # Rhenium
            "Os": 2.00,  # Osmium
            "Ir": 2.00,  # Iridium
            "Pt": 1.75,  # Platinum
            "Au": 1.66,  # Gold
            "Hg": 1.55,  # Mercury
            "Tl": 1.96,  # Thallium
            "Pb": 2.02,  # Lead
            "Bi": 2.07,  # Bismuth
            "Po": 1.97,  # Polonium
            "At": 2.02,  # Astatine
            "Rn": 2.20,  # Radon
            "Fr": 3.48,  # Francium
            "Ra": 2.83,  # Radium
            "Ac": 2.60,  # Actinium
            "Th": 2.40,  # Thorium
            "Pa": 2.00,  # Protactinium
            "U": 1.86,  # Uranium
            "Np": 2.00,  # Neptunium
            "Pu": 2.00,  # Plutonium
            "Am": 2.00,  # Americium
            "Cm": 2.00,  # Curium
            "Bk": 2.00,  # Berkelium
            "Cf": 2.00,  # Californium
            "Es": 2.00,  # Einsteinium
            "Fm": 2.00,  # Fermium
            "Md": 2.00,  # Mendelevium
            "No": 2.00,  # Nobelium
            "Lr": 2.00   # Lawrencium
        }

    def get_dict(self):
        return self.dict


class EquilibriumAnalysis:

    def __init__(self, top, traj, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Initialize the equilibrium analysis object with a topology and a trajectory from
        a production simulation with standard MD
        
        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc)
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'
            
        '''

        self.universe = mda.Universe(top, traj)
        self.n_frames = len(self.universe.trajectory)
        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')

        self.vdW_radii = vdW_radii().get_dict()

        
    def __repr__(self):
        return f'EquilibriumAnalysis object with {len(self.waters)} waters, {len(self.cations)} cations, and {len(self.anions)} anions over {self.n_frames} frames'
    

    def _find_peaks_wrapper(self, bins, data, **kwargs):
        '''Wrapper for scipy.signal.find_peaks to use with SolvationAnalysis to find cutoff'''
        
        peaks, _  = find_peaks(-data, **kwargs)
        radii = bins[peaks[0]]
        return radii
    

    def initialize_Solutes(self, step=1, **kwargs):
        '''
        Initialize the Solute objects from SolvationAnalysis for the ions. Saves the solutes
        in attributes `solute_ci` (cation) and `solute_ai` (anion). 
        
        Parameters
        ----------
        step : int
            Trajectory step for which to run the Solute
        
        '''

        from solvation_analysis.solute import Solute
        
        self.solute_ci = Solute.from_atoms(self.cations, {'water' : self.waters, 'coion' : self.anions}, 
                                           solute_name='Cation', rdf_kernel=self._find_peaks_wrapper, 
                                           kernel_kwargs={'distance':5}, **kwargs)
        self.solute_ai = Solute.from_atoms(self.anions, {'water' : self.waters, 'coion' : self.cations}, 
                                  solute_name='Anion', rdf_kernel=self._find_peaks_wrapper, 
                                  kernel_kwargs={'distance':5}, **kwargs)

        self.solute_ci.run(step=step)
        self.solute_ai.run(step=step)

        print(f"\nHydration shell cutoff for cation-water = {self.solute_ci.radii['water']:.6f}")
        print(f"Hydration shell cutoff for anion-water = {self.solute_ai.radii['water']:.6f}")


    def determine_ion_pairing_cutoffs(self, find_peaks_kwargs={'distance' : 5, 'height' : -1.1}, plot=True):
        '''
        Calculate the cation-anion radial distributions using SolvationAnalysis and identify the cutoffs for
        ion pairing events. Should plot to ensure the cutoff regimes visually look correct, since these are 
        sensitive to the peak detection algorithm. 

        Parameters
        ----------
        find_peak_kwargs : dict
            Keyword arguments for `scipy.find_peaks` used to find the first 3 minima in the cation-anion RDF,
            default={'distance' : 5, 'height' : -1.1} worked well for NaCl at 0.6 M with OPC3 water
        plot : bool
            Whether to plot the RDF with the regions shaded, default=True

        '''

        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        r = self.solute_ci.rdf_data['Cation']['coion'][0]
        rdf = self.solute_ci.rdf_data['Cation']['coion'][1]
        mins, min_props = find_peaks(-rdf, **find_peaks_kwargs)

        self.ion_pairs = Results()
        self.ion_pairs['CIP'] = (0,r[mins[0]])
        self.ion_pairs['SIP'] = (r[mins[0]],r[mins[1]])
        self.ion_pairs['DSIP'] = (r[mins[1]],r[mins[2]])
        self.ion_pairs['FI'] = (r[mins[2]],np.inf)

        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(r, rdf, color='k')

            le = 2
            for i,m in enumerate(mins[:3]):
                ax.fill_betweenx(np.linspace(0,10), le, r[m], alpha=0.25)
                ax.text((le+r[m]) / 2, 8, list(self.ion_pairs.keys())[i], ha='center')
                le = r[m]

            ax.fill_betweenx(np.linspace(0,10), le, 10, alpha=0.25)
            ax.text((le+10) / 2, 8, list(self.ion_pairs.keys())[-1])
            ax.set_xlabel('r ($\mathrm{\AA}$)')
            ax.set_ylabel('g(r)')
            ax.set_xlim(2,10)
            ax.set_ylim(0,9)
            fig.savefig('ion_pairing_cutoffs.png')
            plt.show()

        return self.ion_pairs
    

    def generate_rdfs(self, bin_width=0.05, range=(0,20), step=1, filename=None, njobs=1):
        '''
        Calculate radial distributions for the solution. This method calculates the RDFs for cation-water,
        anion-water, water-water, and cation-anion using MDAnalysis InterRDF. It saves the data in a 
        dictionary attribute `rdfs` with keys 'ci-w', 'ai-w', 'w-w', and 'ci-ai'.

        Parameters
        ----------
        bin_width : float
            Width of the bins for the RDFs, default=0.05 Angstroms
        range : array-like
            Range over which to calculate the RDF, default=(0,20)
        step : int
            Trajectory step for which to calculate the RDF, default=1
        filename : str
            Filename to save RDF data, default=None means do not save to file
        njobs : int
            Number of CPUs to run on, default=1

        Returns
        -------
        rdfs : dict
            Dictionary with all the results from InterRDF
        
        '''

        from ParallelMDAnalysis import ParallelInterRDF as InterRDF

        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {}

        print('\nCalculating cation-water RDF...')
        ci_w = InterRDF(self.cations, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_w.run(step=step, njobs=njobs)
        self.rdfs['ci-w'] = ci_w.results

        print('\nCalculating anion-water RDF...')
        ai_w = InterRDF(self.anions, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ai_w.run(step=step, njobs=njobs)
        self.rdfs['ai-w'] = ai_w.results

        print('\nCalculating water-water RDF...')
        w_w = InterRDF(self.waters, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        w_w.run(step=step, njobs=njobs)
        self.rdfs['w-w'] = w_w.results

        print('\nCalculating cation-anion RDF...')
        ci_ai = InterRDF(self.cations, self.anions, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_ai.run(step=step, njobs=njobs)
        self.rdfs['ci-ai'] = ci_ai.results

        if filename is not None:
            data = np.vstack([ci_w.results.bins, ci_w.results.rdf, ai_w.results.rdf, w_w.results.rdf, ci_ai.results.rdf]).T
            np.savetxt(filename, data, header='r (Angstroms), cation-water g(r), anion-water g(r), water-water g(r), cation-anion g(r)')

        return self.rdfs
    

    def get_coordination_numbers(self, step=1):
        '''
        Calculate the water coordination number as a function of time for both cations and anions.
        
        Parameters
        ----------
        step : int
            Trajectory step for which to calculate coordination numbers
        
        Returns
        -------
        avg_CN : np.array
            Average coordination number over the trajectory for [cations, anions]
        
        '''
    
        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        # initialize coordination number as a function of time
        self.coordination_numbers = np.zeros((2,len(self.universe.trajectory[::step])))

        for i,ts in enumerate(self.universe.trajectory[::step]):
            # first for cations
            d = distances.distance_array(self.cations, self.waters, box=ts.dimensions)
            n_coordinating = (d <= self.solute_ci.radii['water']).sum()
            self.coordination_numbers[0,i] = n_coordinating / len(self.cations)

            # then for anions
            d = distances.distance_array(self.anions, self.waters, box=ts.dimensions)
            n_coordinating = (d <= self.solute_ai.radii['water']).sum()
            self.coordination_numbers[1,i] = n_coordinating / len(self.anions)

        return self.coordination_numbers.mean(axis=1)
        

    def shell_probabilities(self, plot=False):
        '''
        Calculate the shell probabilities for each ion. Must first initialize the SolvationAnalysis Solutes.
        
        Parameters
        ----------
        plot : bool
            Whether to plot the distributions of shells, default=False
            
        '''

        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        df1 = self.solute_ci.speciation.speciation_fraction
        shell = []
        for i in range(df1.shape[0]):
            row = df1.iloc[i]
            shell.append(f'{row.coion:.0f}-{row.water:.0f}')

        df1['shell'] = shell
        self.cation_shells = df1

        df2 = self.solute_ai.speciation.speciation_fraction
        shell = []
        for i in range(df2.shape[0]):
            row = df2.iloc[i]
            shell.append(f'{row.coion:.0f}-{row.water:.0f}')

        df2['shell'] = shell
        self.anion_shells = df2
        
        if plot:
            df = df1.merge(df2, on='shell', how='outer')
            df.plot(x='shell', y=['count_x', 'count_y'], kind='bar', legend=False)
            plt.legend(['Cation', 'Anion'])
            plt.ylabel('probability')
            plt.savefig('shell_probabilities.png')
            plt.show()


    def water_dipole_distribution(self, ion='cation', radius=None, step=1):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        ion : str
            Ion to calculate the distribution for. Options are 'cation' and 'anion'. default='cation'
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only, default=None 
            means pull from SolvationAnalysis.solute.Solute
        step : int
            Step to iterate the trajectory when running the analysis, default=10

        Returns
        -------
        angles : np.array
            Angles for all waters coordinated with all ions, averaged over the number of frames

        '''

        # parse arguments
        if ion == 'cation':
            ions = self.cations
        elif ion == 'anion':
            ions = self.anions
        else:
            raise NameError("Options for kwarg ion are 'cation' or 'anion'")
        
        if radius is None:
            if ion == 'cation':
                radius = self.solute_ci.radii['water']
            elif ion == 'anion':
                radius = self.solute_ai.radii['water']

        # loop through frames and ions to get angle distributions 
        angles = []
        for i, ts in enumerate(self.universe.trajectory[::step]):
            for ci in ions:
                my_atoms = self.universe.select_atoms(f'sphzone {radius} index {ci.index}') - ci
                my_waters = my_atoms & self.waters # intersection operator to get the OW from my_atoms

                for ow in my_waters:

                    dist = ci.position - ow.position

                    # if the water is on the other side of the box, move it back
                    for d in range(3):
                        v = np.array([0,0,0])
                        v[d] = 1
                        if dist[d] >= ts.dimensions[d]/2:
                            ow.residue.atoms.translate(v*ts.dimensions[d])
                        elif dist[d] <= -ts.dimensions[d]/2:
                            ow.residue.atoms.translate(-v*ts.dimensions[d])

                    # calculate and save angles
                    pos = ow.position
                    bonded_Hs = ow.bonded_atoms
                    tmp_pt = bonded_Hs.positions.mean(axis=0)

                    v1 = ci.position - pos
                    v2 = pos - tmp_pt
                    ang = get_angle(v1, v2)*180/np.pi
                    angles.append(ang)
        
        return np.array(angles)


    def angular_water_distribution(self, ion='cation', r_range=(2,5), bin_width=0.05, start=0, step=10):
        '''
        Calculate the angular distributions of water around the ions as a function of the radius.
        The angles are theta (polar angle) and phi (azimuthal angle) from spherical coordinates 
        centered at the ion location. The polar angle defines how far the water molecule is from
        the z-axis, and the azimuthal angle defines where in the orbit the water molecule is around
        the ion. Saves images to '{ion}_theta_distribution.png' and '{ion}_phi_distribution.png'.
        
        Parameters
        ----------
        ion : str
            Ion to calculate the distributions for. Options are 'cation' and 'anion'. default='cation'
        r_range : array-like
            Range for the radius to bin the histogram in Angstroms, default=(2,5)
        bin_width : float
            Width of the bin for the radius histogram in Angstroms, default=0.05
        start : int
            Starting frame index for the trajectory to run, default=0
        step : int
            Step to iterate the trajectory when running the analysis, default=10
        
        
        '''

        if ion == 'cation':
            ions = self.cations
        elif ion == 'anion':
            ions = self.anions

        # get the bins for the histograms
        nbins = int((r_range[1] - r_range[0]) / bin_width)
        rbins = np.linspace(r_range[0], r_range[1], nbins)
        thbins = np.linspace(0,180, nbins)
        phbins = np.linspace(-180,180, nbins)

        # initialize the histograms
        th_hist,th_x,th_y = np.histogram2d([], [], bins=[rbins,thbins])
        ph_hist,ph_x,ph_y = np.histogram2d([], [], bins=[rbins,phbins])
        # ang_hist,ang_ph,ang_th = np.histogram2d([], [], bins=[phbins,thbins])

        for i, ts in enumerate(self.universe.trajectory[start::step]):
            for ci in ions:
                self.universe.atoms.translate(-ci.position) # set the ion as the origin
                my_waters = self.waters.select_atoms(f'point 0 0 0 {r_range[1]}') # select only waters near the ion

                # convert to spherical coordinates, centered at the ion
                r = np.sqrt(my_waters.positions[:,0]**2 + my_waters.positions[:,1]**2 + my_waters.positions[:,2]**2)
                th = np.degrees(np.arccos(my_waters.positions[:,2] / r))
                ph = np.degrees(np.arctan2(my_waters.positions[:,1], my_waters.positions[:,0]))

                # histogram to get the probability density
                h1,_,_ = np.histogram2d(r, th, bins=[rbins,thbins], density=True)
                h2,_,_ = np.histogram2d(r, ph, bins=[rbins,phbins], density=True)
                # h3,_,_ = np.histogram2d(ph, th, bins=[phbins,thbins], density=True)
                th_hist += h1
                ph_hist += h2
                # ang_hist += h3
                

        th_hist = th_hist / len(self.universe.trajectory[start::step]) / len(ions)
        ph_hist = ph_hist / len(self.universe.trajectory[start::step]) / len(ions)
        # ang_hist = ang_hist / len(self.universe.trajectory[start::step])

        # for the theta distribution, scale by the differential area of a strip around sphere
        s = np.zeros(th_hist.shape)
        dth = (th_y[1]-th_y[0])*np.pi/180
        th_centers = (th_y[:-1] + th_y[1:])/2
        r_centers = (th_x[:-1] + th_x[1:])/2
        for i,t in enumerate(th_centers):
            for j,r in enumerate(r_centers):
                s[j,i] = 2*np.pi * r**2 * np.sin(t*np.pi/180) * dth

        th_hist = th_hist/s

        # for the phi distribution, scale by the differential area of a strip from pole to pole
        s = np.zeros(ph_hist.shape)
        dph = (ph_y[1]-ph_y[0])*np.pi/180
        ph_centers = (ph_y[:-1] + ph_y[1:])/2
        r_centers = (ph_x[:-1] + ph_x[1:])/2
        for i,t in enumerate(ph_centers):
            for j,r in enumerate(r_centers):
                s[j,i] = 2*r**2 * dph

        ph_hist = ph_hist/s

        ## TODO
        # for the angle distribution, scale by the differential area of a square on the surface of the sphere
        # s = np.zeros(ang_hist.shape)
        # dth = (ang_th[1]-ang_th[0])*np.pi/180
        # dph = (ang_ph[1]-ang_ph[0])*np.pi/180
        # th_centers = (ang_th[:-1] + ang_th[1:])/2
        # ph_centers = (ang_ph[:-1] + ang_ph[1:])/2
        # for i,t in enumerate(th_centers):
        #     for j,r in enumerate(ph_centers):
        #         s[j,i] = 

        # ang_hist = ang_hist/s

        fig1, ax1 = plt.subplots(1,1)
        # c1 = ax1.pcolor(th_x, th_y, th_hist.T, vmin=0, vmax=0.006)
        c1 = ax1.pcolor(th_x, th_y, th_hist.T)
        ax1.set_xlim(r_range[0],r_range[1])
        ax1.set_ylim(0,180)
        ax1.set_yticks(np.arange(0,180+30,30))
        ax1.set_ylabel('$\\theta$ (degrees)')
        ax1.set_xlabel('r ($\mathrm{\AA}$)')
        fig1.colorbar(c1,label='probability')
        fig1.savefig(f'{ion}_theta_distribution.png')

        fig2, ax2 = plt.subplots(1,1)
        # c2 = ax2.pcolor(ph_x, ph_y, ph_hist.T, vmin=0, vmax=0.003)
        c2 = ax2.pcolor(ph_x, ph_y, ph_hist.T)
        ax2.set_ylabel('$\phi$ (degrees)')
        ax2.set_xlabel('r ($\mathrm{\AA}$)')
        ax2.set_xlim(r_range[0],r_range[1])
        ax2.set_ylim(-180,180)
        plt.yticks(np.arange(-180,200,45))
        fig2.colorbar(c2, label='probability')
        fig2.savefig(f'{ion}_phi_distribution.png')
        
        plt.show()

        return th_hist, (th_x, th_y), ph_hist, (ph_x, ph_y)
    

    def spatial_density(self, group='type OW', ion='cation', r_max=5, grid_pts=20, step=1):
        '''
        Plot a 3D spatial density for the locations of `group` around the ion. Creates an interactive
        plot using `plotly`.
        
        Parameters
        ----------
        group : str
            MDAnalysis atom selection language describing the group whose density will be plotted, default=`type OW`
            are the water oxygens
        ion : str
            Ion to calculate the distributions for. Options are 'cation' and 'anion'. default='cation'
        r_max : float
            Radial distance (Angstroms) to go out around the ion, default=5
        grid_pts : int
            Number of grid points for the 3D meshgrid, default=20
        step : int
            Step to iterate the trajectory when running the analysis, default=1

        Returns
        -------
        hist : np.array
            Occupation density around the ion. This is the counts over the whole trajectory divided by the maximum
            value to scale between 0 and 1.
        edges : tuple
            The (X,Y,Z) meshgrid necessary for plotting `hist` in 3D space.
        
        '''

        import plotly.graph_objects as go

        # initialize grid space, bins, and histogram
        X, Y, Z = np.meshgrid(np.linspace(-r_max,r_max,grid_pts), np.linspace(-r_max,r_max,grid_pts), np.linspace(-r_max,r_max,grid_pts))
        bins = np.linspace(-r_max, r_max, grid_pts+1)
        init_sample = np.array([[0,0,0],
                                [0,0,0]])
        hist, edges = np.histogramdd(init_sample, bins=(bins,bins,bins))
        hist[hist > 0] = 0 # set the initialized histogram to 0

        for i,ts in enumerate(self.universe.trajectory[::step]):
            for ci in self.cations:
                self.universe.atoms.translate(-ci.position)
                my_atoms = self.universe.select_atoms(f'sphzone {r_max} index {ci.index}') - ci
                my_selection = my_atoms.select_atoms(group)

                h,_ = np.histogramdd(my_selection.positions, bins=(bins,bins,bins))
                hist += h

        # hist = hist / hist.max()

        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=hist.flatten(),
        isomin=0,
        isomax=hist.max(),
        opacity=0.05, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        colorscale='jet'
        ))

        fig.show()

        return hist, (X,Y,Z)
    

    def polyhedron_size(self, ion='cation', r0=None, njobs=1, step=1):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        ion : str
            Whether to calculate the volumes and areas for the cations or anions, options are `cation` and `anion`
        r0 : float
            Hydration shell cutoff in Angstroms, default=None means will calculate using `self.initialize_Solutes()`
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.
        step : int
            Trajectory step for analysis

        Returns
        -------
        results : MDAnalysis.analysis.base.Results object
            Volume and area time series, saved in `volumes` and `areas` attributes
        
        '''

        if ion == 'cation':
            ions = self.cations
            if r0 is None:
                try:
                    r0 = self.solute_ci.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')

        elif ion == 'anion':
            ions = self.anions
            if r0 is None:
                try:
                    r0 = self.solute_ai.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')

        else:
            raise NameError("Options for kwarg ion are 'cation' or 'anion'")
        
        # Prepare the Results object
        results = Results()
        results.areas = np.zeros((len(ions), len(self.universe.trajectory[::step])))
        results.volumes = np.zeros((len(ions), len(self.universe.trajectory[::step])))

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                a,v = self._polyhedron_size_per_frame(i, ions, r0=r0)
                results.areas[:,i] = a
                results.volumes[:,i] = v

        else: # run in parallel

            import multiprocessing
            from multiprocessing import Pool
            from functools import partial
            
            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._polyhedron_size_per_frame,
                                    ions=ions,
                                    r0=r0)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            result = np.asarray(result)
            results.areas = result[:,0,:].T
            results.volumes = result[:,1,:].T

        return results
    

    def _polyhedron_size_per_frame(self, frame_idx, ions, r0):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        ions : MDAnalysis.AtomGroup
            Ions in the simulation to calculate polyhedrons for
        r0 : float
            Hydration shell cutoff for the biased ion in Angstroms, default=3.15

        Returns
        -------
        area, volume : np.ndarray
            Volumes and maximum cross-sectional areas for the polyhedrons for each ion, shape is len(ions)
        
        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        volumes = np.zeros(len(ions))
        areas = np.zeros(len(ions))

        for j,ion in enumerate(ions):

            # Unwrap the shell
            shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
            pos = self._unwrap_shell(ion, r0)
            shell.positions = pos
            pos = self._points_on_atomic_radius(shell, n_points=200)
            center = ion.position

            # Create the polyhedron with a ConvexHull and save volume
            hull = ConvexHull(pos)
            volumes[j] = hull.volume

            # Get the major axis (first principal component)
            pca = PCA(n_components=3).fit(pos[hull.vertices])

            # Find all the edges of the convex hull
            edges = []
            for simplex in hull.simplices:
                for s in range(len(simplex)):
                    edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                    edges.append(edge)

            # Create a line through the polyhedron along the principal component
            d = distances.distance_array(shell, shell, box=ions.universe.dimensions)
            t_values = np.linspace(-d.max()/2, d.max()/2, 100)
            center_line = np.array([center + t*pca.components_[0,:] for t in t_values])

            # Find the maximum cross-sectional area along the line through polyhedron
            area = 0
            for pt in center_line:
                # Find the plane normal to the principal component
                A, B, C, D = create_plane_from_point_and_normal(pt, pca.components_[0,:])

                # Find the intersection points of the hull edges with the slicing plane
                intersection_points = []
                for edge in edges:
                    p1 = pos[edge[0]]
                    p2 = pos[edge[1]]
                    intersection_point = line_plane_intersection(p1, p2, A, B, C, D)
                    if intersection_point is not None:
                        intersection_points.append(intersection_point)

                # If a slicing plane exists and its area is larger than any other, save
                if len(intersection_points) > 0:
                    intersection_points = np.array(intersection_points)
                    projected_points, rot_mat, mean_point = project_to_plane(intersection_points)
                    intersection_hull = ConvexHull(projected_points)

                    if intersection_hull.volume > area:
                        area = intersection_hull.volume
            
            areas[j] = area

        return areas, volumes


    def _unwrap_shell(self, ion, r0):
        '''
        Unwrap the hydration shell, so all coordinated groups are on the same side of the box as ion.

        Parameters
        ----------
        ion : MDAnalysis.Atom
            Ion whose shell to unwrap
        r0 : float
            Hydration shell radius for the ion

        Returns
        -------
        positions : np.ndarray
            Unwrapped coordinates for the shell

        '''

        dims = self.universe.dimensions
        shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
        dist = ion.position - shell.positions

        correction = np.where(np.abs(dist) > dims[:3]/2, # check if beyond half-box distance
                              np.sign(dist) * dims[:3],  # add or subtract box size based on sign of dist
                              0)                         # otherwise, do not move

        return shell.positions + correction
    

    def _points_on_atomic_radius(self, shell, n_points=200):
        '''
        Generate points on the "surface" of the atoms, so we have points that encompass the volume of the atoms.

        Parameters
        ----------
        shell : MDAnalysis.AtomGroup
            Hydration shell with all atoms to generate points

        Returns
        -------
        positions : np.ndarray
            Points on the "surface" of the atoms
        '''

        # get all radii
        radii = np.array([self.vdW_radii[atom.element] for atom in shell])
            
        # randomly sample theta and phi angles
        rng = np.random.default_rng()
        theta = np.arccos(rng.uniform(-1,1, (len(shell),n_points)))
        phi = rng.uniform(0,2*np.pi, (len(shell),n_points))

        # convert to Cartesian coordinates
        x = radii[:,None] * np.sin(theta) * np.cos(phi) + shell.positions[:,0,None]
        y = radii[:,None] * np.sin(theta) * np.sin(phi) + shell.positions[:,1,None]
        z = radii[:,None] * np.cos(theta) + shell.positions[:,2,None]

        positions = np.stack((x,y,z), axis=-1)
        return positions.reshape(-1,3)


class MetaDAnalysis:

    def __init__(self, COLVAR_file='COLVAR', T=300):
        '''
        Initialize the metadynamics analysis object with a collective variable file from plumed
        
        Parameters
        ----------
        COLVAR_file : str
            Name of the collective variable file form plumed, default='COLVAR'
        T : float
            Temperature (K), default=300
            
        '''
        
        self.kB = 1.380649 * 10**-23 * 10**-3 * 6.022*10**23 # Boltzmann (kJ / K)
        self.kT = self.kB*T
        self.beta = 1/self.kT

        self._COLVAR = np.loadtxt(COLVAR_file, comments='#')
        self.time = self._COLVAR[:,0]
        self.coordination_number = self._COLVAR[:,2]
        self.bias = self._COLVAR[:,3]
        self.lwall = self._COLVAR[:,4]
        self.lwall_force = self._COLVAR[:,5]
        self.uwall = self._COLVAR[:,6]
        self.uwall_force = self._COLVAR[:,7]


    def calculate_FES(self, bins=np.linspace(2,9,250), mintozero=True, filename=None):
        '''
        Calculate the free energy surface by reweighting the histogram
        
        Parameters
        ----------
        bins : np.array
            Bins for the histogram, default=np.linspace(2,9,250)
        mintozero : bool
            Whether to shift the minimum of the FES to 0, default=True
        filename : str
            Name of the file to save the FES, default=None (do not save)

        Returns
        -------
        bin_centers : np.array
            Coordination numbers for the FES
        fes : np.array
            FES along the coordination number in kJ/mol

        '''

        x = self.coordination_number
        V_x = self.bias
        V_wall = self.lwall + self.uwall
        self.weights = np.exp(self.beta*V_x + self.beta*V_wall)

        hist, bin_edges = np.histogram(x, bins=bins, weights=self.weights)
        self.bin_centers = bin_edges[:-1] + (bin_edges[:-1] - bin_edges[1:]) / 2
        self.fes = -self.kT*np.log(hist)

        if mintozero:
            self.fes -= self.fes.min()

        if filename is not None:
            np.savetxt(filename, np.vstack([self.bin_centers, self.fes]).T, header='coordination number, free energy (kJ/mol)')

        return self.bin_centers, self.fes
    

class UmbrellaSim:

    def __init__(self, COLVAR_file='COLVAR', start=10, stop=-1, by=100):
        '''
        Initialize an umbrella simulation object with a COLVAR file

        Parameters
        ----------
        COLVAR_file : str
            Pattern for the COLVAR files from plumed, default=COLVAR
        start : int
            Index of the first coordination number to read, default=10
        stop : int 
            Index of the last coordination number to read, default=-1
        by : int
            Step by which to read COLVAR entries, default=100

        '''
        
        tmp = np.loadtxt(COLVAR_file, comments='#')

        if tmp.shape[1] == 5: # for a single restrain
            cols = ['time', 'n', 't', 'r.bias', 'r.force2']
            self.data = pd.DataFrame(tmp[start:stop:by,:], columns=cols)
            self.time = self.data.time.to_numpy()
            self.coordination_number = self.data.t.to_numpy()
            self.bias = self.data['r.bias'].to_numpy()
            self.force = self.data['r.force2'].to_numpy()
        elif tmp.shape[1] == 9: # for 2 restraints in water and ion coordination
            cols = ['time', 'n1', 't1', 'r1.bias', 'r1.force2', 'n2', 't2', 'r2.bias', 'r2.force2']
            self.data = pd.DataFrame(tmp[start:stop:by,:], columns=cols)
            self.time = self.data.time.to_numpy()
            self.coordination_number = self.data.t1.to_numpy()
            self.ion_coordination_number = self.data.t2.to_numpy()
            self.water_bias = self.data['r1.bias'].to_numpy()
            self.ion_bias = self.data['r2.bias'].to_numpy()
        else:
            raise ValueError(f'Cannot read file {COLVAR_file}')


    def get_coordination_numbers(self, biased_ion, group, radius, step=1):
        '''
        Calculate the discrete water coordination number as a function of time for biased ion.
        
        Parameters
        ----------
        biased_ion : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the biased ion
        group : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the group to calculate the coordination numbers for (e.g. waters, cations, anions)
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        step : int
            Trajectory step for which to calculate coordination numbers
        
        Returns
        -------
        discrete_coordination_numbers : np.array
            Discrete coordination numbers over the trajectory
        
        '''

        if self.universe is None:
            raise NameError('No universe data found. Try `create_Universe()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # initialize coordination number as a function of time
        self.discrete_coordination_numbers = np.zeros(len(self.universe.trajectory[::step]))

        for i,ts in enumerate(self.universe.trajectory[::step]):
            d = distances.distance_array(ion, group, box=ts.dimensions)
            self.discrete_coordination_numbers[i] = (d <= radius).sum()

        return self.discrete_coordination_numbers


    def get_performance(self, log_file='prod.log'):
        '''
        Get the performance of this umbrella simulation from the log file
        
        Parameters
        ----------
        log_file : str
            Name of the log file for the umbrella simulation, default=prod.log

        Returns
        -------
        performance : float
            Performance for the simulation in ns/day

        '''

        f = open(log_file)
        lines = f.readlines()
        tmp = [float(line.split()[1]) for line in lines if line.startswith('Performance')]
        self.performance = tmp[0]

        return self.performance
    

    def create_Universe(self, top, traj=None, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Create an MDAnalysis Universe for the individual umbrella simulation.

        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc)
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'

        Returns
        -------
        universe : MDAnalysis.Universe object
            MDAnalysis Universe with the toplogy and coordinates for this umbrella

        '''

        self.universe = mda.Universe(top, traj)    

        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')
    
        return self.universe


    def initialize_Solute(self, ion, cutoff, step=1):
        '''
        Initialize the Solute object from SolvationAnalysis for the ion. Saves the solute
        in attribute `solute`. 
        
        Parameters
        ----------
        ion : MDAnalysis.AtomGroup or str
            Ion to create a Solute object for, if a str should be MDAnalysis selection language
        cutoff : float
            Hydration shell cutoff in Angstroms
        step : int
            Trajectory step for which to run the Solute

        Returns
        -------
        solute : solvation_analysis.solute.Solute
            SolvationAnalysis Solute object for `ion` with hydration shell `cutoff`
            
        '''

        from solvation_analysis.solute import Solute

        if isinstance(ion, str): # if provided selection language, make AtomGroup
            g = self.universe.select_atoms(ion)
        else: # else assume input is AtomGroup
            g = ion

        if g[0].charge > 0:
            other_ions = self.cations - g
            coions = self.anions
            name = 'cation'
        elif g[0].charge < 0:
            other_ions = self.anions - g
            coions = self.cations
            name = 'anion'
        else:
            raise TypeError('Your ion is not charged, and so not an ion.')

        
        self.solute = Solute.from_atoms(g, {'water' : self.waters, 'ion' : other_ions, 'coion' : coions}, 
                                        solute_name=name, radii={'water' : cutoff, 'ion' : cutoff, 'coion' : cutoff})
        self.solute.run(step=step)

        return self.solute
    

class UmbrellaAnalysis:

    def __init__(self, n_umbrellas, COLVAR_file='COLVAR_', start=10, stop=-1, by=100, T=300, verbose=True):
        '''
        Initialize the umbrella sampling analysis object with collective variable files for each simulation
        
        Parameters
        ----------
        n_umbrellas : int
            Number of umbrella simulations
        COLVAR_file : str
            Pattern for the COLVAR files from plumed, default=COLVAR_
        start : int
            Index of the first coordination number to read, default=10
        stop : int
            Index of the last coordination number to read, default=-1
        by : int
            Step by which to read COLVAR entries, default=100
        T : float
            Temperature (K), default=300
        verbose : bool
            Verbosity, controls whether to print detailed information and progress bars, default=True

        '''

        # initialize some variables
        self.kB = 1.380649 * 10**-23 * 10**-3 * 6.022*10**23 # Boltzmann (kJ / K)
        self.kT = self.kB*T
        self.beta = 1/self.kT
        self._fes = None
        self.universe = None
        self.coordination_numbers = None
        self.polyhedron_sizes = None
        self.verbose = verbose

        # read in collective variable files
        self.colvars = []

        for i in range(n_umbrellas):
            filename = f'{COLVAR_file}{i}'
            self.colvars.append(UmbrellaSim(filename, start=start, stop=stop, by=by))

        self.vdW_radii = vdW_radii().get_dict() # I am sure there is a better way to do this... but I am not taking the time now

    
    def __repr__(self):
        if self.universe is not None:
            return f'UmbrellaAnalysis object with {len(self.colvars)} simulations and {len(self.universe.trajectory)} frames'
        else:
            return f'UmbrellaAnalysis object with {len(self.colvars)} simulations'

        
    def calculate_FES(self, CN0_k, KAPPA=100, n_bootstraps=0, nbins=200, d_min=2, d_max=8, bw=0.02, error=True, mintozero=True, filename=None):
        '''
        Calculate the free energy surface with pymbar
        
        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers
        KAPPA : float, array-like
            Strength of the harmonic potential (kJ/mol/CN^2), default=100
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        nbins : int
            Number of bins for the free energy surface
        d_min : float
            Minimum coordination number for the free energy surface
        d_max : float
            Maximum coordination number for the free energy surface
        bw : float
            Bandwidth for the KDE
        error : bool
            Calculate error. If True and n_bootstraps > 0, then will calculate the bootstrapped error. 
            Otherwise, calculates the analytical histogram error, default=True
        mintozero : bool
            Shift the minimum of the free energy surface to 0
        filename : str
            Name of the file to save the free energy surface, default=None

        Returns
        -------
        bin_centers : np.array
            Coordination numbers for the FES
        fes : np.array
            FES along the coordination number in kJ/mol

        '''

        # Step 1: Subsample timeseries
        print('Subsampling timeseries...')
        u_kn, u_kln, N_k, d_kn = self._subsample_timeseries(error=error, plot=True)
        
        # Step 2: Bin the data
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(d_min, d_max, nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        # Step 3: Evaluate reduced energies in all umbrellas
        print('Evaluating energies...')
        u_kln = self._evaluate_reduced_energies(CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA)

        # Step 4: Compute and output the FES
        print('Calculating the free energy surface...')
        fes = pymbar.FES(u_kln, N_k, verbose=False)
        d_n = pymbar.utils.kn_to_n(d_kn, N_k=N_k)
        if not error:
            fes.generate_fes(u_kn, d_n, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method=None)
            results['df_i'] = np.zeros(len(results['f_i']))
        elif n_bootstraps == 0:
            fes.generate_fes(u_kn, d_n, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='analytical')
        else:
            fes.generate_fes(u_kn, d_n, fes_type='kde', kde_parameters=kde_params, n_bootstraps=n_bootstraps)
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='bootstrap')

        if mintozero:
            results['f_i'] = results['f_i'] - results['f_i'].min()

        # Step 5: Save FES information in the object
        print('Saving results...')
        self.umbrella_centers = CN0_k
        self._u_kln = u_kln
        self.u_kn = u_kn
        self._N_k = N_k
        self._fes = fes                     # underlying pymbar.FES object
        self._results = results             # underlying results object
        self.bin_centers = bin_center_i
        self.fes = results['f_i']*self.kT
        self.error = results['df_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([self.bin_centers, self.fes, self.error]).T, header='coordination number, free energy (kJ/mol), error (kJ/mol)')

        return self.bin_centers, self.fes
    

    def calculate_discrete_FE(self, biased_ion, radius, n_bootstraps=0, cn_range=None, filename=None, **kwargs):
        '''
        Calculate the free energies associated with the discrete coordination number states from the continuous coordination number simulations

        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        cn_range : array-like
            Coordination number range to calculate the discrete free energoes, default=None means use the min and max observed
        filename : str
            Name of the file to save the discrete free energies

        Returns
        -------
        results : MDAnalysis Results class with attributes `coordination_number`, `free_energy`, and `error`
            Free energies for the discrete coordination numbers. If `n_bootstraps` is 0, all errors will be 0.

        '''

        if self._fes is None:
            raise ValueError('Continuous coordination number free energy surface not found. Try `calculate_FES()` first')
        
        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # determine indices to remove to ensure COLVAR time and Universe time match
        n_sims = len(self.colvars)
        total_frames = self.universe.trajectory.n_frames
        umb_frames = self.colvars[0].time.shape[0]
        to_remove = np.arange(umb_frames, total_frames+1, umb_frames+1)

        if self.coordination_numbers is None:
            cn = self.get_coordination_numbers(ion, radius, filename='tmp_CN.csv', **kwargs)
        else:
            cn = self.coordination_numbers
        
        cn = np.delete(cn, to_remove)
        if cn_range is None:
            cn_range = (cn.min(), cn.max())
        print(f'\tDiscrete coordination numbers range: ({cn.min()}, {cn.max()})')

        # prepare the Results object
        results = Results()
        results.coordination_number = np.arange(cn_range[0], cn_range[1]+1)
        results.free_energy = np.zeros((cn_range[1] - cn_range[0] + 1))
        results.error = np.zeros((cn_range[1] - cn_range[0] + 1))

        # get the discrete bins
        bin_edges = np.arange(cn_range[0]-0.5, cn_range[1]+1.5)
        bins = np.arange(cn_range[0], cn_range[1]+1)

        if n_bootstraps > 0:
            print(f'Calculating discrete free energies with {n_bootstraps} bootstraps...')
            # if calculating error, get uncorrelated discrete coordination numbers
            N_k = self._N_k
            cn_kn = cn.reshape((n_sims, umb_frames))
            for k in range(n_sims):
                idx = self.uncorrelated_indices[k]
                cn_kn[k, 0:N_k[k]] = cn_kn[k, idx]

            cn = pymbar.utils.kn_to_n(cn_kn, N_k=N_k)

            self._fes.generate_fes(self.u_kn, cn, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges}, n_bootstraps=n_bootstraps)
            res = self._fes.get_fes(bins, reference_point='from-lowest', uncertainty_method='bootstrap')
            results.error = res['df_i']*self.kT

        else:
            print(f'Calculating discrete free energies without error...')
            # do not calculate error, since unsure what histogram error means for this case
            self._fes.generate_fes(self.u_kn, cn, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            res = self._fes.get_fes(bins, reference_point='from-lowest', uncertainty_method=None)

        # convert to kJ/mol and save in Results object
        results.free_energy = res['f_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([results.coordination_number, results.free_energy, results.error]).T, header='coordination number, free energy (kJ/mol), error (kJ/mol)')

        return results
    

    def calculate_area_FES(self, area_range=None, nbins=50, n_bootstraps=0, filename=None):
        '''
        Calculate the free energy surfaces in the coordination shell cross-sectional areas collective variable space

        Parameters
        ----------
        area_range : array-like, shape (2,)
            Min and max area values to calculate the FES, default=None means use the minimum and maximum areas from the timeseries
        nbins : int
            Number of bins for the FES histogram, default=50
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        filename : str
            Name of the file to save the FES in area, default=None means do not save

        Returns
        -------
        results : MDAnalysis Results class with attributes `coordination_number`, `free_energy`, and `error`
            Free energies for the discrete coordination numbers. If `n_bootstraps` is 0, all errors will be 0.

        '''

        if self._fes is None:
            raise ValueError('Continuous coordination number free energy surface not found. Try `calculate_FES()` first')
        
        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')
        
        if self.polyhedron_sizes is None:
            raise ValueError('No polyhedron size data. Try  `polyhedron_size()` first')

        # load in polyhedrons and remove extra frames
        n_sims = len(self.colvars)
        total_frames = self.universe.trajectory.n_frames
        umb_frames = self.colvars[0].time.shape[0]
        to_remove = np.arange(umb_frames, total_frames+1, umb_frames+1)

        poly = self.polyhedron_sizes
        area = np.delete(poly.areas, to_remove)

        # get uncorrelated areas
        N_k = self._N_k
        area_kn = area.reshape((n_sims, umb_frames))
        for k in range(n_sims):
            idx = self.uncorrelated_indices[k]
            area_kn[k, 0:N_k[k]] = area_kn[k, idx]

        area = pymbar.utils.kn_to_n(area_kn, N_k=N_k)

        if area_range is None:
            area_range = (area.min(), area.max())

        # bin the areas for the FES
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(area_range[0], area_range[1], nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        # generate the FES in area
        self._fes.generate_fes(self.u_kn, area, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges}, n_bootstraps=n_bootstraps)
        res = self._fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='bootstrap')
        res['f_i'] = res['f_i']*self.kT
        res['df_i'] = res['df_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([bin_center_i, res['f_i'], res['df_i']]).T, header='max polyhedron area (Angstroms^2), free energy (kJ/mol), error (kJ/mol)')

        return bin_center_i, res['f_i'], res['df_i']


    def get_decoordination_energy_error(self, discrete_FE):
        '''
        Calculate the bootstrapped error for the de-coordination free energies. The de-coordination energies
        are the sequential differences between dG(CN). Perform the differences within the bootstrap, so you 
        get an estimate of the 1st and 2nd de-coordination free energy for every bootstrap. This is possible 
        by accessing the `histogram_datas` attribute in the pymbar.FES object. As a result, this method
        requires that UmbrellaAnalysis.calculate_discrete_FE() to be complete.
        
        For example, for Na+ whose minimum-energy coordination number is 6,
        the first de-coordination free energy is dG(CN = 5) - dG(CN = 6), and the second de-coordination 
        free energy is dG(CN = 4) - dG(CN = 5).
        
        The bootstrapped error is standard deviation across the bootstrapped samples.
        
        Parameters
        ----------
        discrete_FE : MDAnalysis.analysis.base.Results
            MDAnalysis Results class from the UmbrellaAnalysis.calculate_discrete_FE() call. Should have the
            attributes `coordination_number`, `free_energy`, and `error`.

        Returns
        -------
        dG : np.ndarray, shape=(2,)
            First 2 de-coordination free energies. Note: the bootstrapped data are in self.dG_boot
        dG_err : np.ndarray, shape=(2,)
            The bootstrapped error in the first 2 de-coordination free energies. 

        '''

        # get some variables from the pymbar.FES object
        fes = self._fes
        x = discrete_FE.coordination_number.reshape(-1,1)
        bins = fes.histogram_data['bins']
        loc_indices = np.digitize(x, bins[0]) - 1
        n_bootstraps = len(fes.histogram_datas)

        h = fes.histogram_data
        j = h['f'].argmin()
        f_i = h['f'] - h['f'][j]

        # build a mapping file to get the index of the free energy that corresponds to a given CN
        x_map = {}
        for i,l in enumerate(loc_indices):
            bin_label = h['bin_label'][tuple(l)]
            bin_order = h['bin_order'][bin_label]
            x_map[x[i][0]] = bin_order
            if bin_order == j: # save which CN is the minimum free energy
                x_min = x[i][0]

        # get the CN of 1st and 2nd de-coordination
        x_1st = x_min - 1
        x_2nd = x_min - 2

        if x_2nd not in x_map.keys():
                raise ValueError(f'We do not have data for the 2nd de-coordination (CN = {x_2nd}). The lowest CN is {min(x_map.keys())}.')
        
        # calculate the 1st and 2nd de-coordination free energies
        f_x_min = f_i[x_map[x_min]] # will be zero for `from-lowest`
        f_x_1st = f_i[x_map[x_1st]]
        f_x_2nd = f_i[x_map[x_2nd]]

        dG = np.zeros(2)
        dG[0] = (f_x_1st - f_x_min)*self.kT # kJ/mol
        dG[1] = (f_x_2nd - f_x_1st)*self.kT

        # now calculate the uncertainties
        self.dG_boot = np.zeros((2,n_bootstraps))
        for b in range(n_bootstraps):
            f_boot = fes.histogram_datas[b]['f'] - fes.histogram_datas[b]['f'][j]

            # get the free energies for those CN
            f_x_min = f_boot[x_map[x_min]] # will be zero for `from-lowest`
            f_x_1st = f_boot[x_map[x_1st]]
            f_x_2nd = f_boot[x_map[x_2nd]]

            # calculate the 1st and 2nd de-coordination free energies for this bootstrap
            self.dG_boot[0,b] = (f_x_1st - f_x_min)*self.kT # kJ/mol
            self.dG_boot[1,b] = (f_x_2nd - f_x_1st)*self.kT

        dG_err = np.std(self.dG_boot, axis=1)

        return dG, dG_err
    

    def show_overlap(self):
        '''
        Compute the overlap matrix and plot as a heatmap
        
        Returns
        -------
        heatmap : sns.Axes
            Heatmap of overlap from seaborn
        
        '''

        import seaborn as sns
        
        overlap = self._fes.mbar.compute_overlap()

        df = pd.DataFrame(overlap['matrix'], columns=[i for i in range(len(self.colvars))])
        fig, ax = plt.subplots(1,1, figsize=(10,8))
        heatmap = sns.heatmap(df, annot=True, fmt='.2f', ax=ax)

        return heatmap
    

    def average_coordination_number(self, CN0_k=None, KAPPA=100):
        '''
        Compute the average coordination number with a Boltzmann-weighted average
        
        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers, default=None because it is not
            necessary if there is already an underlying MBAR object
        KAPPA : float
            Strength of the harmonic potential (kJ/mol/nm^2), default=100

        Returns
        -------
        results['mu'] : float
            Boltzmann-weighted average coordination number
        results['sigma'] : float
            Standard deviation of the mean coordination number

        '''

        import pymbar

        # first, subsample the timeseries to get d_kn (the uncorrelated coordination numbers)
        u_kn, u_kln, N_k, d_kn = self._subsample_timeseries()

        if self._fes is None: # if no underlying MBAR object, create one
            u_kln = self._evaluate_reduced_energies(CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA)
            mbar = pymbar.MBAR(u_kln, N_k)

        else: # otherwise get it from FES
            mbar = self._fes.get_mbar()

        results = mbar.compute_expectations(d_kn)

        return results['mu'], results['sigma']

        
    def find_minima(self, plot=False, method='find_peaks', **kwargs):
        '''
        Find the local minima of the free energy surface. `method` options are 'find_peaks'
        and 'spline_roots'. 'find_peaks' uses scipy.signal find_peaks to locate the minima
        based on peak properties. 'spline_roots' fits a UnivariateSpline to the FES and finds
        its minima by solving df/dx=0. 

        Parameters
        ----------
        plot : bool
            Whether to plot the minima on the free energy surface, default=False
        method : str
            Method to use to locate the minima, default='find_peaks'
        
        Returns
        -------
        minima_loc : np.array
            Bin locations of the minima in the FES

        '''

        if method == 'find_peaks':
    
            from scipy.signal import find_peaks

            peaks,_ = find_peaks(-self.fes, **kwargs)
            self.minima_idx = peaks
            self.minima_locs = self.bin_centers[peaks]
            self.minima_vals = self.fes[peaks]

        elif method == 'spline_roots':

            self.spline = self._fit_spline(**kwargs)
            self.minima_locs, self.minima_vals = self._get_spline_minima()

        if plot:
            plt.plot(self.bin_centers, self.fes)
            plt.scatter(self.minima_locs, self.minima_vals, marker='x', c='r')
            plt.xlabel('Coordination number')
            plt.ylabel('Free energy (kJ/mol)')

        return self.minima_locs
    
    
    def get_dehydration_energy(self, cn1, cn2, uncertainty_method=None):
        '''
        Calculate the dehydration energy from cn1 to cn2. This function fits a spline to the free energy surface
        and estimates the energies as the spline evaluated at cn1 and cn2. For positive free energy, corresponding to
        how much free energy is needed to strip a coordinated water, cn1 should be the higher energy coordination state.

        Parameters
        ----------
        cn1 : float
            Coordination number of state 1 to calculate dG = G_1 - G_2
        cn2 : float
            Coordination number of state 2 to calculate dG = G_1 - G_2
        uncertainty_method : str
            Method to calculate the uncertainty. Currently, the only method available is 'bootstrap'. Default=None means
            it will not calculate uncertainty.

        Returns
        -------
        dG : float
            Free energy difference between cn1 and cn2
        dG_std : float
            Standard deviation in the free energy difference, only returned if uncertainty_method='bootstrap'
        
        '''

        if uncertainty_method == 'bootstrap':
            n_bootstraps = len(self._fes.kdes)
            x = self.bin_centers.reshape(-1,1)

            dG_boots = np.zeros(n_bootstraps)
            for b in range(n_bootstraps):
                fes_boot = -self._fes.kdes[b].score_samples(x)*self.kT
                spline = self._fit_spline(self.bin_centers, fes_boot)
                dG_boots[b] = spline(cn1) - spline(cn2)

            return dG_boots.mean(), dG_boots.std()
        
        else:
            spline = self._fit_spline(self.bin_centers, self.fes)
            dG = spline(cn1) - spline(cn2)

            return dG
        

    def rdfs_by_coordination(self, biased_ion, CN_range, bin_width=0.05, range=(0,20)):
        '''
        Calculate radial distribution functions as a function of the biased coordination number. This method 
        calculates the RDFs for ion-water, ion-ion, and ion-coion using MDAnalysis InterRDF. It saves 
        the data in a dictionary attribute `rdfs` with keys 'i-w', 'i-i', 'i-ci'. Each key corresponds 
        to a dictionary of coordination numbers. 
        
        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        CN_range : array-like
            Range of coordination numbers to calculate the RDF for
        bin_width : float
            Width of the bins for the RDFs, default=0.05
        range : array-like
            Range over which to calculate the RDF, default=(0,20)

        Returns
        -------
        rdfs : dict
            Dictionary of dictionaries with all the results from InterRDF
        
        '''

        if self.coordination_numbers is None:
            raise ValueError('Discrete coordination number data not found. Try `get_coordination_numbers()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # decide which ions are the same as the biased ion
        if ion in self.cations:
            ions = self.cations - ion
            coions = self.anions - ion
        elif ion in self.anions:
            ions = self.anions - ion
            coions = self.cations - ion

        from MDAnalysis.analysis import rdf

        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {
            'i-w'  : {},
            'i-i'  : {},
            'i-ci' : {}
        }

        for CN in CN_range:
            idx = self.coordination_numbers == CN
            print(f'Coordination number {CN}: {idx.sum()} frames')

            if idx.sum() > 0:
                i_w = rdf.InterRDF(ion, self.waters, nbins=nbins, range=range, norm='rdf')
                i_w.run(frames=idx)
                self.rdfs['i-w'][CN] = i_w.results

                i_i = rdf.InterRDF(ion, ions, nbins=nbins, range=range, norm='rdf')
                i_i.run(frames=idx)
                self.rdfs['i-i'][CN] = i_i.results

                i_ci = rdf.InterRDF(ion, coions, nbins=nbins, range=range, norm='rdf')
                i_ci.run(frames=idx)
                self.rdfs['i-ci'][CN] = i_ci.results

        return self.rdfs


    def angular_distributions_by_coordination(self, biased_ion, CN_range, bin_width=0.05, range=(1,10)):
        '''
        Calculate water angular distributions as a function of the biased coordination number. This method
        saves the data in a dictionary attribute `angular_distributions` with keys 'theta' and 'phi'. 
        
        Parameters
        ----------
        biased_ion : MDAnalysis.Atom
            An MDAnalysis Atom of the biased ion
        CN_range : array-like
            Range of coordination numbers to calculate the distributions for
        bin_width : float
            Width of the bins in the r direction, default=0.05
        range : array-like
            Radial range over which to calculate the distributions, default=(1,10)

        Returns
        -------
        angular_distributions : dict
            Dictionary of dictionaries with all the results
        
        '''

        if self.coordination_numbers is None:
            raise ValueError('Discrete coordination number data not found. Try `get_coordination_numbers()` first')

        nbins = int((range[1] - range[0]) / bin_width)
        rbins = np.linspace(range[0], range[1], nbins)
        thbins = np.linspace(0,180, nbins)
        phbins = np.linspace(-180,180, nbins)

        self.angular_distributions = {
            'theta' : {},
            'phi' : {}
        }

        for CN in CN_range:
            th_hist,th_x,th_y = np.histogram2d([], [], bins=[rbins,thbins])
            ph_hist,ph_x,ph_y = np.histogram2d([], [], bins=[rbins,phbins])

            idx = self.coordination_numbers == CN
            print(f'Coordination number {CN}: {idx.sum()} frames')

            if idx.sum() > 0:
                for i, ts in enumerate(self.universe.trajectory[idx]):
                    d = distances.distance_array(mda.AtomGroup([biased_ion]), self.waters, box=ts.dimensions)
                    closest_water = self.waters[d.argmin()]
                    self.universe.atoms.translate(-biased_ion.position) # set the ion as the origin
                    my_waters = self.waters.select_atoms(f'(point 0 0 0 {range[1]}) and (not index {closest_water.index})', updating=True) # select only waters near the ion

                    if len(my_waters) > 0:
                        # rotate system so z axis is oriented with ion-closest water vector
                        v2 = np.array([0,0,1])
                        rotation_matrix = self._rotation_matrix_from_vectors(closest_water.position, v2)
                        positions = rotation_matrix.dot(my_waters.positions.T).T

                        # convert to spherical coordinates, centered at the ion
                        r = np.sqrt(positions[:,0]**2 + positions[:,1]**2 + positions[:,2]**2)
                        th = np.degrees(np.arccos(positions[:,2] / r))
                        ph = np.degrees(np.arctan2(positions[:,1], positions[:,0]))

                        # histogram to get the probability density
                        h1,_,_ = np.histogram2d(r, th, bins=[rbins,thbins])
                        h2,_,_ = np.histogram2d(r, ph, bins=[rbins,phbins])
                        
                        th_hist += h1
                        ph_hist += h2

                th_data = {'r' : th_x, 'theta' : th_y, 'hist' : th_hist.T}
                ph_data = {'r' : ph_x, 'phi' : ph_y, 'hist' : ph_hist.T}
                self.angular_distributions['theta'][CN] = th_data
                self.angular_distributions['phi'][CN] = ph_data

        return self.angular_distributions
    

    def water_dipole_distribution(self, biased_ion, radius, n_max=12, njobs=1, step=1):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        biased_ion : MDAnalysis.Atom
            Ion to calculate the distribution for, the ion whose coordination shell has been biased.
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only
        n_max : int
            Maximum number of coordinated waters, if discrete coordination numbers have been calculated, will use
            the max of `self.coordination_numbers`, default=12
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.
        step : int
            Step to iterate the trajectory when running the analysis, default=1

        Returns
        -------
        results :  MDAnalysis Results class with attribute angles
            Angles for all waters coordinated with biased ion

        '''

        if self.coordination_numbers is not None: # if discrete coordination numbers have been calculated create a time x max number coordinating array
            n_max = self.coordination_numbers.max()

        # prepare the Results object
        results = Results()
        results.angles = np.empty((len(self.universe.trajectory[::step]),n_max))
        results.angles[:] = np.nan # should be NaN if not specified

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                ang = self._water_dipole_per_frame(i, biased_ion, radius=radius)
                results.angles[i,:] = ang

        else: # run in parallel

            import multiprocessing
            from multiprocessing import Pool
            from functools import partial
            
            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._water_dipole_per_frame,
                                    biased_ion=biased_ion,
                                    radius=radius,
                                    n_max=n_max)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            ang = np.asarray(result)
            results.angles = ang 
        
        return results
    

    def polyhedron_size(self, biased_ion, r0=3.15, njobs=1, step=1):
        '''
        Calculate the maximum cross-sectional areas and volumes as time series for coordination shells.
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        biased_ion : str, MDAnalysis.Atom
            Biased ion in the simulation to calculate polyhedrons for
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.

        Returns
        -------
        results : MDAnalysis Results class with attributes `volumes` and `areas`
            Volume and maximum cross-sectional area for the polyhedron
        
        '''

        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')

        # prepare the Results object
        results = Results()
        results.areas = np.zeros(len(self.universe.trajectory[::step]))
        results.volumes = np.zeros(len(self.universe.trajectory[::step]))

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                a,v = self._polyhedron_size_per_frame(i, biased_ion, r0=r0)
                results.areas[i] = a
                results.volumes[i] = v

        else: # run in parallel

            import multiprocessing
            from multiprocessing import Pool
            from functools import partial
            
            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._polyhedron_size_per_frame,
                                    biased_ion=biased_ion,
                                    r0=r0,
                                    for_visualization=False)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            result = np.asarray(result)
            results.areas = result[:,0]
            results.volumes = result[:,1]

        self.polyhedron_sizes = results

        return results


    def ion_pairing(self, biased_ion, ion_pair_cutoffs, plot=False, njobs=1):
        '''
        Calculate the frequency of ion pairing events as defined in https://doi.org/10.1063/1.4901927 
        over the umbrella trajectories. This method saves the time series of the ion pairing states for
        the biased ion and returns the frequency distribution.

        Parameters
        ----------
        biased_ion : str or MDAnalysis.AtomGroup
            Biased ion in the simulation
        ion_pair_cutoffs : dict or MDAnalysis.analysis.base.Results of tuples
            Dictionary with keys ['CIP', 'SIP', 'DSIP', 'FI'] with values (min, max) for each region
        plot : bool
            Whether to plot the distribution, default=False
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use MDAnalysis
            OpenMP backend to calculate distances.
        
        Returns
        -------
        freq : pandas.DataFrame
            Distribution of ion pairing frequencies, sums to 1

        '''

        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # check whether biased ion is cation or anion
        if ion in self.cations:
            coions = self.anions
        elif ion in self.anions:
            coions = self.cations
        else:
            raise ValueError(f'Biased ion {ion} does not belong to anions ({self.anions}) or cations ({self.cations}).')

        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')

        self.ion_pairs = Results()
        self.ion_pairs['CIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['SIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['DSIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['FI'] = np.zeros(len(self.universe.trajectory))

        # set backend depending on number of CPUs available
        if njobs == 1:
            backend = 'serial'
        else:
            backend = 'OpenMP'

        # increment the state the biased ion is in
        for i,ts in tqdm(enumerate(self.universe.trajectory)):
            d = distances.distance_array(ion, coions, box=ts.dimensions, backend=backend)[0,:]
            idx, dist = d.argmin(), d.min()
            for ip,range in ion_pair_cutoffs.items():                
                if range[0] <= dist <= range[1]:
                    self.ion_pairs[ip][i] += 1
                    break

        # calculate the distribution from the time series
        df = pd.DataFrame(self.ion_pairs.data)
        freq = pd.DataFrame(df.sum() / len(self.universe.trajectory))

        if plot:
            freq.plot(kind='bar', legend=None)
            plt.ylabel('Frequency')
            plt.savefig('ion_pair_distribution.png')
            plt.show()

        return freq


    def create_Universe(self, top, traj=None, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Create an MDAnalysis Universe for the individual umbrella simulation.

        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc), default=None
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'

        Returns
        -------
        universe : MDAnalysis.Universe object
            MDAnalysis Universe with the toplogy and coordinates for this umbrella

        '''

        self.universe = mda.Universe(top, traj)    

        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')
    
        return self.universe
    

    def get_coordination_numbers(self, biased_ion, radius, filename=None, njobs=1, verbose=None):
        '''
        Calculate the discrete total coordination number as a function of time for biased ion.
        
        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        filename : str
            Name of the file to save the discrete coordination numbers, should be a csv file, default=None means do not save
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use MDAnalysis
            OpenMP backend to calculate distances.
        verbose : bool
            Whether to show progress bar, default=None means use UmbrellaAnalysis settings
        
        Returns
        -------
        coordination_numbers : np.array
            Discrete coordination numbers over the trajectory
        
        '''

        if self.universe is None:
            raise NameError('No universe data found. Try `create_Universe()` first')
        
        if verbose is None:
            verbose = self.verbose

        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        if njobs == 1: # run on 1 CPU

            print(f'\nCalculating the discrete coordination numbers...')

            # initialize coordination number as a function of time
            self.coordination_numbers = np.zeros(len(self.universe.trajectory))

            if verbose:
                for i,ts in tqdm(enumerate(self.universe.trajectory)):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='serial')
            else:
                for i,ts in enumerate(self.universe.trajectory):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='serial')

        else: # run in parallel

            print(f'\nCalculating the discrete coordination numbers using {njobs} CPUs...')

            # initialize coordination number as a function of time
            self.coordination_numbers = np.zeros(len(self.universe.trajectory))

            if verbose:
                for i,ts in tqdm(enumerate(self.universe.trajectory)):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='OpenMP')
            else:
                for i,ts in enumerate(self.universe.trajectory):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='OpenMP')

        if filename is not None:
            df = pd.DataFrame()
            df['idx'] = np.arange(0,len(self.coordination_numbers))
            df['coordination_number'] = self.coordination_numbers
            df.to_csv(filename, index=False)

        return self.coordination_numbers


    def _subsample_timeseries(self, error=True, plot=False):
        '''
        Subsample the timeseries to get uncorrelated samples. This function also sets up the variables 
        needed for pymbar.MBAR object and pymbar.FES object.

        Parameters
        ----------
        error : bool
            Calculate error. If False, we do not need to subsample timeseries, default=True
        plot : bool
            Plot the sampling distributions. If both `plot` and `error` are True, then plot the uncorrelated samples, default=False
        
        Returns
        -------
        u_kn : array-like
            u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k, 
            reshaped for uncorrelated samples
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k, shaped properly
        N_k : array-like
            Number of samples frum umbrella k, reshaped for uncorrelated samples
        d_kn : array-like
            d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k, reshaped for uncorrelated samples

        '''

        from pymbar import timeseries
        
        # Step 1a: Setting up
        K = len(self.colvars)                       # number of umbrellas
        N_max = self.colvars[0].time.shape[0]       # number of data points in each timeseries of coordination number
        N_k, g_k = np.zeros(K, int), np.zeros(K)    # number of samples and statistical inefficiency of different simulations
        d_kn = np.zeros([K, N_max])                 # d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k
        u_kn = np.zeros([K, N_max])                 # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
        self.uncorrelated_samples = []              # Uncorrelated samples of different simulations
        self.uncorrelated_indices = []
        ion_restraint = (self.colvars[0].data.shape[1] == 9) # determine if there is an ion coordination restraint

        # Step 1b: Read in and subsample the timeseries
        for k in range(K):
            # if using 2 restraints, calculate the potential from the ion coordination restraint
            if ion_restraint:
                u_kn[k] = self.beta * 10000/2 * self.colvars[k].ion_coordination_number**2 # KAPPA = 10,0000 and centered at CN = 0

            d_kn[k] = self.colvars[k].coordination_number
            N_k[k] = len(d_kn[k])
            d_temp = d_kn[k, 0:N_k[k]]
            if error:
                g_k[k] = timeseries.statistical_inefficiency(d_temp)     
                print(f"Statistical inefficiency of simulation {k}: {g_k[k]:.3f}")
                indices = timeseries.subsample_correlated_data(d_temp, g=g_k[k]) # indices of the uncorrelated samples
            else:
                indices = np.arange(len(self.colvars[k].coordination_number))
            
            # Update u_kn and d_kn with uncorrelated samples if calculating error
            N_k[k] = len(indices)    # At this point, N_k contains the number of uncorrelated samples for each state k                
            u_kn[k, 0:N_k[k]] = u_kn[k, indices]
            d_kn[k, 0:N_k[k]] = d_kn[k, indices]
            if error:
                self.uncorrelated_samples.append(d_kn[k, indices])
                self.uncorrelated_indices.append(indices)

        N_max = np.max(N_k) # shorten the array size
        u_kln = np.zeros([K, K, N_max]) # u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k

        if error and plot:
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.set_xlabel('Coordination number, continuous')
            ax.set_ylabel('Counts')
            for k in range(K):
                ax.hist(self.uncorrelated_samples[k], bins=50, alpha=0.5)

            ax.set_xlim(d_kn.min()-0.5, d_kn.max()+0.5)
            fig.savefig('uncorrelated_sampling.png')
            plt.close()

        elif plot:
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.set_xlabel('Coordination number, continuous')
            ax.set_ylabel('Counts')
            for k in range(K):
                ax.hist(self.colvars[k].coordination_number, bins=50, alpha=0.5)

            ax.set_xlim(d_kn.min()-0.5, d_kn.max()+0.5)
            fig.savefig('sampling.png')
            plt.close()

        return u_kn, u_kln, N_k, d_kn
    

    def _evaluate_reduced_energies(self, CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA=100):
        '''
        Create the u_kln matrix of reduced energies from the umbrella simulations.

        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers
        u_kn : array-like
            u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k
        N_k : array-like
            Number of samples frum umbrella k
        d_kn : array-like
            d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k
        KAPPA : float, array-like
            Strength of the harmonic potential (kJ/mol/CN^2), default=100

        Returns
        -------
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k, calculated from
            u_kn and the harmonic restraint, shaped properly

        '''


        K = len(self.colvars)                       # number of umbrellas
        beta_k = np.ones(K) * self.beta             # inverse temperature of simulations (in 1/(kJ/mol)) 

        # spring constant (in kJ/mol/CN^2) for different simulations
        # coerce into a np.array
        if isinstance(KAPPA, (float, int)): 
            K_k = np.ones(K)*KAPPA                   
        elif not isinstance(KAPPA, np.ndarray):
            K_k = np.array(KAPPA)
        else:
            K_k = KAPPA

        for k in range(K):
            for n in range(N_k[k]):
                # Compute the distance from the center of simulation k in coordination number space
                dd = d_kn[k,n] - CN0_k

                # Compute energy of snapshot n from simulation k in umbrella potential l
                u_kln[k,:,n] = u_kn[k,n] + beta_k[k] * (K_k / 2) * dd ** 2

        return u_kln


    def _fit_spline(self, bins=None, fes=None):
        '''
        Fit a scipy.interpolate.UnivariateSpline to the FES. Uses a quartic spline (k=4) 
        and interpolates with all points, no smoothing (s=0)

        Parameters
        ----------
        bins : np.array
            Bins of the FES to fit to spline, default=None means use self.bin_centers
        fes : np.array
            FES to fit to spline, default=None means use self.fes

        Returns
        -------
        f : the interpolated spline

        '''

        from scipy.interpolate import UnivariateSpline

        if bins is None:
            bins = self.bin_centers
        if fes is None:
            fes = self.fes

        f = UnivariateSpline(bins, fes, k=4, s=0)
        
        return f
    

    def _coordination_number_per_frame(self, frame_idx, biased_ion, radius, **kwargs):
        '''
        Calculate the discrete total coordination number as a function of time for biased ion.

        Parameters
        ----------
        frame_idx : int
            Index of the trajectory frame
        biased_ion : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)

        Returns
        -------
        coordination_number : int
            Discrete coordination number around the biased ion
        
        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        # calculate distances and compare to hydration shell cutoff
        d = distances.distance_array(biased_ion, self.universe.select_atoms('not type HW* MW') - biased_ion, 
                                     box=self.universe.dimensions, **kwargs)
        coordination_number = (d <= radius).sum()

        return coordination_number
    

    def _water_dipole_per_frame(self, frame_idx, biased_ion, radius, n_max=12):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        biased_ion : MDAnalysis.Atom
            Ion to calculate the distribution for, the ion whose coordination shell has been biased.
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only
        n_max : int
            Maximum number of coordinated waters, if discrete coordination numbers have been calculated, will use
            the max of `self.coordination_numbers`, default=12

        Returns
        -------
        angles : np.array
            Angles for the waters coordinated on this frame, size n_max

        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        my_atoms = self.universe.select_atoms(f'sphzone {radius} index {biased_ion.index}') - biased_ion
        my_waters = my_atoms & self.waters # intersection operator to get the OW from my_atoms

        angles = np.empty(n_max)
        angles[:] = np.nan
        for j,ow in enumerate(my_waters):

            dist = biased_ion.position - ow.position

            # if the water is on the other side of the box, move it back
            for d in range(3):
                v = np.array([0,0,0])
                v[d] = 1
                if dist[d] >= self.universe.dimensions[d]/2:
                    ow.residue.atoms.translate(v*self.universe.dimensions[d])
                elif dist[d] <= -self.universe.dimensions[d]/2:
                    ow.residue.atoms.translate(-v*self.universe.dimensions[d])

            # calculate and save angles
            pos = ow.position
            bonded_Hs = ow.bonded_atoms
            tmp_pt = bonded_Hs.positions.mean(axis=0)

            v1 = biased_ion.position - pos
            v2 = pos - tmp_pt
            angles[j] = get_angle(v1, v2)*180/np.pi

        return angles


    def _polyhedron_size_per_frame(self, frame_idx, biased_ion, r0=3.15, for_visualization=False):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        biased_ion : str, MDAnalysis.Atom
            Biased ion in the simulation to calculate polyhedrons for
        r0 : float
            Hydration shell cutoff for the biased ion in Angstroms, default=3.15
        for_visualization : bool
            Whether to use this function to output points for visualization, default=False

        Returns
        -------
        area, volume : float
            Volume and maximum cross-sectional area for the polyhedron
        
        '''
    
        # make biased_ion into MDAnalysis Atom
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)[0]
        else:
            ion = biased_ion
        
        # initialize the frame
        self.universe.trajectory[frame_idx]

        # Unwrap the shell and generate points on atomic spheres
        shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
        pos = self._unwrap_shell(ion, r0)
        shell.positions = pos
        pos = self._points_on_atomic_radius(shell, n_points=200)
        center = ion.position

        # Create the polyhedron with a ConvexHull and save volume
        hull = ConvexHull(pos)
        volume = hull.volume

        # Get the major axis (first principal component)
        pca = PCA(n_components=3).fit(pos[hull.vertices])

        # Find all the edges of the convex hull
        edges = []
        for simplex in hull.simplices:
            for s in range(len(simplex)):
                edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                edges.append(edge)

        # Create a line through the polyhedron along the principal component
        d = distances.distance_array(shell, shell, box=self.universe.dimensions)
        t_values = np.linspace(-d.max()/2, d.max()/2, 100)
        center_line = np.array([center + t*pca.components_[0,:] for t in t_values])

        # Find the maximum cross-sectional area along the line through polyhedron
        area = 0
        for pt in center_line:

            # Find the plane normal to the principal component
            A, B, C, D = create_plane_from_point_and_normal(pt, pca.components_[0,:])

            # Find the intersection points of the hull edges with the slicing plane
            intersection_points = []
            for edge in edges:
                p1 = pos[edge[0]]
                p2 = pos[edge[1]]
                intersection_point = line_plane_intersection(p1, p2, A, B, C, D)
                if intersection_point is not None:
                    intersection_points.append(intersection_point)

            # If a slicing plane exists and its area is larger than any other, save
            if len(intersection_points) > 0:
                intersection_points = np.array(intersection_points)
                projected_points, rot_mat, mean_point = project_to_plane(intersection_points)
                intersection_hull = ConvexHull(projected_points)

                if intersection_hull.volume > area:
                    saved_points = (pt, intersection_points, projected_points, mean_point)
                    area = intersection_hull.volume

        if for_visualization:
            return area, volume, saved_points
        else:
            return area, volume
    

    def _unwrap_shell(self, ion, r0):
        '''
        Unwrap the hydration shell, so all coordinated groups are on the same side of the box as ion.

        Parameters
        ----------
        ion : MDAnalysis.Atom
            Ion whose shell to unwrap
        r0 : float
            Hydration shell radius for the ion

        Returns
        -------
        positions : np.ndarray
            Unwrapped coordinated for the atoms in the shell

        '''

        dims = self.universe.dimensions
        shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
        dist = ion.position - shell.positions

        correction = np.where(np.abs(dist) > dims[:3]/2, # check if beyond half-box distance
                              np.sign(dist) * dims[:3],  # add or subtract box size based on sign of dist
                              0)                         # otherwise, do not move

        return shell.positions + correction


    def _points_on_atomic_radius(self, shell, n_points=200):
        '''
        Generate points on the "surface" of the atoms, so we have points that encompass the volume of the atoms.

        Parameters
        ----------
        shell : MDAnalysis.AtomGroup
            Hydration shell with all atoms to generate points

        Returns
        -------
        positions : np.ndarray
            Points on the "surface" of the atoms
        '''

        # get all radii
        radii = np.array([self.vdW_radii[atom.element] for atom in shell])
            
        # randomly sample theta and phi angles
        rng = np.random.default_rng()
        theta = np.arccos(rng.uniform(-1,1, (len(shell),n_points)))
        phi = rng.uniform(0,2*np.pi, (len(shell),n_points))

        # convert to Cartesian coordinates
        x = radii[:,None] * np.sin(theta) * np.cos(phi) + shell.positions[:,0,None]
        y = radii[:,None] * np.sin(theta) * np.sin(phi) + shell.positions[:,1,None]
        z = radii[:,None] * np.cos(theta) + shell.positions[:,2,None]

        positions = np.stack((x,y,z), axis=-1)
        return positions.reshape(-1,3)


    def _get_spline_minima(self):
        '''
        Get the spline minima by solving df/dx = 0. Root-finding only works for cubic splines
        for FITPACK (the backend software), so the spline must be a 4th degree spline

        Returns
        -------
        minima_locs : np.array
            Locations of the minima
        minima_vals : np.array
            Values of the minima

        '''

        minima_locs = self.spline.derivative(1).roots()
        minima_vals = self.spline(minima_locs)

        return minima_locs, minima_vals


    # from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    def _rotation_matrix_from_vectors(self,vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
