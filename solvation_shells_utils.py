# Hold functions and classes to analyze solvation shells

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import MDAnalysis as mda

import subprocess
from textwrap import dedent
from glob import glob


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
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=nasc4134@colorado.edu

    module load gcc
    module load openmpi

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


def unitize(v):
    '''Create a unit vector from vector v'''
    return v / np.linalg.norm(v)


def get_angle(v1, v2):
    '''Get the angle between two vectors v1 and v2'''
    
    v1 = unitize(v1)
    v2 = unitize(v2)

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


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

        
    def __repr__(self):
        return f'EquilibriumAnalysis object with {len(self.waters)} waters, {len(self.cations)} cations, and {len(self.anions)} anions over {self.n_frames} frames'
    

    def _find_peaks_wrapper(self, bins, data, **kwargs):
        '''Wrapper for scipy.signal.find_peaks to use with SolvationAnalysis to find cutoff'''
        
        from scipy.signal import find_peaks

        peaks, _  = find_peaks(-data, **kwargs)
        radii = bins[peaks[0]]
        return radii
    

    def initialize_Solutes(self, step=1):
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
                                           kernel_kwargs={'distance':5})
        self.solute_ai = Solute.from_atoms(self.anions, {'water' : self.waters, 'coion' : self.cations}, 
                                  solute_name='Anion', rdf_kernel=self._find_peaks_wrapper, 
                                  kernel_kwargs={'distance':5})

        self.solute_ci.run(step=step)
        self.solute_ai.run(step=step)

        print(f"\nHydration shell cutoff for cation-water = {self.solute_ci.radii['water']:.6f}")
        print(f"Hydration shell cutoff for anion-water = {self.solute_ai.radii['water']:.6f}")
    

    def generate_rdfs(self, bin_width=0.05, range=(0,20), **kwargs):
        '''
        Calculate radial distributions for the solution. This method calculates the RDFs for cation-water,
        anion-water, water-water, and cation-anion using MDAnalysis InterRDF. It saves the data in a 
        dictionary attribute `rdfs` with keys 'ci-w', 'ai-w', 'w-w', and 'ci-ai'.

        Parameters
        ----------
        bin_width : float
            Width of the bins for the RDFs, default=0.05
        range : array-like
            Range over which to calculate the RDF

        Returns
        -------
        rdfs : dict
            Dictionary with all the results from InterRDF
        
        '''

        from MDAnalysis.analysis import rdf

        nbins = int(range[1] - range[0]) / bin_width
        self.rdfs = {}

        ci_w = rdf.InterRDF(self.cations, self.waters, nbins=nbins, range=range, norm='rdf', **kwargs)
        ci_w.run(**kwargs)
        self.rdfs['ci-w'] = ci_w.results

        ai_w = rdf.InterRDF(self.anions, self.waters, nbins=nbins, range=range, norm='rdf', **kwargs)
        ai_w.run(**kwargs)
        self.rdfs['ai-w'] = ai_w.results

        w_w = rdf.InterRDF(self.waters, self.waters, nbins=nbins, range=range, norm='rdf', **kwargs)
        w_w.run(**kwargs)
        self.rdfs['w-w'] = w_w.results

        ci_ai = rdf.InterRDF(self.cations, self.anions, nbins=nbins, range=range, norm='rdf', **kwargs)
        ci_ai.run(**kwargs)
        self.rdfs['ci-ai'] = ci_ai.results

        return self.rdfs
    

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
        self.data = pd.DataFrame(tmp[start:stop:by,:], columns=['time', 'n', 't', 'r.bias', 'r.force2'])
        self.time = self.data.time.to_numpy()
        self.coordination_number = self.data.t.to_numpy()
        self.bias = self.data['r.bias'].to_numpy()
        self.force = self.data['r.force2'].to_numpy()


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

    def __init__(self, n_umbrellas, COLVAR_file='COLVAR_', start=10, stop=-1, by=100, T=300):
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

        '''

        # initialize some variables
        self.kB = 1.380649 * 10**-23 * 10**-3 * 6.022*10**23 # Boltzmann (kJ / K)
        self.kT = self.kB*T
        self.beta = 1/self.kT
        self._fes = None

        # read in collective variable files
        self.colvars = []

        for i in range(n_umbrellas):
            filename = f'{COLVAR_file}{i}'
            self.colvars.append(UmbrellaSim(filename, start=start, stop=stop, by=by))

    
    def __repr__(self):
        return f'UmbrellaAnalysis object with {len(self.colvars)} simulations'

        
    def calculate_FES(self, CN0_k, KAPPA=100, n_bootstraps=0, nbins=200, d_min=2, d_max=8, bw=0.02, mintozero=True, filename=None):
        '''
        Calculate the free energy surface with pymbar
        
        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers
        KAPPA : float
            Strength of the harmonic potential (kJ/mol/nm^2), default=100
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

        import pymbar

        # Step 1: Subsample timeseries
        u_kn, u_kln, N_k, d_kn = self._subsample_timeseries()
        
        # Step 2: Bin the data
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(d_min, d_max, nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        # Step 3: Evaluate reduced energies in all umbrellas
        u_kln = self._evaluate_reduced_energies(CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA)

        # Step 4: Compute and output the FES
        fes = pymbar.FES(u_kln, N_k, verbose=False)
        kde_params = {'bandwidth' : bw}
        d_n = pymbar.utils.kn_to_n(d_kn, N_k=N_k)
        if n_bootstraps == 0:
            # fes.generate_fes(u_kn, d_n, fes_type='kde', kde_parameters=kde_params)
            fes.generate_fes(u_kn, d_n, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='analytical')
        else:
            fes.generate_fes(u_kn, d_n, fes_type='kde', kde_parameters=kde_params, n_bootstraps=n_bootstraps)
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='bootstrap')

        if mintozero:
            results['f_i'] = results['f_i'] - results['f_i'].min()

        # Step 5: Save FES information in the object
        self.umbrella_centers = CN0_k
        self._u_kln = u_kln
        self._N_k = N_k
        self._fes = fes                     # underlying pymbar.FES object
        self._results = results             # underlying results object
        self.bin_centers = bin_center_i
        self.fes = results['f_i']*self.kT
        # if n_bootstraps > 0:
        self.error = results['df_i']*self.kT

        if filename is not None:
            if n_bootstraps == 0:
                np.savetxt(filename, np.vstack([self.bin_centers, self.fes]).T, header='coordination number, free energy (kJ/mol)')
            else:
                np.savetxt(filename, np.vstack([self.bin_centers, self.fes, self.error]).T, header='coordination number, free energy (kJ/mol), error (kJ/mol)')

        return self.bin_centers, self.fes
    

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


    def _subsample_timeseries(self):
        '''
        Subsample the timeseries to get uncorrelated samples. This function also sets up the variables 
        needed for pymbar.MBAR object and pymbar.FES object.
        
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

        # Step 1b: Read in and subsample the timeseries
        for k in range(K):
            d_kn[k] = self.colvars[k].coordination_number
            N_k[k] = len(d_kn[k])
            d_temp = d_kn[k, 0:N_k[k]]
            g_k[k] = timeseries.statistical_inefficiency(d_temp)     
            print(f"Statistical inefficiency of simulation {k}: {g_k[k]:.3f}")
            indices = timeseries.subsample_correlated_data(d_temp, g=g_k[k]) # indices of the uncorrelated samples
            
            # Update u_kn and d_kn with uncorrelated samples
            N_k[k] = len(indices)    # At this point, N_k contains the number of uncorrelated samples for each state k                
            u_kn[k, 0:N_k[k]] = u_kn[k, indices]
            d_kn[k, 0:N_k[k]] = d_kn[k, indices]
            self.uncorrelated_samples.append(d_kn[k, indices])

        N_max = np.max(N_k) # shorten the array size
        u_kln = np.zeros([K, K, N_max]) # u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k

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
        KAPPA : float
            Strength of the harmonic potential (kJ/mol/nm^2), default=100

        Returns
        -------
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k, calculated from
            u_kn and the harmonic restraint, shaped properly

        '''

        K = len(self.colvars)                       # number of umbrellas
        beta_k = np.ones(K) * self.beta             # inverse temperature of simulations (in 1/(kJ/mol)) 
        K_k = np.ones(K)*KAPPA                      # spring constant (in kJ/mol/nm**2) for different simulations

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



# TEMPORARY
# define some functions for analysis
def unitize(v):
    '''Create a unit vector from vector v'''
    return v / np.linalg.norm(v)


def get_angle(v1, v2):
    '''Get the angle between two vectors v1 and v2'''
    
    v1 = unitize(v1)
    v2 = unitize(v2)

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def get_distributions(solute_Na, ion, shell_OW, shell):
    '''
    Get the water angle and water orientation distributions for a given shell
    
    shell : dict, the shell structure to look at, e.g. {water : 5, ion : 0}
    '''

    df1 = solute_Na.speciation.get_shells(shell)
    frames = np.array(df1.droplevel(1).index.to_list())

    angles = []
    orientations = []
    for i, ts in enumerate(u.trajectory[frames]):

        # calculate angles between ref-ion-OW
        tmp_point = ion.positions[0] + np.array([0,0,1])
        v1 = tmp_point - ion.positions[0]
        for pos in shell_OW.positions:
            v2 = pos - ion.positions[0]
            ang = get_angle(v1, v2)*180/np.pi
            angles.append(ang)

        # calculate angles between avg_HW-OW-ion
        for O in shell_OW:
            pos = O.position
            bonded_Hs = O.bonded_atoms
            tmp_point = bonded_Hs.positions.mean(axis=0)
            
            v1 = ion.positions[0] - pos
            v2 = pos - tmp_point
            ang = get_angle(v1, v2)*180/np.pi
            orientations.append(ang)

    return np.array(angles), np.array(orientations)