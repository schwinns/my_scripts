# Class to convert MD simulations to 3D trajectories for use in HDP-AR-HMM

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import MDAnalysis.transformations as trans

import multiprocessing


def plot_colored_line(x, y, c, ax, **lc_kwargs): # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    '''
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.

    '''
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


class IonDynamics:

    def __init__(self, topology, trajectory, transform=True, njobs=1,
                 polymer_selection='resname PA*', water_selection='resname SOL',
                 cation_selection='resname NA', anion_selection='resname CL'):
        '''
        Class to handle ion trajectories from MD simulations of ions within a polymer membrane

        Parameters
        ----------
        topology : str
            Topology file for the MDAnalysis Universe
        trajectory : str, list of str
            Trajectory file(s) for the MDAnalysis Universe
        transform : bool, optional
            Whether to perform the unwrapping transformations. Only set to False if the transforms
            have already been applied to `trajectory`. Default = True
        njobs : int, optional
            Number of CPUs for the NoJump transformation. Default = 1. -1 means all available.
        polymer_selection : str, optional
            MDAnalysis selection language for all polymer atoms. Default = 'resname PA*'
        water_selection : str, optional
            MDAnalysis selection language for all water atoms. Default = 'resname SOL'
        cation_selection : str, optional
            MDAnalysis selection language for cations. Default = 'resname NA'
        anion_selection : str, optional
            MDAnalysis selection language for anions. Default = 'resname CL'
        
        '''
        
        # initialize Universe and AtomGroups
        self.universe = mda.Universe(topology, trajectory)
        self.polymer = self.universe.select_atoms(polymer_selection)
        self.waters = self.universe.select_atoms(water_selection)
        self.cations = self.universe.select_atoms(cation_selection)
        self.anions = self.universe.select_atoms(anion_selection)
        self.ions = self.cations + self.anions

        for ag in (self.polymer, self.waters, self.cations, self.anions):
            print(f'\t{ag[0].resname} group has {len(ag)} atoms')

        # initialize some useful variables
        self.n_frames = self.universe.trajectory.n_frames
        self.dt = self.universe.trajectory[1].time - self.universe.trajectory[0].time # timestep (ps)
        self.time = np.arange(self.universe.trajectory[0].time, self.universe.trajectory[-1].time+self.dt, self.dt)
        self.state_sequence = None

        # set up the trajectory transformations
        if njobs == -1:
            self.njobs = multiprocessing.cpu_count()
        else:
            self.njobs = njobs

        if transform:
            self.universe.trajectory[0]
            workflow = [
                trans.unwrap(self.universe.atoms),
                trans.center_in_box(self.polymer, center='mass'),
                trans.wrap(self.universe.atoms),
                trans.nojump.NoJump(max_threads=self.njobs)
                ]
            self.universe.trajectory.add_transformations(*workflow)

    
    def __str__(self):
        return f'<IonDynamics object with {self.cations.n_atoms} cations and {self.anions.n_atoms} anions over {self.dt*self.n_frames/1000} ns>'


    def __repr__(self):
        return self.__str__()
        

    def extract_ion_trajectories(self, ions, filename='ion_trajectories.csv'):
        '''
        Extract the ion trajectories from the MD simulation and save as a csv file

        Parameters
        ----------
        ions : MDAnalysis AtomGroup
            AtomGroup with the ions to extract
        filename : str
            Name of the file to save trajectories to. Should be a csv. Default = 'ion_trajectories.csv'

        Returns
        -------
        ion_trajectories : pd.DataFrame
            Pandas DataFrame with ion trajectories. Columns are the ion index in the simulation, 
            the ion type, and the x,y,z coordinates. Also stored as an attribute.

        '''

        # first, get the trajectories as np.ndarray
        traj = self._trajectory_to_numpy(ions)

        # pick only ions that are within the middle 50% of the polymer throughout the simulation
        self.membrane_ions = {}
        for a, atom in enumerate(ions):
            below_ub = (traj[:,a,2] >= self.membrane_bounds[:,1]).all()
            above_lb = (traj[:,a,2] <= self.membrane_bounds[:,3]).all()
            if below_ub and above_lb:
                self.membrane_ions[atom.index] = traj[:,a,:]

        # save as a pd.DataFrame and save as csv
        data = [(ion_id, *coord) for ion_id, coords in self.membrane_ions.items() for coord in coords]
        self.ion_trajectories = pd.DataFrame(data, columns=['ion_index', 'x', 'y', 'z'])
        
        ion_types = []
        for idx in self.membrane_ions:
            ion = self.universe.select_atoms(f'index {idx}')[0]
            [ion_types.append(t) for t in [ion.type]*self.n_frames]

        self.ion_trajectories['ion_type'] = ion_types
        self.ion_trajectories.to_csv(filename, index=False)

        return self.ion_trajectories


    def write_xtc(self, filename, atom_group=None, frames=None):
        '''
        Write an xtc file with MD trajectory

        Parameters
        ----------
        filename : str
            Name of the output xtc file
        atom_group : MDAnalysis AtomGroup, optional
            Atom group to write to xtc. Default = None, means use all atoms.
        frames : array-like, optional
            Frames to write to the xtc file. Default = None, means write all frames.

        '''

        if atom_group is None:
            atom_group = self.universe.atoms

        if frames is None:
            frames = np.arange(self.n_frames)

        with mda.Writer(filename, atom_group.n_atoms) as w:
            for ts in tqdm(self.universe.trajectory[frames]):
                w.write(atom_group)


    def write_gro(self, filename, atom_group=None):
        '''
        Write a gro file with the coordinates of the atom group

        Parameters
        ----------
        filename : str
            Name of the output gro file
        atom_group : MDAnalysis AtomGroup, optional
            Atom group to write to gro. Default = None, means use all atoms.

        '''

        if atom_group is None:
            atom_group = self.universe.atoms

        atom_group.write(filename)


    def load_state_sequence(self, state_sequence, molecules):
        '''
        Load the state sequence that was output from the HDP-AR-HMM. Saved as an attribute `self.state_sequence`
        in the format {molecule_index: state_sequence}. Creates attributes `self.unique_states` (dict) that contains 
        the unique states for each molecule, `self.found_states` (np.ndarray) that contains all the states identified, 
        and `self.n_states` (int) with the number of `found_states`.

        Parameters
        ----------
        state_sequence : np.ndarray
            State sequence, shape (n_frames,n_molecules). This can be directly loaded from hdphmm.hdphmm.InifiniteHMM.z.T,
            i.e. the transpose of the state matrix.
        molecules : array-like or MDAnalysis AtomGroup
            Molecules to which the state sequence corresponds. This should be the same as the number of columns in
            `state_sequence`. This can be a list of molecule indices or an MDAnalysis AtomGroup with the same number
            of atoms as the number of columns in `state_sequence`. If a list of indices, the indices should be
            the same as the indices in the MDAnalysis Universe.

        Returns
        -------
        state_sequence : dict
            State sequences for each molecule. The keys are the indices of the molecules in the MDAnalysis Universe 
            and the values are the state sequences for each molecule. The shape of each state sequence is (n_frames,).
        
        '''

        if len(molecules) != state_sequence.shape[1]:
            raise ValueError(f'Number of molecules ({len(molecules)}) does not match the number of columns in state_sequence ({state_sequence.shape[1]})')

        self._state_sequnce_array = state_sequence
        self.state_sequence = {}
        self.unique_states = {}

        for i,molecule in enumerate(molecules):
            if isinstance(molecule, mda.AtomGroup):
                molecule = molecule.index
            elif isinstance(molecule, (int, np.integer)):
                pass
            else:
                raise ValueError(f'molecules must be a list of indices or an MDAnalysis AtomGroup but received {type(molecule)}')

            self.state_sequence[molecule] = state_sequence[:,i]
            self.unique_states[molecule] = np.unique(state_sequence[:,i])

        self.found_states = np.unique(state_sequence)
        self.n_states = len(self.found_states)
        return self.state_sequence


    def density_profile(self, frame, atom_groups, bins=None, bin_width=0.5, dim='z', method='atom'):
        '''
        Calculate the partial density across the box for a given atom group
        
        Parameters
        ----------
        frame : int
            Frame of the trajectory
        atom_groups : list of MDAnalysis AtomGroups
            List of AtomGroups to calculate the density profile for
        bins : np.ndarray, optional
            Bins to use for the density profile. Default = None, which means bins will be created
            based on the bin_width
        bin_width : float, optional
            Width of the bins for the density profile. Default = 0.5 Angstroms
        dim : str, optional
            Dimension to calculate the density profile. Default = 'z'
        method : str, optional
            Method to calculate the density profile. Options are 'atom', 'molecule', 'mass', and 'charge'. 
            Default = 'atom' gives the number of atoms in each bin. 'molecule' gives the number of molecules in
            each bin. 'mass' gives the mass density in each bin. 'charge' gives the charge density in each bin.

        Returns
        -------
        bin_centers : np.ndarray
            Array of bin centers, shape (n_bins,)
        counts : np.ndarray
            Array of counts per bin, shape (n_bins, n_atom_groups)

        '''

        self.universe.trajectory[frame]

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions

        if bins is None:
            n_bins = int(box[d] / bin_width)
            bins = np.linspace(0, box[d], num=n_bins)
        else:
            n_bins = len(bins)

        counts = np.zeros((n_bins-1, len(atom_groups)))
        
        for b in range(n_bins-1):
            lb = bins[b]
            ub = bins[b+1]
            for ag,atom_group in enumerate(atom_groups):
                bin_atoms = atom_group.select_atoms(f'(prop {dim} >= {lb}) and (prop {dim} <= {ub})')
                if method == 'atom':
                    counts[b,ag] += len(bin_atoms)
                elif method == 'molecule': 
                    counts[b,ag] += bin_atoms.n_residues
                elif method == 'mass':
                    dV = box[0] * box[1] * (ub-lb) * (10**-8)**3 # volume in cm^3
                    mass = bin_atoms.masses.sum() / 6.022 / 10**23
                    counts[b,ag] += mass / dV
                elif method == 'charge':
                    counts[b,ag] += bin_atoms.charges.sum()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, counts


    def plot_membrane(self, frame=None, cation_color='blue', anion_color='limegreen', 
                      ydim='x', show_ions=True, show_zones=False, ax=None):
        '''
        Plot the membrane in the ydim-z plane
        
        Parameters
        ----------
        frame : int, None, optional
            Frame of the trajectory to plot. If None, whatever frame the trajectory reader is at will be used.
            Default = None
        cation_color : str, optional
            Color to plot cations. Default = 'blue'
        anion_color : str, optional
            Color to plot anions. Default = 'limegreen'
        ydim : str, optional
            Which dimension to plot on the y-axis. Options are 'x' and 'y'. Default = 'x'
        show_ions : bool, optional
            Whether to plot ions. Default = True
        show_zones : bool, optional
            Whether to plot geometric polymer zones. Default = False
        ax : matplotlib axis, optional
            Axis to plot on. If None, a new figure will be created. Default = None

        Returns
        -------
        ax : matplotlib axis
            Axis with the membrane plotted

        '''

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,4))
        
        if frame is not None:
            self.universe.trajectory[frame]

        if ydim == 'x':
            d = 0
        elif ydim == 'y':
            d = 1

        # plot the polymer with a kernel density estimate
        polymer = self._coordinates_as_dataframe(self.polymer)
        sns.kdeplot(polymer, x='z', y=ydim, cmap='Greys', ax=ax, fill=True, 
                    levels=10, thresh=0.05, cut=0, weights=self.polymer.masses)

        # plot scatter points of ions
        if show_ions:
            ax.scatter(self.cations.positions[:,2], self.cations.positions[:,d], c=cation_color)
            ax.scatter(self.anions.positions[:,2], self.anions.positions[:,d], c=anion_color)

        # plot lines for geometric polymer regions
        if show_zones:
            for z in self.polymer_zones:
                ax.axvline(z, ls='dashed', c='r')


        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0,self.universe.dimensions[2])
        ax.set_ylim(0,self.universe.dimensions[d])

        return ax
    

    def rdfs_by_state(self, ion, bin_width=0.05, range=(0,20)):
        '''
        Calculate radial distribution functions for each state for `ion`. This method calculates the RDFs for
        cation-water, cation-cation, cation-anion, cation-COOH, cation-COO-, cation-amideO, and cation-NH2
        using MDAnalysis InterRDF. It saves the data in a dictionary attribute `rdfs` with keys for each state. 
        Each state then corresponds to a dictionary of with the RDF types above. 
        
        Parameters
        ----------
        ion : int, MDAnalysis Atom
            Either index of the ion or an MDAnalysis Atom of the ion
        bin_width : float
            Width of the bins for the RDFs, default=0.05
        range : array-like
            Range over which to calculate the RDF, default=(0,20)

        Returns
        -------
        rdfs : dict
            Dictionary of dictionaries with the results for each state. The keys are the states and the values
            are dictionaries with the RDF types. The keys of the inner dictionaries are the RDF types and the values
            are the RDF results.
        
        '''

        # check for state sequence and proper types
        if self.state_sequence is None:
            raise ValueError('No state sequence loaded. Please load a state sequence first.')

        if isinstance(ion, mda.Atom):
                ion = ion.index
        elif isinstance(molecule, (int, np.integer)):
            pass
        else:
            raise ValueError(f'ion must be an index or an MDAnalysis Atom but received {type(ion)}')
        
        # initialize bins and selections
        nbins = int((range[1] - range[0]) / bin_width)

        xlink = self.universe.select_atoms(f'(type c) and (bonded type n)')
        cooh_C = self.universe.select_atoms(f'(type c) and (bonded type oh)')
        cooh_OH = self.universe.select_atoms(f'(type oh)')
        amideO = self.universe.select_atoms(f'(type o) and (bonded group xlink)', xlink=xlink)
        coo = self.universe.select_atoms(f'(type o) and (not bonded group cooh_C) and (not bonded group xlink)', cooh_C=cooh_C, xlink=xlink)
        nh2 = self.universe.select_atoms(f'(type nh)')
        waters = self.waters.select_atoms(f'(type OW)')

        # prepare the selections and their labels
        selections = [waters, self.cations, self.anions, cooh_OH, 
                      coo, amideO, nh2]
        rdf_types = ['cation-water', 'cation-cation', 'cation-anion', 'cation-COOH', 
                     'cation-COO-', 'cation-amideO', 'cation-NH2']
        
        # generate the RDF data
        rdfs = {}
        for state in self.found_states:
            idx = np.where(self.state_sequence[ion] == state)[0]
            print(f'Calculating RDFs for state {state} ({len(idx)} frames)')
            rdfs[state] = {}
            for l,label in enumerate(rdf_types):
                rdf = InterRDF(self.universe.select_atoms(f'index {ion}'), 
                               selections[l], 
                               range=range, norm='rdf', verbose=True)
                rdf.run(frames=idx)
                rdfs[state][label] = rdf.results


        return self.rdfs

    
    def plot_xyz_trajectories(self, x, y, z, c=None, c_label='time (ns)', ax=None, **lc_kwargs):
        '''
        Plot x, y, and z trajectories as a time series colored by a separate variable c
        
        Parameters
        ----------
        x : array-like
            x-coordinate of trajectory
        y : array-like
            y-coordinate of trajectory
        z : array-like
            z-coordinate of trajectory
        c : array-like, optional
            The color values, which should be the same size as x, y, and z. Default = None means use time.
        c_label : str, optional
            The label for the colorbar. Default = 'time (ns)'
        ax : matplotlib axis, optional
            Axis to plot on. If None, a new figure will be created. Default = None
        **lc_kwargs
            Any additional arguments to pass to matplotlib.collections.LineCollection
            constructor. This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.

        Returns
        -------
        ax : matplotlib axis
            Axis with the membrane plotted

        '''

        if ax is None:
            fig, ax = plt.subplots(3,1, figsize=(16,6), sharex=True)

        if c is None:
            c = self.time / 1000

        traj = [x,y,z]
        for d,dim in enumerate(['x','y','z']):
            line = plot_colored_line(self.time/1000, traj[d], c, ax[d], linewidth=2, **lc_kwargs)
            ax[d].set_ylabel(f'{dim} coordinate ($\AA$)')
            ax[d].set_ylim(traj[d].min(), traj[d].max())

        fig.colorbar(line, ax=ax[:], location='right', label=c_label, fraction=0.046, pad=0.04)
        ax[2].set_xlabel('time (ns)')
        ax[2].set_xlim(0, (self.time/1000).max())

        return ax
    

    def _trajectory_to_numpy(self, atom_group):
        '''
        Convert the trajectories for atoms in a given atom group to a np.ndarray. Also, during each
        iteration, saves the polymer zones for each frame to `self.membrane_bounds`, which has shape
        (n_frames, 5)

        Parameters
        ----------
        atom_group : MDAnalysis AtomGroup
            AtomGroup to convert to a DataFrame
        
        Returns
        -------
        traj : np.ndarray
            Array of shape (n_frames, n_atoms, 3)
        
        '''

        traj = np.zeros((self.universe.trajectory.n_frames, atom_group.n_atoms, 3))
        self.membrane_bounds = np.zeros((self.universe.trajectory.n_frames, 5))
        for i,ts in tqdm(enumerate(self.universe.trajectory)):
            self.membrane_bounds[i,:] = self.polymer_zones
            traj[i,:,:] = atom_group.positions

        return traj
    

    def _coordinates_as_dataframe(self, atom_group, frame=None):
        '''
        Convert the coordinates of a given atom group for a single frame to a pandas DataFrame

        Parameters
        ----------
        atom_group : MDAnalysis AtomGroup
            AtomGroup to convert to a DataFrame
        frame : int, None, optional
            Frame of the trajectory. If None, whatever frame the trajectory reader is at will be used.
            Default = None
        
        Returns
        -------
        coords : pandas DataFrame
            DataFrame with columns 'x', 'y', 'z' for each atom in the Universe
        
        '''

        if frame is not None:
            self.universe.trajectory[frame]
        coords = pd.DataFrame(atom_group.positions, columns=['x', 'y', 'z'])
        return coords


    @property
    def polymer_zones(self):
        '''
        Get the polymer zones based on simple percentage of z-coordinate

        Returns
        -------
        np.ndarray, shape (5,)
            Polymer zones

        '''
        return np.linspace(self.polymer.positions[:,2].min(), self.polymer.positions[:,2].max(), num=5)