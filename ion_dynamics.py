# Class to convert MD simulations to 3D trajectories for use in HDP-AR-HMM

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings

import MDAnalysis as mda
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
            print(f'{ag[0].resname} group has {len(ag)} atoms')

        # initialize some useful variables
        self.n_frames = self.universe.trajectory.n_frames
        self.dt = self.universe.trajectory[1].time - self.universe.trajectory[0].time # timestep
        self.time = np.arange(self.universe.trajectory[0].time, self.universe.trajectory[-1].time+self.dt, self.dt)

        # set up the trajectory transformations
        if njobs == -1:
            self.njobs = multiprocessing.cpu_count()
        else:
            self.njobs = njobs

        if transform:
            workflow = [
                trans.unwrap(self.universe.atoms),
                trans.center_in_box(self.polymer, center='mass'),
                trans.wrap(self.universe.atoms),
                trans.nojump.NoJump(max_threads=self.njobs)
                ]
            self.universe.trajectory.add_transformations(*workflow)


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
        membrane_ions = {}
        for a, atom in enumerate(ions):
            below_ub = (traj[:,a,2] >= self.membrane_bounds[:,1]).all()
            above_lb = (traj[:,a,2] <= self.membrane_bounds[:,3]).all()
            if below_ub and above_lb:
                membrane_ions[atom.index] = traj[:,a,:]

        # save as a pd.DataFrame and save as csv
        data = [(ion_id, *coord) for ion_id, coords in membrane_ions.items() for coord in coords]
        self.ion_trajectories = pd.DataFrame(data, columns=['ion_index', 'x', 'y', 'z'])
        
        ion_types = []
        for idx in membrane_ions:
            ion = self.universe.select_atoms(f'index {idx}')[0]
            [ion_types.append(t) for t in [ion.type]*self.n_frames]

        self.ion_trajectories['ion_type'] = ion_types
        self.ion_trajectories.to_csv(filename, index=False)

        return self.ion_trajectories


    def get_membrane_ions(self, frame):
        '''
        For a given frame, get the atoms within the "bulk" membrane, i.e. select the ions within the center 50%
        of the polymer as determined by the geometric polymer zones.

        Parameters
        ----------
        frame : int
            Frame of the trajectory

        Returns
        -------
            membrane_ions : MDAnalysis AtomGroup
                AtomGroup with the ions that are in the membrane

        '''

        self.universe.trajectory[frame]

        lb, ub = self.polymer_zones[1], self.polymer_zones[3]
        membrane_ions = self.ions.select_atoms(f'(prop z >= {lb}) and (prop z <= {ub})')
        return membrane_ions


    def write_xtc(self, filename, atom_group=None):
        '''
        Write a centered and NoJump xtc file with MD trajectory

        Parameters
        ----------
        filename : str
            Name of the output xtc file
        atom_group : MDAnalysis AtomGroup, optional
            Atom group to write to xtc. Default = None, means use all atoms.

        '''

        if atom_group is None:
            atom_group = self.universe.atoms

        with mda.Writer(filename, atom_group.n_atoms) as w:
            for ts in tqdm(self.universe.trajectory):
                w.write(atom_group)


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