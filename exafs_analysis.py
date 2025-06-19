# Class to extract clusters from MD simulations and calculate EXAFS spectra with FEFF, optimize with experimental data

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from textwrap import dedent
from time import perf_counter as time
import subprocess

import networkx as nx
from networkx.algorithms import isomorphism as iso

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from ase.data import atomic_numbers
from MDAnalysis.analysis.base import Results

from larch import xafs
from larch import Group
from ParallelMDAnalysis import ParallelAnalysisBase

# from skopt import gp_minimize
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.size'] = 16

import multiprocessing
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", SyntaxWarning)


def run(commands : list | str):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        out = subprocess.run(cmd, shell=True)

    return out


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def k2chi_axis(ion='Unspecified', xmin=1, xmax=6, ymin=-0.6, ymax=0.6, 
               xtick=0.2, ytick=0.05, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6, 4))

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$k$ (1/$\AA$)')
    ax.set_ylabel(f'$k^2 \chi(k)$ around {ion}')
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_minor_locator(MultipleLocator(xtick))
    ax.yaxis.set_minor_locator(MultipleLocator(ytick))
    return ax


def chiR_axis(ion='Unspecified', xmin=0, xmax=6, ymin=-0.6, ymax=0.6, 
               xtick=0.2, ytick=0.05, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6, 4))

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$R$ ($\AA$)')
    ax.set_ylabel(f'|$\chi(R)$| ($\AA^{-3}$) around {ion}')
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_minor_locator(MultipleLocator(xtick))
    ax.yaxis.set_minor_locator(MultipleLocator(ytick))
    return ax


def list2str(lst):
    '''Convert a list to a delimited string'''
    lst = [str(i) for i in lst]
    return ' '.join(lst)


def find_files_dat_files(base_path):
    '''Recursively find all files.dat files under base_path.'''
    return glob(os.path.join(base_path, '**', 'files.dat'), recursive=True)


def write_feff(potentials, atoms, title='FEFF8.5 input',  edge='K', s02=1.0, kmax=20, 
               print_vals=[0,0,0,0,0,3], control=[1,1,1,1,1,1], exchange=[0,0.0,0.0], 
               cfaverage=[0,1,0], scf=[6,0,30,0.2], ion=[], tdlda=0, debye=[], rpath=None,
               nleg=8, criteria=[4,2.5], filename='feff.inp'):
    '''
    Write a FEFF8.5 input file. This function writes a typical FEFF input to generate EXAFS chi(k).
    More advanced options are not included here. Defaults are taken from the defaults for FEFF. The 
    exception is SCF, which uses a default rfms1 from the Pappalardo et al. 2021 example file.

    Pappalardo et al. 2021: https://doi.org/10.1021/acs.inorgchem.1c01888
    
    Parameters
    ----------
    potentials : list
        List of POTENTIALS lines, expects {ipot: >3} {atomic_number: >8} {tag: >8}\n
    atoms : list
        List of ATOMS lines, expects {pos_x: .8f} {pos_y: .8f} {pos_z: .8f} {ipot: <8} {tag: <8}\n for each line
    title : str
        TITLE for the file, default='FEFF8.5 input'
    edge : str
        Which EDGE to calculate (i.e. K, L1, L2, etc.), default='K'
    s02 : float
        Amplitude reduction factor, default=1.0
    kmax : float
        Maximum value of k to calculate, default=20
    print_vals : list
        List of length 6 with the output options, default=[0,0,0,0,0,3]
    control : list
        List of length 6 with the modules to run, default=[1,1,1,1,1,1] means run all
    exchange : list
        List of length 3 with [ixc, vr0, vi0] index of potential model, real, and imagninary corrections,
        default=[0,0.0,0.0]
    cfaverage : list
        List of length 3 with [iphabs, nabs, rclabs] potential index, average over nabs absorbers, radius
        of smaller atoms list, default=[0,1,0] means no configurational average
    scf : list
        List of length 4 with [rfms1,lfms1,nscmt,ca], default=[6,0,30,0.2]
    ion : list
        List of length 0 or 2 with [ipot,ionization], default=[] means no ionization
    tdlda : int
        Use time-dependent local density approximation to account for screening, options are 0 (static)
        and 1 (dynamic), default=0
    debye : list
        List of length 2 with [temperature, Debye-temperature], default=[] does not calculate the Debye-Waller factors
    rpath : float or None
        Maximum effective distance, default=None means use 2.2 times nearest neighbor distance
    nleg : int
        Limit the number of legs of each scattering path, default=8
    criteria : list
        List of length 2 with [critcw,critpw], default=[4,2.5]
    filename : str
        Name of the input file to write, default='feff.inp'

    Returns
    -------
    inp : str
        Name of the FEFF input file

    '''

    # decide whether to comment out sections
    if len(ion) == 0:
        ion_comment = '* '
    else:
        ion_comment = ''

    if rpath is None:
        rpath_comment = '* '
    else:
        rpath_comment = ''

    if len(debye) == 0:
        debye_comment = '* '
    else:
        debye_comment = ''

    f = dedent(f'''\
        TITLE     {title}

        EDGE      {edge}
        EXAFS     {kmax}
        S02       {s02}

        *         pot xsph fms path genfmt ff2x
        CONTROL   {list2str(control)}
        PRINT     {list2str(print_vals)}

        *          ixc vr0 vi0 ixc0
        EXCHANGE  {list2str(exchange)}

        *         iphabs nabs rclabs
        CFAVERAGE {list2str(cfaverage)}

        *         rfms1 lfms1 nscmt ca
        SCF       {list2str(scf)}

        *         ipot ionization
        {ion_comment}ION       {list2str(ion)}

        *         ixfc
        TDLDA     {tdlda}

        *         temperature Debye-temperature
        {debye_comment}DEBYE    {list2str(debye)}

        *         rpath
        {rpath_comment}RPATH     {rpath}

        *         nleg
        NLEG      {nleg}

        *         critcw critpw
        CRITERIA  {list2str(criteria)}        
        
    ''')

    inp = open(filename, 'w')
    inp.write(f) # write all the settings information above

    # now write POTENTIALS section
    inp.write('POTENTIALS\n')
    inp.write('* ipot    Z        tag      lmax1      lmax2      xnatph      spinph\n')
    inp.writelines(potentials)

    # now write ATOMS section
    inp.write('\nATOMS\n')
    inp.write('* x           y           z          ipot     tag\n')
    inp.writelines(atoms)

    # END
    inp.write('END')

    inp.close()

    return filename


def load_feff(chi_file):
    data = np.loadtxt(chi_file, comments='#')
    feff = pd.DataFrame(data, columns=['k', 'chi', 'mag', 'phase'])
    feff['k2chi'] = feff.k**2 * feff.chi
    feff = feff.query('k > 0') # remove k=0, if present
    return feff


class EXAFS(ParallelAnalysisBase):
    '''
    Parallelized EXAFS analysis class. Performs a simple parallelization over the frames of a trajectory.
    
    Parameters
    ----------
    absorber : MDAnalysis.AtomGroup
        AtomGroup containing the absorber atom, i.e. the atom around which to calculate the EXAFS.
    write_feff_kwargs : dict
        Dictionary of keyword arguments to pass to the `write_feff` function, which writes the FEFF input file.
        This can include parameters like `edge`, `s02`, `kmax`, etc. See the `write_feff` function for details.
    dir : str
        Directory to write the FEFF input files and results. Default is './'.
    **kwargs : dict
        Additional keyword arguments passed to the base class.

    Results are available through the :attr:`results`.
    
    '''
    def __init__(self, absorber, write_feff_kwargs={}, dir='./', **kwargs):
        super(EXAFS, self).__init__(absorber.universe.trajectory, **kwargs)
        self.absorber = absorber[0] # ensure we have a single atom
        self.u = absorber.universe

        self.dir = dir
        if not self.dir.endswith('/'):
            self.dir += '/'

        if not os.path.exists(self.dir):
            run(f'mkdir {self.dir}')  # create the directory if it does not exist

        self.feff_settings = write_feff_kwargs


    def _prepare(self):

        self.results = Results()
        self.results._per_frame = {}

        # save information for timing
        self.start_time = time()
        self.frame_times = {}

    def _single_frame(self, frame_idx):
        self._frame_index = frame_idx
        self._ts = self.u.trajectory[frame_idx]

        frame_start = time()

        # if the calculation has already been run, skip
        if os.path.exists(f'{self.dir}frame{frame_idx:04d}/chi.dat'):
            df = load_feff(f'{self.dir}frame{frame_idx:04d}/chi.dat')

            frame_end = time()

            print(f'{self.dir}frame{frame_idx:04d}/ already performed, skipping...')

            # return the results for this frame
            return df, frame_idx, frame_end - frame_start

        cluster = self.u.select_atoms(f'sphzone 8 index {self.absorber.index}') - self.absorber # get atoms in a 8 A sphere around the cation

        # first, create a dictionary of unique potential indices
        ipots = {}
        for atom in cluster:
            if not atom.element in ipots.keys():
                ipots[atom.element] = len(ipots.keys())+1

        # write the POTENTIAL lines with indices, tags, and atomic number
        potential_lines = []
        absorber_line = f"{0: >3} {atomic_numbers[self.absorber.element]: >8} {self.absorber.element+'0': >8}\n"
        potential_lines.append(absorber_line) # first, for the absorber
        for el,ipot in ipots.items():
            line = f'{ipot: >3} {atomic_numbers[el]: >8} {el: >8}\n'
            potential_lines.append(line)

        # write the ATOM lines with Cartesian coordinates, indices, tags
        atom_lines = []
        absorber_line = f' {self.absorber.position[0]: .8f} {self.absorber.position[1]: .8f} {self.absorber.position[2]: .8f} {0: <8} {self.absorber.element: <8}\n' # first, for the absorber
        atom_lines.append(absorber_line)
        for atom in cluster:
            line = f' {atom.position[0]: .8f} {atom.position[1]: .8f} {atom.position[2]: .8f} {ipots[atom.element]: <8} {atom.element: <8}\n'
            atom_lines.append(line)

        # write FEFF input and run
        run(f'mkdir {self.dir}frame{frame_idx:04d}')
        inp = write_feff(potential_lines, atom_lines, filename=f'{self.dir}frame{frame_idx:04d}/feff.inp',
                         **self.feff_settings)
        feff = xafs.feff8l(folder=f'{self.dir}frame{frame_idx:04d}', feffinp=inp)
        # feff.run() # I think this is repetitive

        # report results and time for this frame
        df = load_feff(f'{self.dir}frame{frame_idx:04d}/chi.dat')
        frame_end = time()

        # return the results for this frame
        return df, frame_idx, frame_end - frame_start

    
    def _conclude(self):
        
        self.results.k = load_feff(f'{self.dir}frame{0:04d}/chi.dat')['k'].values
        self.results.k2chi = np.zeros(self.results.k.shape)
        
        # extract results from all processors and average k2chi
        for res in self._result:
            df, idx, t = res
            self.results._per_frame[idx] = df
            self.results.k2chi += res['k2chi'].values
            self.frame_times[idx] = t

        self.results.k2chi /= len(self._result)

        # report timing
        self.end_time = time()
        print('\n')
        print('-'*20 + '  Timing  ' + '-'*20)
        for f,t in self.frame_times.items():
            print(f'Frame {f:04d}'.ljust(25) + f'{t:.2f} s'.rjust(25))

        total_time = self.end_time - self.start_time # s
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f'\nTotal time:'.ljust(25) + f'{hours:d}:{minutes:d}:{seconds:d}'.rjust(25))
        print('-'*50)


def feff_per_path(file_idx, files, k, params=None, scattering_paths=None, cluster_map=None, deltar=0.0, e0=0.0, sigma2=0.0, s02=1.0):
    '''
    Function to read a FEFF path file and calculate the chi(k) for a single path.
    
    Parameters
    ----------
    file_idx : int
        Index of the FEFF path file from a list of file paths.
    files : list of str
        List of FEFF path files.
    k : np.ndarray
        k-space points at which to calculate chi(k).
    params : np.ndarray, optional
        Parameters for the FEFF calculation, structured as a 2D array with shape (n_files, 3).
        Each row corresponds to a different path and contains [deltar, e0, sigma2]. Default is None,
        which means the parameters will be used as specified with the below keywords.
    scattering_paths : list of ScatteringPath objects, optional
        List of ScatteringPath objects, each representing a scattering path with its associated cluster.
    cluster_map : dict, optional
        Dictionary mapping each cluster to a parameter index. This is used to determine which parameters
        to apply to each path.
    deltar : float, optional
        Delta R parameter for the FEFF calculation. Default is 0.0.
    e0 : float, optional
        E0 parameter for the FEFF calculation. Default is 0.0.
    sigma2 : float, optional
        Sigma^2 parameter for the FEFF calculation. Default is 0.0.
    s02 : float, optional
        S0^2 parameter for the FEFF calculation. Default is 1.0.

    Returns
    -------
    chi : np.ndarray
        Calculated chi(k) for the given FEFF path.
    
    '''
    
    file = files[file_idx]

    if params is not None and scattering_paths is not None and cluster_map is not None:
        mypath = scattering_paths[file_idx]  # get the ScatteringPath object for this file
        parameter_idx = cluster_map[mypath.cluster_key]  # get the parameter index for this path
        (deltar, e0, sigma2) = params[parameter_idx,:]

    path = xafs.feffpath(file, deltar=deltar, e0=e0, sigma2=sigma2, s02=s02)
    xafs.path2chi(path, k=k) # calculate chi(k) for the path at the same points as experimental data
    k2chi = path.chi * path.k**2  # k^2 * chi(k)
    return k2chi
            

def feff_average_paths_equal(params, k, filepath='./frame*/', top_paths=-1, sort_by_amp_ratio=False, njobs=1, progress_bar=True):
    '''
    Function to average k^2*chi(k) from multiple FEFF calculations assuming all paths have the same parameters
    
    Parameters
    ----------
    params : np.ndarray
        Parameters for the FEFF calculations. These are structured as follows:
        [deltar, e0, sigma2]
        where each element corresponds to a different parameter in the chi(k) calculation.
    k : np.ndarray
        k-space points at which to calculate chi(k).
    filepath : str, optional
        Path to the directory containing the FEFF files. Default is './frames*/'.
    top_paths : int
        Number of top paths to consider for averaging. Default is -1, which means all paths are considered.
    sort_by_amp_ratio : bool, optional
        If True, the paths will be sorted by their amplitude ratio before selecting the top paths.
        Default is False.
    njobs : int, optional
        Number of parallel jobs to run. Default is 1 (no parallelization).
        If set to -1, it will use all available CPU cores.
    progress_bar : bool, optional
        If True, a progress bar will be displayed during the processing of paths. Default is True

    Returns
    -------
    k2chi : np.ndarray
        Averaged k^2*chi(k) values across all frames, structured as a 1D array with same shape as experiment.

    '''

    # Unpack parameters
    deltar = params[0]
    e0 = params[1]
    sigma2 = params[2]

    # parallelize
    if njobs == -1:
        n = multiprocessing.cpu_count()
    else:
        n = njobs

    # format filepath
    if not filepath.endswith('/'):
        filepath += '/'
    
    # Find all files.dat files recursively
    files_dat_list = find_files_dat_files(filepath)
    n_frames = len(files_dat_list) # count number of frames

    paths = []
    for files_dat in files_dat_list:

        try:
            files = read_files(files_dat)
        except:
            print(f'Error reading {files_dat}, skipping...')
            n_frames -= 1 # decrement frame count if this file cannot be read
            continue
        
        if sort_by_amp_ratio:
            files.sort_values('amp_ratio', inplace=True, ascending=False) # get the paths with highest amp_ratio
        
        frame_dir = os.path.dirname(files_dat)
        frame_paths = [os.path.join(frame_dir, f) for f in files.file.values[:top_paths]] if top_paths != -1 else [os.path.join(frame_dir, f) for f in files.file.values]
        paths.extend(frame_paths)

    run_per_path = partial(feff_per_path, files=paths, k=k, deltar=deltar, e0=e0, sigma2=sigma2, s02=0.816)

    if progress_bar:
        with Pool(n) as worker_pool:
            result = []
            for r in tqdm(worker_pool.imap_unordered(run_per_path, np.arange(len(paths))), total=len(paths), desc='Processing paths'):
                result.append(r)

    else:
        with Pool(n) as worker_pool:
            result = worker_pool.map(run_per_path, np.arange(len(paths)))

    result = np.asarray(result) 
    
    return result.sum(axis=0) / n_frames  # sum over all paths, averaged over all frames


def feff_average(params, k, scattering_paths, cluster_map, njobs=1):
    '''
    Function to average k^2*chi(k) from multiple FEFF calculations assuming all paths have different parameters
    
    Parameters
    ----------
    params : np.ndarray
        Parameters for the FEFF calculations. These are structured as a 1D array as follows:
        [cluster0_deltar, cluster0_e0, cluster0_sigma2,
         cluster1_deltar, cluster1_e0, cluster1_sigma2, ...]
        where each element corresponds to a different parameter in the chi(k) calculation.
    scattering_paths : list of ScatteringPath objects
        List of ScatteringPath objects, each representing a scattering path with its associated cluster.
        Each ScatteringPath object should have a `cluster_key` attribute that maps to the cluster_map.
    cluster_map : dict
        Dictionary mapping each cluster to a parameter index. This is used to determine which parameters
        to apply to each path.
    k : np.ndarray
        k-space points at which to calculate chi(k).
    njobs : int, optional
        Number of parallel jobs to run. Default is 1 (no parallelization).
        If set to -1, it will use all available CPU cores.

    Returns
    -------
    k2chi : np.ndarray
        Averaged k^2*chi(k) values across all frames, structured as a 1D array with same shape as experiment.

    '''

    if isinstance(params, list):
        # If using Bayesian optimization, params needs to be converted from a list to a numpy array
        params = np.asarray(params)        


    # parallelize
    if njobs == -1:
        n = multiprocessing.cpu_count()
    else:
        n = njobs
    
    n_frames = len(glob('./frame*/chi.dat'))  # count number of frames
    paths = sorted(glob(f'./frame*/feff*.dat')) # assumes FEFF files are named consistently within frameXXXX
    params = params.reshape(-1,3)  # convert params to a 2D array with shape (n_clusters, 3)
    run_per_path = partial(feff_per_path, files=paths, k=k, params=params, s02=0.816, cluster_map=cluster_map, scattering_paths=scattering_paths)

    with Pool(n) as worker_pool:
        result = []
        for r in tqdm(worker_pool.imap_unordered(run_per_path, np.arange(len(paths))), total=len(paths), desc='Processing paths'):
            result.append(r)

    result = np.asarray(result) 
    
    return result.sum(axis=0) / n_frames  # sum over all paths, averaged over all frames


def R_factor(calc_data, exp_data):
    '''
    Function to calculate the R-factor for the fit of k^2*chi(k) data.
    
    Parameters
    ----------
    calc_data : np.ndarray
        Calculated k^2*chi(k) data, structured as a 1D array with shape (nk,).
    exp_data : np.ndarray
        Experimental k^2*chi(k) data for comparison, structured as a 1D array with shape (nk,).

    Returns
    -------
    R : float
        The R-factor, which is the ratio of the root mean square deviation of the experimental data from the calculated data
        to the root mean square of the experimental data.
    
    '''
    
    R = np.sqrt(np.sum((exp_data - calc_data)**2) / np.sum(exp_data**2))
    
    return R


def opt_func(params, exp_data, k, feff_func=feff_average_paths_equal, loss=R_factor, **kwargs):
    '''
    Function to optimize for the best fit of the average k2chi
    
    Parameters
    ----------
    params : np.ndarray
        Parameters to be optimized. Should be structured as 1D array as described in `feff_average`.
    exp_data : np.ndarray
        Experimental k^2*chi(k) data for comparison, structured as a 2D array with shape (nk,) where nk is the
        number of k-space points in the experimental data.
    k : np.ndarray
        k-space points at which to calculate chi(k).
    feff_func : callable, optional
        Function to calculate the k^2*chi(k) from FEFF paths. Default is feff_average_paths_equal.
        This function should take parameters, k, and any additional keyword arguments.
    loss : callable, optional
        Loss function to minimize. Default is R_factor.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the feff_func function.

    Returns
    -------
    loss_value : float
        The computed loss value between the averaged k2chi and the experimental data.
        
    '''
    
    k2chi = feff_func(params, k=k, **kwargs)

    if np.any(np.isnan(k2chi)) or np.any(np.isinf(k2chi)):
        return 1e6  # Large penalty for invalid results
    loss_value = loss(k2chi, exp_data)
    if np.isnan(loss_value) or np.isinf(loss_value):
        return 1e6
    
    return loss_value


class EarlyStopper:
    '''
    Early stopping callback for optimization routines.
    This class monitors the loss function during optimization and raises a StopIteration exception
    if the loss does not improve for a specified number of iterations (patience).

    Parameters
    ----------
    tol : float, optional
        Tolerance for improvement in the loss function. Default is 1e-4.
    patience : int, optional
        Number of iterations to wait for an improvement before stopping. Default is 10.
    verbose : int, optional
        Levels for verbosity. Default is 0.
        - 0: no output
        - 1: prints the iteration number
        - 2: prints the iteration number and function value
        - 3: prints the iteration number, function value, and parameters

    Attributes
    ----------
    best : float
        Best value of the loss function observed so far.
    counter : int
        Counter for the number of iterations without improvement.
    iteration : int
        Current iteration number, starting from 1.
    
    '''
    
    def __init__(self, tol=1e-4, patience=10, verbose=0):
        self.tol = tol
        self.patience = patience
        self.best = np.inf
        self.counter = 0
        self.verbose = verbose
        self.iteration = 1

    def __call__(self, intermediate_result):
        if self.verbose == 1:
            print(f'Iteration: {self.iteration}')
        elif self.verbose == 2:
            print(f'Iteration: {self.iteration}, function value: {intermediate_result.fun:.6e}')
        elif self.verbose == 3:
            print(f'Iteration: {self.iteration}, function value: {intermediate_result.fun:.6e}, parameters: {intermediate_result.x}')
        self.iteration += 1

        # Save a checkpoint with the current best parameters and intermediate result object
        np.savetxt('./.params.txt', intermediate_result.x, fmt='%.8f')
        with open(f'./checkpoint.pl', 'wb') as output:
            pickle.dump(intermediate_result, output, pickle.HIGHEST_PROTOCOL)

        # Check if the current result is better than the best observed
        current = np.min(intermediate_result.fun)
        if current < self.best - self.tol:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            raise StopIteration("Early stopping triggered due to no improvement in loss function.")


class Atom:
    '''
    Class to represent an atom from the FEFF calculation.
    
    From a line in the FEFF paths.dat file.
    '''

    def __init__(self, line):
        atom_line = line.strip()
        atom_parts = atom_line.split()
        x,y,z = map(float, atom_parts[:3])
        self.xyz = np.array([x, y, z])
        self.ipot = int(atom_parts[3])
        self.label = atom_parts[4].strip("' ")
        self._rleg = float(atom_parts[6])
        self.beta = float(atom_parts[7])
        self.eta = float(atom_parts[8])

    def __repr__(self):
        return f"{self.label} at {self.xyz}"
    

    def __hash__(self): # in case you want to use this as a node
        return hash((self.label, tuple(self.xyz)))


def build_graph_from_atoms(G, atoms):
    '''
    Build a NetworkX directed graph from a list of Atom objects.
    
    Each atom is a node, and edges are the rleg length of the scattering path.
    '''

    coord_to_node = {} # need a mapping from coordinates to node indices
    node_sequence = [] # sequence of node indices in the path
    node_counter = 0

    # add nodes to the graph
    for atom in atoms:
        
        coord = tuple(atom.xyz)  # use tuple for immutability in dict keys
        if coord in coord_to_node: # reuse existing node if coordinates match
            node_idx = coord_to_node[coord]
        else:
            node_idx = node_counter
            coord_to_node[coord] = node_idx
            G.add_node(node_idx, label=atom.label, xyz=atom.xyz, 
                    ipot=atom.ipot, beta=atom.beta, eta=atom.eta)
            node_counter += 1

        node_sequence.append(node_idx)

    # add edges to the graph, weights are the rleg values
    edges = [(node_sequence[idx-1],node_sequence[idx],atoms[idx-1]._rleg) for idx in range(1,len(node_sequence))] + [(node_sequence[-1],node_sequence[0],atoms[-1]._rleg)]  # add last edge to first node to close the path
    G.add_weighted_edges_from(edges)
    
    return G


def parse_paths(filename):
    graphs = []
    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(\d+)\s+(\d+)\s+([\d.]+)\s+index, nleg, degeneracy, r=\s*([\d.]+)', line)
        if m: # start of a path definition

            # Extract path index, number of legs, degeneracy, and r
            path_index = int(m.group(1))
            nleg = int(m.group(2))
            degeneracy = float(m.group(3))
            r = float(m.group(4))
            
            i += 2  # skip to first atom line
            atoms = []
            for atom_idx in range(nleg): # create a list of all Atoms that are in the path
                atoms.append(Atom(lines[i]))
                i += 1
                
            # Build directed graph for this path, reusing nodes with same xyz
            G = nx.DiGraph(path_index=path_index, nleg=nleg, degeneracy=degeneracy, r=r, filename=filename)
            G = build_graph_from_atoms(G, atoms)
            
            graphs.append(G)

        else: # not a path definition, just skip to the next line
            i += 1

    return graphs


def plot_path_graph(G):
    # Extract positions from node attributes
    pos = {n: (d['xyz'][0], d['xyz'][1]) for n, d in G.nodes(data=True)}
    labels = {n: d['label'] for n, d in G.nodes(data=True)}
    
    plt.figure(figsize=(2, 2))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos)
    plt.title(f"Path index: {G.graph.get('path_index', 'N/A')}")
    plt.show()


def get_paths_and_clusters(path_files):
    '''
    Function to read in paths from the FEFF calculation and determine isomorph groups and clusters.
    This function reads the paths.dat files, creates graphs for each scattering path, determines isomorph groups,
    and clusters the paths based on their edge weights and node attributes.

    Parameters
    ----------
    path_files : list of str
        List of paths.dat files to read in. Each file corresponds to a frame of the FEFF calculation.
        Each file should be named in the format 'frameXXXX/paths.dat', where XXXX is the frame index.

    Returns
    -------
    paths : list of ScatteringPath objects
        List of ScatteringPath objects, each representing a scattering path with its associated graph.
    cluster_map : dict
        Dictionary mapping each unique cluster key to a unique index. The keys are tuples of (isomorph_group, cluster).

    '''

    # read in paths and create graphs for each scattering path
    paths = []
    for path in path_files:
        graphs = parse_paths(path)
        frame = int(re.search(r'frame(\d+)', path).group(1))  # extract frame index from filename
        for g in graphs: # loop over each path in the paths.dat file
            mypath = ScatteringPath(frame_idx=frame, path_idx=g.graph['path_index'])
            mypath.graph = g  # attach the graph to the ScatteringPath object
            paths.append(mypath)

    # determine isomorph groups of graphs
    # Use NetworkX's isomorphism module to find isomorphic graphs
    # Create a node matcher that matches nodes based on their 'label' attribute, aka atom identity
    atom_matcher = iso.categorical_node_match('label', 'unknown')
    isomorph_groups = []

    for path in paths:
        g = path.graph  # get the graph from the ScatteringPath object
        found_group = False
        for idx,group in enumerate(isomorph_groups):
            # Compare with the first graph in the group (representative)
            if iso.is_isomorphic(g, group[0], node_match=atom_matcher):
                group.append(g)
                path.isomorph_group = idx  # assign the isomorph group index
                found_group = True
                break
        if not found_group:
            isomorph_groups.append([g])
            path.isomorph_group = len(isomorph_groups) - 1  # assign the new group index

    # Now we have isomorph groups, we can create a DataFrame for each group
    # Each DataFrame contains edge weights and node attributes
    group_dataframes = []
    group_paths = []  # Keep track of ScatteringPath objects for each group

    for group in isomorph_groups:   
        rows = []
        paths_in_group = []
        for g in group:
            row = {}
            # Edge weights
            for i, (u, v, d) in enumerate(g.edges(data=True)):
                row[f'edge{i}'] = d.get('weight', None)
            # Node eta and beta
            for i, (n, d) in enumerate(g.nodes(data=True)):
                # row[f'eta{i}'] = d.get('eta', None) # ignore eta for now
                row[f'beta{i}'] = d.get('beta', None)
        
            rows.append(row)
            # Find the ScatteringPath object corresponding to this graph
            for p in paths:
                if p.graph is g:
                    paths_in_group.append(p)
                    break

        df = pd.DataFrame(rows)
        group_dataframes.append(df)
        group_paths.append(paths_in_group)

    # cluster the paths based on the edge and node attributes
    # these clusters will be the distinct paths for optimization
    n_clusters = 0
    for idx, df in enumerate(group_dataframes):
        paths_in_group = group_paths[idx]
        if len(df) > 5:
            # Scale the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
            
            # Run HDBSCAN
            cluster = HDBSCAN()
            labels = cluster.fit_predict(df_scaled)
            n_clusters += len(set(labels))

            # assign cluster labels to the ScatteringPath objects
            for path_obj, label in zip(paths_in_group, labels):
                path_obj.cluster = int(label)

        else: # if there are not enough paths, assign them all to one cluster
            for path_obj in paths_in_group:
                path_obj.cluster = 0
            n_clusters += 1

    print(f"Total number of clusters: {n_clusters}")

    # Create a map from each unique cluster_key to a unique index
    cluster_keys = set(p.cluster_key for p in paths)
    cluster_map = {key: idx for idx, key in enumerate(sorted(cluster_keys))}

    return paths, cluster_map


class ScatteringPath:
    '''
    Class to represent a scattering path from the FEFF calculation.
    '''

    def __init__(self, frame_idx=None, path_idx=None):
        
        if frame_idx is None or path_idx is None:
            raise ValueError("Both frame_idx and path_idx must be provided to initialize ScatteringPath.")
        
        self.frame = frame_idx
        self.path = path_idx
        self.filename = f'./frame{frame_idx:04d}/feff{path_idx:04d}.dat'
        
        # initalize other data to add
        self.graph = None
        self.isomorph_group = None
        self.cluster = None


    def __repr__(self):
        return f'ScatteringPath(frame={self.frame}, path={self.path})'


    @property
    def cluster_key(self):
        '''Unique key for the cluster parameters'''
        return (self.isomorph_group, self.cluster)


def read_files(filename):
    '''
    Read files.dat file and return a DataFrame
    
    Parameters
    ----------
    filename : str
        Path to the files.dat file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the file paths and their corresponding frame and path indices.
        The DataFrame has columns "file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective".
    
    '''

    # Find the line after the separator (-------)
    with open(filename) as f:
        for i, line in enumerate(f):
            if set(line.strip()) == {"-"}:
                header_line = i + 1
                break

    # Read the table
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=header_line + 1,
        names=["file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective"]
    )

    return df


def k2chi_to_chiR(k, k2chi, kmin=2.5, kmax=8, dk=1):
    '''
    Convert k^2*chi(k) to chi(R) using the Fourier transform. Default parameters are set for Rb analysis
    from Sasha.
    
    Parameters
    ----------
    k : np.ndarray
        k-space points at which k^2*chi(k) is defined.
    k2chi : np.ndarray
        k^2*chi(k) values corresponding to the k points.
    kmin : float, optional
        Minimum k value for the Fourier transform. Default is 2.5.
    kmax : float, optional
        Maximum k value for the Fourier transform. Default is 8.
    dk : float, optional
        Step size in k for the Fourier transform. Default is 1.

    Returns
    -------
    R : np.ndarray
        Real space distances.
    chiR : np.ndarray
        chi(R) values corresponding to the R distances.
    
    '''
    
    chi = k2chi / k**2  # convert k^2*chi(k) to chi(k)
    chi[np.isnan(chi)] = 0  # replace NaNs with 0

    grp = Group(k=k, chi=chi)
    xafs.xftf(grp.k, grp.chi, group=grp, kmin=kmin, kmax=kmax, dk=dk, kweight=2, window='hanning')  # perform Fourier transform

    R = grp.r
    chiR = grp.chir_mag
    
    return R, chiR
