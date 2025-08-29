# Class to extract clusters from MD simulations and calculate EXAFS spectra with FEFF, optimize with experimental data

# standard
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter as time
from tqdm import tqdm

# file handling
from glob import glob
import os
import pickle
import re
import tarfile
from textwrap import dedent
import tempfile
import subprocess
import shutil

# NetworkX
import networkx as nx
from networkx.algorithms import isomorphism as iso

# clustering
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

# MD data
from ase.data import atomic_numbers
from MDAnalysis.analysis.base import Results
from ParallelMDAnalysis import ParallelAnalysisBase

# Larch
from larch import xafs
from larch import Group

# plotting preferences
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.size'] = 16

# parallel processing
import multiprocessing
from multiprocessing import Pool
from functools import partial

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


def extract_from_tar(func): # decorator to extract files from tar archives
    def wrapper(filename, *args, **kwargs):
        # Check if this is a tar path like: archive.tar.gz/path/inside.tar
        is_tar = False
        parts = filename.split(os.sep)
        for i, part in enumerate(parts):
            if part.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
                archive_path = os.sep.join(parts[:i + 1])
                inner_path = os.sep.join(parts[i + 1:])
                is_tar = True
                break

        if not is_tar:
            # Not a tar path; call the original function
            return func(filename, *args, **kwargs)

        # Extract inner file from the tar archive
        with tarfile.open(archive_path, 'r:*') as tar:
            try:
                member = tar.getmember(inner_path)
                f = tar.extractfile(member)
                return func(f, *args, **kwargs)
            
            except KeyError:
                # Fallback for the cluster
                frame = [part for part in filename.split(os.sep) if part.startswith('frame')][0].strip('tar.gz')
                filepath = os.path.join(frame, inner_path)
                try:
                    member = tar.getmember(filepath)
                    f = tar.extractfile(member)
                    return func(f, *args, **kwargs)
                except KeyError:
                    raise FileNotFoundError(f"Neither '{inner_path}' nor '{filepath}' found in '{archive_path}'")
    return wrapper


def extract_from_tar_class(func): # decorator to extract files from tar archives
    def wrapper(self, filename, *args, **kwargs):
        # Check if this is a tar path like: archive.tar.gz/path/inside.tar
        is_tar = False
        parts = filename.split(os.sep)
        for i, part in enumerate(parts):
            if part.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
                archive_path = os.sep.join(parts[:i + 1])
                inner_path = os.sep.join(parts[i + 1:])
                is_tar = True
                break

        if not is_tar:
            # Not a tar path; call the original function
            return func(self, filename, *args, **kwargs)

        # Extract inner file from the tar archive
        with tarfile.open(archive_path, 'r:*') as tar:
            try:
                member = tar.getmember(inner_path)
                f = tar.extractfile(member)
                return func(self, f, *args, **kwargs)
            
            except KeyError:
                # Fallback for the cluster
                frame = [part for part in filename.split(os.sep) if part.startswith('frame')][0].strip('tar.gz')
                filepath = os.path.join(frame, inner_path)
                try:
                    member = tar.getmember(filepath)
                    f = tar.extractfile(member)
                    return func(self, f, *args, **kwargs)
                except KeyError:
                    raise FileNotFoundError(f"Neither '{inner_path}' nor '{filepath}' found in '{archive_path}'")
    return wrapper


def check_file_in_tar(archive_path, inner_path):
    """
    Check if a file exists within a compressed tar archive.
    
    Parameters
    ----------
    archive_path : str
        Path to the tar archive (e.g., 'archive.tar.gz', 'archive.tar', etc.)
    inner_path : str
        Path to the file within the archive (e.g., 'folder/file.txt')
        
    Returns
    -------
    bool
        True if the file exists in the archive, False otherwise
        
    Examples
    --------
    >>> check_file_in_tar('frame0001.tar.gz', 'chi.dat')
    True
    >>> check_file_in_tar('data.tar.gz', 'results/output.txt')
    False
    """
    try:
        with tarfile.open(archive_path, 'r:*') as tar:
            try:
                # Try to get the member - this will raise KeyError if not found
                tar.getmember(inner_path)
                return True
            except KeyError:
                # Fallback for the cluster
                frame = [part for part in archive_path.split(os.sep) if part.startswith('frame')][0].strip('tar.gz')
                filepath = os.path.join(frame, inner_path)
                try:
                    member = tar.getmember(filepath)
                    return True
                except KeyError:
                    return False
    except (tarfile.TarError, FileNotFoundError, OSError):
        # Archive doesn't exist or is corrupted
        return False


def check_file_in_tar_path(full_path):
    """
    Check if a file exists within a tar archive using a full path notation.
    
    This function handles paths like 'archive.tar.gz/path/inside/file.txt' and
    determines if the file exists within the archive.
    
    Parameters
    ----------
    full_path : str
        Full path including the archive and internal path
        (e.g., 'frame0001.tar.gz/chi.dat' or 'data.tar.gz/results/output.txt')
        
    Returns
    -------
    bool
        True if the file exists in the archive, False otherwise
        
    Examples
    --------
    >>> check_file_in_tar_path('frame0001.tar.gz/chi.dat')
    True
    >>> check_file_in_tar_path('data.tar.gz/results/output.txt')
    False
    >>> check_file_in_tar_path('regular_file.txt')  # Not a tar path
    False
    """
    # Check if this is a tar path
    parts = full_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
            archive_path = os.sep.join(parts[:i + 1])
            inner_path = os.sep.join(parts[i + 1:])
            return check_file_in_tar(archive_path, inner_path)
    
    # Not a tar path, check if it's a regular file
    return os.path.exists(full_path)


def compress_directory(source_dir, output_filename, compression='gz', arcname=None):
    """
    Compress an entire directory into a tar archive
    
    Args:
        source_dir (str): Path to the directory to compress
        output_filename (str): Output filename for the archive
        compression (str): Compression method - 'gz', 'bz2', 'xz', or None for no compression
        arcname (str, optional): Name to use for the archive inside the tar file. If None, uses the base name of source_dir.
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        compress_directory('/path/to/my_folder', 'my_folder.tar.gz')
        compress_directory('/path/to/my_folder', 'my_folder.tar.bz2', compression='bz2')
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return False
    
    if not os.path.isdir(source_dir):
        print(f"Error: '{source_dir}' is not a directory.")
        return False
    
    # Set compression mode
    if compression:
        mode = f'w:{compression}'
        if not output_filename.endswith(f'.tar.{compression}'):
            print(f"Warning: Output filename should end with '.tar.{compression}' for {compression} compression")
    else:
        mode = 'w'
        if not output_filename.endswith('.tar'):
            print(f"Warning: Output filename should end with '.tar' for uncompressed archives")
    
    try:
        if arcname is None:
            arcname = os.path.basename(source_dir)  # Use the base name of the source directory as the archive name

        with tarfile.open(output_filename, mode) as tar:
            tar.add(source_dir, arcname=arcname)
        
        print(f"Successfully created archive: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error creating archive: {e}")
        return False


def get_user_confirmation(files_to_remove, always_confirm=False):
    """
    Ask for user confirmation
    
    Args:
        files_to_remove: List of files that will compressed then removed
        always_confirm: If True, always allow, never ask for confirmation
    
    Returns:
        bool: True if user confirms, False otherwise
    """
    if not files_to_remove:
        print("No files will be excluded from compression.")
        return True

    print('The following files will be compressed and then removed:')
    for file in files_to_remove:
        print(file)
    
    print(f"\nTotal files to compress: {len(files_to_remove)}")
    
    if always_confirm:
        return True

    while True:
        response = input("\nDo you want to proceed with compressing and deleting these files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes, 'n' for no")


def remove_files(file_list):
    """
    Remove files from the filesystem
    
    Args:
        file_list: List of files to remove
    """
    for file_path in file_list:
        try:
            os.remove(file_path)
            # print(f"Removed {file_path}")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")


def k2chi_axis(ion='Unspecified', xmin=1, xmax=6, ymin=-0.6, ymax=0.6, 
               xtick=0.2, ytick=0.05, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6, 4))

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$k$ (1/$\AA$)')
    ax.set_ylabel(f'$k^2 \chi(k)$ around {ion}')
    ax.set_xticks(np.arange(0, 40, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_minor_locator(MultipleLocator(xtick))
    ax.yaxis.set_minor_locator(MultipleLocator(ytick))
    return ax


def chiR_axis(ion='Unspecified', xmin=0, xmax=6, ymin=0, ymax=0.6, 
               xtick=0.2, ytick=0.05, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6, 4))

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$R$ ($\AA$)')
    ax.set_ylabel(f'|$\chi(R)$| ($\AA^{-3}$) around {ion}')
    ax.set_xticks(np.arange(0, 40, 1))
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
    inp.write(f'{f}') # write all the settings information above

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

@extract_from_tar
def load_feff(chi_file):
    data = np.loadtxt(chi_file, comments='#')
    feff = pd.DataFrame(data, columns=['k', 'chi', 'mag', 'phase'])
    feff['k2chi'] = feff.k**2 * feff.chi
    feff = feff.query('k > 0') # remove k=0, if present
    return feff


@extract_from_tar
def load_path(chi_file):
    return np.loadtxt(chi_file, comments='#')


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
        if os.path.exists(f'{self.dir}frame{frame_idx:04d}.tar.gz'):
            
            df = load_feff(f'{self.dir}frame{frame_idx:04d}.tar.gz/chi.dat')
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

        # report results and time for this frame
        df = load_feff(f'{self.dir}frame{frame_idx:04d}/chi.dat')

        # clean up the files to save disk space
        compressed_file = f'{self.dir}frame{frame_idx:04d}.tar.gz'
        directory_to_compress = f'{self.dir}frame{frame_idx:04d}/'
        if compress_directory(directory_to_compress, compressed_file, compression='gz'):
            shutil.rmtree(directory_to_compress)  # remove the directory after compression

        frame_end = time()

        # return the results for this frame
        return df, frame_idx, frame_end - frame_start

    
    def _conclude(self):
        
        self.results.k = load_feff(f'{self.dir}frame{0:04d}.tar.gz/chi.dat')['k'].values
        self.results.k2chi = np.zeros(self.results.k.shape)
        
        # extract results from all processors and average k2chi
        for res in self._result:
            df, idx, t = res
            self.results._per_frame[idx] = df
            self.results.k2chi += df['k2chi'].values
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


class Averager:
    '''
    Class to calculate the average k^2*chi(k) from multiple FEFF path files.
    
    Attributes
    ----------
    deltar : float
        DeltaR parameter for the FEFF calculation.
    e0 : float
        E0 parameter for the FEFF calculation.
    sigma2 : float
        Sigma^2 parameter for the FEFF calculation.
    s02 : float
        S0^2 parameter for the FEFF calculation, default is 0.816 (for Rb from Sasha).
    njobs : int
        Number of parallel jobs to run.
    progress_bar : bool
        Whether to show a progress bar during processing.
    files : pd.DataFrame
        DataFrame containing the file paths and their corresponding information from files.dat.
    paths : list of ScatteringPath objects
        List of ScatteringPath objects containing the FEFF path data.
    n_frames : int
        Number of frames.
    n_paths : int
        Number of paths.

    Methods
    -------
    average(paths)
        Calculate the average k^2*chi(k) from multiple FEFF path files.
    remove_hydrogen_paths()
        Remove paths that contain hydrogen atoms from the list of paths and update the files DataFrame.
    update_paths(files=None)
        Update the paths based on the files.dat information. Either you can directly edit the `files` attribute then run this method,
        or you can pass a DataFrame with the same columns as the `files` attribute.

    '''


    def __init__(self, frames, frame_dir='./', deltar=0, e0=0, sigma2=0, s02=0.816, include_hydrogen=False, njobs=1, progress_bar=False):
        '''
        Initialize the Averager class.

        Parameters
        ----------
        frames : np.ndarray
            Array of frame indices to be averaged.
        frame_dir : str, list
            One or more directories containing the frame directories with FEFF path files. 
            Each frame should have a subdirectory named 'frameXXXX' where XXXX is the frame index.
        deltar : float, optional
            DeltaR parameter for the FEFF calculation. Default is 0.0.
        e0 : float, optional
            E0 parameter for the FEFF calculation. Default is 0.0.
        sigma2 : float, optional
            Sigma^2 parameter for the FEFF calculation. Default is 0.0.
        s02 : float, optional
            S0^2 parameter for the FEFF calculation. Default is 0.816, which is for Rb from Sasha.
        include_hydrogen : bool, optional
            Whether to include paths that contain hydrogen atoms. Default is False.
        njobs : int, optional
            Number of parallel jobs to run. Default is 1. If -1, uses all available CPU cores.
        progress_bar : bool, optional
            Whether to show a progress bar during processing. Default is False.

        '''
        
        self.deltar = deltar
        self.e0 = e0
        self.sigma2 = sigma2
        self.s02 = s02
        self.progress_bar = progress_bar
        
        if njobs == -1:
            self.njobs = os.cpu_count()
        else:
            self.njobs = njobs

        if not isinstance(frame_dir, list):
            frame_dir = [frame_dir]  # ensure frame_dir is a list

        for fd in frame_dir:
            if not os.path.isdir(fd):
                raise FileNotFoundError(f"Directory {fd} does not exist.")

        # read in all frame information
        dfs = []
        self.paths = []  # list of ScatteringPath objects
        self.n_frames = 0  # number of frames
        for fd in frame_dir:
            for frame in frames:
                frame_path = os.path.join(fd, f'frame{frame:04d}.tar.gz')
                if not os.path.exists(frame_path):
                    if os.path.exists(frame_path[:-7]): # check for non-archive directory
                        raise FileNotFoundError(f"Frame directory {frame_path} is not compressed.")
                        compress_directory(frame_path[:-7], frame_path, compression='gz')
                    else:
                        raise FileNotFoundError(f"Frame directory {frame_path} does not exist.")

                try:
                    df = self._read_files_dat(os.path.join(frame_path, 'files.dat'))
                    df['file'] = df['file'].apply(lambda x: os.path.join(frame_path, x))
                    df['frame'] = frame  # add a frame column for reference
                    df['path_index'] = df['file'].str.extract(r'feff(\d+)').astype(int) # faster than apply with regex

                    self._graphs = self._graphs_from_paths_dat(os.path.join(frame_path, 'paths.dat')) # get graphs for all scattering paths
                    file_map = dict(zip(df['path_index'], df['file'])) # faster than DataFrame lookup
                    mypaths = []
                    for g in self._graphs:
                        file = file_map.get(g.graph["path_index"])
                        if file:
                            mypaths.append(ScatteringPath(path_index=g.graph["path_index"], filename=file, graph=g))

                except:
                    print(f"Warning: Could not read files.dat or paths.dat in {frame_path}. Skipping this frame.")
                    continue

                if df.shape[0] != len(mypaths):
                    print(f"Warning: Number of paths ({len(mypaths)}) does not match number of files ({df.shape[0]}) in {frame_path}. Skipping this frame.")
                    continue

                dfs.append(df)
                self.paths.extend(mypaths)  # extend the paths list with the new paths
                self.n_frames += 1  # increment the number of frames

        self.files = pd.concat(dfs, ignore_index=True) # save files.dat from all frames in a single DataFrame
        self.files['paths'] = self.paths

        if not include_hydrogen:
            self.remove_hydrogen_paths()
        

    def __repr__(self):
        return f'Averager with {self.n_paths} paths across {self.n_frames} frames'
    

    def __str__(self):
        return self.__repr__()


    def copy(self):
        '''
        Create a copy of the current Averager instance.
        This method creates a deep copy of the Averager instance, including all attributes and paths.
        '''
        return copy.deepcopy(self)


    @property
    def n_paths(self):
        return len(self.paths)
    

    def average(self, k, paths=None, njobs=None, chunk_size=None):
        '''
        Calculate the average k^2*chi(k) from multiple FEFF path files.

        Parameters
        ----------
        k : np.ndarray
            k-space points at which to calculate chi(k).
        paths : list of ScatteringPath objects, optional
            List of ScatteringPath objects containing the FEFF path data. Default is None, which uses the `paths` attribute.
        njobs : int, optional
            Number of parallel jobs to run. Default is None, which uses the `njobs` attribute. If this is specified, it overrides
            the class attribute.
        chunk_size : int, optional
            Number of paths to process in each chunk for better load balancing. Default is None, which uses adaptive sizing.
        
        Returns
        -------
        k2chi_avg : np.ndarray
            Average k^2*chi(k) across all frames.

        '''

        if paths is None:
            paths = self.paths

        if njobs is not None:
            self.njobs = njobs
        
        if chunk_size is None:
            # Adaptive chunk size: balance between overhead and load balancing
            chunk_size = max(1, len(paths) // (self.njobs * 4))

        # run_per_path = partial(_feff_per_path, paths=paths, k=k, deltar=self.deltar, e0=self.e0, sigma2=self.sigma2, s02=self.s02)
        filenames = [p.filename for p in paths]
        run_per_path = partial(_feff_per_path, filenames=filenames, k=k, deltar=self.deltar, e0=self.e0, sigma2=self.sigma2, s02=self.s02)

        if self.njobs > 1:
            with Pool(self.njobs) as worker_pool:
                if self.progress_bar:
                    results = list(tqdm(worker_pool.imap(run_per_path, range(len(paths)), chunksize=chunk_size),
                                    total=len(paths), desc='Processing paths'))
                else:
                    results = worker_pool.map(run_per_path, range(len(paths)), chunksize=chunk_size)
        else:
            # Single-threaded execution
            results = tqdm([run_per_path(i) for i in range(len(paths))])

        # Convert to numpy array and sum in one operation
        self._result_array = np.array(results)
        return self._result_array.sum(axis=0) / self.n_frames
    

    def remove_hydrogen_paths(self):
        '''
        Remove paths that contain hydrogen atoms from the list of paths and update the files DataFrame.
        
        This method modifies the `paths` attribute in place, removing any paths that contain hydrogen atoms,
        and also removes corresponding entries from the `files` DataFrame attribute.
        '''
        
        if self.progress_bar:
            print('Removing paths with hydrogen atoms...')
        
        # Create lists
        filtered_paths = []
        to_remove = []
        
        # Use tqdm for progress bar if enabled
        iterator = tqdm(self.paths) if self.progress_bar else self.paths
        
        for p in iterator:
            # Use any() with generator expression for early exit
            has_hydrogen = any(atom['label'] == 'H' for atom in p.atoms)
            if has_hydrogen:
                to_remove.append(p.filename)
            else:
                filtered_paths.append(p)
        
        # Replace the paths list
        self.paths = filtered_paths
        
        # Remove rows from self.files where 'file' is in to_remove
        if to_remove:
            self.files = self.files[~self.files['file'].isin(to_remove)].reset_index(drop=True)


    def update_paths(self, files=None):
        '''
        Update the paths based on the files.dat information. Either you can directly edit the `files` attribute then run this method,
        or you can pass a DataFrame with the same columns as the `files` attribute. This allows you to filter the paths based on efficient
        pandas operations on the files.dat information.

        For example, if you only want the paths with 100% amplitude ratio, you can do:
        >>> Averager.update_paths(Averager.files.query('amp_ratio == 1.0'))

        Parameters
        ----------
        files : pd.DataFrame, optional
            DataFrame that looks like the `files` attribute, i.e. with the same columns. Note that this option overrides the `files` attribute.
            If None, it uses the `files` attribute. Default is None.

        '''

        if isinstance(files, pd.DataFrame):
            if set(files.columns) != set(self.files.columns):
                raise ValueError("The new DataFrame does not have the same columns as the `files` attribute.")
            
            self.files = files  # directly set the files attribute if a DataFrame is passed

        self.paths = self.files['paths'].to_list()  # update paths based on the files DataFrame


    def get_isomorph_groups(self):
        '''
        Get isomorph groups of the paths based on their graphs. This method uses NetworkX's isomorphism to find groups of paths that are isomorphic.
        Implements an atom matcher so that it checks for isomorphism based on the elements, not just the graph structure.

        '''
        
        # Create a node matcher that matches nodes based on their 'label' attribute, aka atom identity
        atom_matcher = iso.categorical_node_match('label', 'unknown')
        self.isomorph_groups = []
        isomorph_indices = []  # to keep track of indices of isomorph groups

        iterator = tqdm(self.paths) if self.progress_bar else self.paths
        for path in iterator:
            g = path.graph  # get the graph from the ScatteringPath object
            found_group = False
            for idx,group in enumerate(self.isomorph_groups):
                # Compare with the first graph in the group (representative)
                if iso.is_isomorphic(g, group[0].graph, node_match=atom_matcher):
                    group.append(path)
                    path.isomorph_group = idx  # assign the isomorph group index
                    isomorph_indices.append(idx)
                    found_group = True
                    break
            if not found_group:
                self.isomorph_groups.append([path])
                path.isomorph_group = len(self.isomorph_groups) - 1  # assign the new group index
                isomorph_indices.append(len(self.isomorph_groups) - 1)

        # add to the files DataFrame
        self.files['isomorph_group'] = isomorph_indices


    @extract_from_tar_class
    def _graphs_from_paths_dat(self, filename):
        graphs = []
        
        # Handle both file paths and file-like objects
        if hasattr(filename, 'read'):
            # It's a file-like object (from tar)
            content = filename.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            lines = content.splitlines()
        else:
            # It's a file path
            with open(filename, 'r') as f:
                lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines]  # Remove newlines for consistency

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


    @extract_from_tar_class
    def _read_files_dat(self, filename):
        '''
        Read files.dat file and return a DataFrame
        
        Parameters
        ----------
        filename : str or file-like object
            Path to the files.dat file or file-like object from tar archive.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the file paths and their corresponding frame and path indices.
            The DataFrame has columns "file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective".
        
        '''

        # Handle both file paths and file-like objects
        if hasattr(filename, 'read'):
            # It's a file-like object (from tar)
            content = filename.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            lines = content.splitlines()
        else:
            # It's a file path
            with open(filename, 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip('\n\r') for line in lines]  # Remove newlines for consistency

        # Find the line after the separator (-------)
        header_line = None
        for i, line in enumerate(lines):
            if set(line.strip()) == {"-"}:
                header_line = i + 1
                break
        
        if header_line is None:
            raise ValueError(f"Could not find dash line separator in the file: {filename}")

        # Extract data lines (skip header line and column names)
        data_lines = lines[header_line + 1:]
        
        # Parse the data manually
        data = []
        for line in data_lines:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 6:  # Ensure we have all required columns
                    data.append(parts[:6])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective"])
        
        # Convert numeric columns to appropriate types
        numeric_columns = ["sig2", "amp_ratio", "deg", "nlegs", "r_effective"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


# def _feff_per_path(idx, paths, k, deltar=0.0, e0=0.0, sigma2=0.0, s02=1.0):
def _feff_per_path(idx, filenames, k, deltar=0.0, e0=0.0, sigma2=0.0, s02=1.0):
    '''
    Function to read a FEFF path file and calculate the chi(k) for a single path.
    
    Parameters
    ----------
    idx : int
        Index of the FEFF path file from a list of file paths.
    paths : list of ScatteringPath objects
        List of ScatteringPath objects containing the FEFF path data.
    k : np.ndarray
        k-space points at which to calculate chi(k).
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
    k2chi : np.ndarray
        Calculated k^2*chi(k) for the given FEFF path.
    
    '''
    
    # path = paths[idx]
    # file_pattern = path.filename.split('.dat')[0]
    filename = filenames[idx]
    file_pattern = filename.split('.dat')[0]
    perform_calc = True  # flag to determine if we need to perform the calculation

    # check if the file pattern contains a tar archive
    is_tar = False
    # parts = path.filename.split(os.sep)
    parts = filename.split(os.sep)
    for i, part in enumerate(parts):
        if part.endswith(('.tar.gz', '.tgz', '.tar', '.tar.bz2')):
            archive_path = os.sep.join(parts[:i + 1])
            internal_path = os.sep.join(parts[i + 1:])
            is_tar = True
            break

    # read in the text file if it has already been calculated
    chi_file = file_pattern + '_chi.dat'
    if check_file_in_tar_path(chi_file):

        data = load_path(chi_file)

        if data[:,0].shape == k.shape: # check if the k values match
            perform_calc = False
            k2chi = data[:, 2]
            # path.k2chi = k2chi  # save k^2*chi(k) in the path object for later use

        return k2chi

    if perform_calc:

        # If we are using a tar archive, we need to create a temporary feff path file
        if is_tar:
            rng = np.random.default_rng()
            i = rng.integers(0, 1e6)

            # frame = [part for part in path.filename.split(os.sep) if part.startswith('frame')][0].strip('tar.gz')
            frame = [part for part in filename.split(os.sep) if part.startswith('frame')][0].strip('tar.gz')
            filepath = os.path.join(frame, internal_path)
            with tarfile.open(archive_path, 'r:*') as tar:
                while os.path.exists(f'./.tmp_path{i}'):
                    i = rng.integers(0, 1e6)

                tmp_dir = f'./.tmp_path{i}'
                tar.extract(filepath, path=tmp_dir)
                
            filepath = os.path.join(tmp_dir, filepath)

        else:
            # filepath = path.filename
            filepath = filename

        p = xafs.feffpath(filepath, deltar=deltar, e0=e0, sigma2=sigma2, s02=s02)
        xafs.path2chi(p, k=k) # calculate chi(k) for the path at the same points as experimental data
        k2chi = p.chi * p.k**2  # k^2 * chi(k)
        # path.k2chi = k2chi  # save k^2*chi(k) in the path object for later use
        
        # # Save the results to a file
        if is_tar:
            shutil.rmtree(tmp_dir)

        #     tmp_dir = f'./.tmp_{frame}'
        #     with tarfile.open(archive_path, 'r:*') as tar:
        #         tar.extractall(path=tmp_dir)  # extract the tar archive to a temporary directory

        #     # add per frame chi to temporary directory
        #     header = f'deltar={deltar:.3f} e0={e0:.3f} sigma2={sigma2:.3f} s02={s02:.3f}\n# k chi k2chi'
        #     np.savetxt(os.path.join(tmp_dir, frame, internal_path.split('.dat')[0]+'_chi.dat'), np.column_stack((p.k, p.chi, k2chi)), header=header, fmt='%.6f %.6f %.6f')

        #     if compress_directory(tmp_dir, archive_path, compression='gz', arcname=''):
        #         shutil.rmtree(tmp_dir)  # remove the temporary directory after saving

        # else: # just save to a regular file
        #     header = f'deltar={deltar:.3f} e0={e0:.3f} sigma2={sigma2:.3f} s02={s02:.3f}\n# k chi k2chi'
        #     np.savetxt(f'{file_pattern}_chi.dat', np.column_stack((p.k, p.chi, k2chi)), header=header, fmt='%.6f %.6f %.6f')


    return k2chi


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
        self.checkpoint = './.checkpoint.pkl'  # file to save the checkpoint

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
        with open(self.checkpoint, 'wb') as output:
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

@extract_from_tar
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
        Dictionary mapping each cluster to a parameter index. This is used to determine which parameters
        to apply to each path.
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

    def __init__(self, path_index, filename, graph=None):
        
        self.path = path_index
        self.filename = filename
        
        # initalize other data to add
        self.graph = graph
        self.isomorph_group = None
        self.cluster = None
        self.k2chi = None  # will be set later when calculating chi(k)

        if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
            self.atoms = [a for n,a in graph.nodes(data=True)]  # extract atoms from the graph


    def __repr__(self):
        return f'ScatteringPath from {self.filename}'


    def __str__(self):
        return self.__repr__()


    def show(self):
        '''
        Show the path graph using NetworkX's drawing capabilities.
        '''
        plot_path_graph(self.graph)


    # @property
    # def cluster_key(self):
    #     '''Unique key for the cluster parameters'''
    #     return (self.isomorph_group, self.cluster)

@extract_from_tar
def read_files(filename):
    '''
    Read files.dat file and return a DataFrame
    
    Parameters
    ----------
    filename : str or file-like object
        Path to the files.dat file or file-like object from tar archive.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the file paths and their corresponding frame and path indices.
        The DataFrame has columns "file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective".
    
    '''

    # Handle both file paths and file-like objects
    if hasattr(filename, 'read'):
        # It's a file-like object (from tar)
        content = filename.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        lines = content.splitlines()
    else:
        # It's a file path
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines]  # Remove newlines for consistency

    # Find the line after the separator (-------)
    header_line = None
    for i, line in enumerate(lines):
        if set(line.strip()) == {"-"}:
            header_line = i + 1
            break
    
    if header_line is None:
        raise ValueError(f"Could not find dash line separator in the file: {filename}")

    # Extract data lines (skip header line and column names)
    data_lines = lines[header_line + 1:]
    
    # Parse the data manually
    data = []
    for line in data_lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split()
            if len(parts) >= 6:  # Ensure we have all required columns
                data.append(parts[:6])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["file", "sig2", "amp_ratio", "deg", "nlegs", "r_effective"])
    
    # Convert numeric columns to appropriate types
    numeric_columns = ["sig2", "amp_ratio", "deg", "nlegs", "r_effective"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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


def find_chiR_peak(r, chiR):
    '''
    Find the peak in chi(R) and return the position and value of the peak.
    This function assumes that the peak is the maximum value in chi(R) and returns the corresponding r value.

    Parameters
    ----------
    r : np.ndarray
        Real space distances corresponding to chi(R).
    chiR : np.ndarray
        chi(R) values corresponding to the r distances.

    Returns
    -------
    peak_r : float
        The r value at which the peak of chi(R) occurs.
    peak_chiR : float
        The value of chi(R) at the peak position.

    '''

    peak_idx = np.argmax(chiR)  # find the index of the maximum value in chi(R)
    peak_r = r[peak_idx]  # get the corresponding r value
    peak_chiR = chiR[peak_idx]  # get the value of chi(R) at the peak position
    
    return peak_r, peak_chiR