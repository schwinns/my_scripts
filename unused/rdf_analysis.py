# RDF analysis with MDTraj
# Scales by f_i(0)*f_j(0) / <f>^2

import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry.distance import compute_distances
import numpy as np
import json
from gromacs.formats import XVG
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from time import time
from tqdm import tqdm

################################# INPUTS ########################################

frame_start = -1000
frame_by = 100                      # Only calculate the RDF when frame % frame_by = 0
timing = True                       # if True, display timing information
plot = True                         # if True, show final RDF plot

traj = './prod.xtc'             # input trajectory
gro = './prod.gro'              # input coordinate file
json_factors = './form_factors.json'      # json with atomic form factors
filename = './rdf.xvg'   # output RDF filename

#################################################################################

# compute_rdf taken from MDTraj and modified to scale distances
def compute_rdf(traj, pairs, r_range=None, bin_width=0.005, n_bins=None,
                periodic=True, opt=True, scaling_factors=None):
    """Compute radial distribution functions for pairs in every frame.
    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute radial distribution function in.
    pairs : array-like, shape=(n_pairs, 2), dtype=int
        Each row gives the indices of two atoms.
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will compute distances under the minimum image
        convention.
    opt : bool, default=True
        Use an optimized native library to compute the pair wise distances.
    scaling_factors : array-like, shape=(n_pairs,), dtype=float, default=None
        Weight the distances by the form factors of the atoms involved. 
        Each row gives the scaling factor for the corresponding pair
    Returns
    -------
    r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radii values corresponding to the centers of the bins.
    g_r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radial distribution function values at r.
    See also
    --------
    Topology.select_pairs
    """
    if r_range is None:
        r_range = np.array([0.0, 1.0])
    r_range = ensure_type(r_range, dtype=np.float64, ndim=1, name='r_range',
                          shape=(2,), warn_on_cast=False)
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('`n_bins` must be a positive integer')
    else:
        n_bins = int((r_range[1] - r_range[0]) / bin_width)
    
    distances = compute_distances(traj, pairs, periodic=periodic, opt=opt)

    if scaling_factors is not None:
        g_r, edges = np.histogram(distances, range=r_range, bins=n_bins, weights=scaling_factors)
    else:
        g_r, edges = np.histogram(distances, range=r_range, bins=n_bins)
    
    r = 0.5 * (edges[1:] + edges[:-1])
    unitcell_vol = traj.unitcell_volumes

    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf.
    V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = len(pairs) * np.sum(1.0 / unitcell_vol) * V
    g_r = g_r.astype(np.float64) / norm  # From int64.

    return r, g_r


#################################################################################
########################### MAIN USE OF FUNCTIONS ###############################
#################################################################################

# Load trajectory
start = time()
print('\n\n--------------------------- PROGRESS ---------------------------')
print("Loading trajectory '%s' with topology '%s'..." %(traj, gro) )
t = md.load(traj, top=gro)
load_time = time()

# Get pairs and calculate scaling factors
print('Finding pairs and calculating scaling factors...')
atom_idx = t.top.select('all')
form_factors = json.load(open('form_factors.json'))

np_factors = np.zeros(len(atom_idx))
for i, idx in enumerate(atom_idx):
    f_i = form_factors[t.top.atom(idx).element.symbol]
    np_factors[i] = f_i
    
avg_f = np_factors.mean()

pairs = t.top.select_pairs('all', 'all')
scaling = np.zeros(len(pairs))
n = 0
for i in tqdm(range(atom_idx.shape[0]-1)):
    f_i = np_factors[i]
    for j in range(i+1,atom_idx.shape[0]):
        f_j = np_factors[j]
        scaling[n] = f_i*f_j
        n += 1
        
scaling_factors = np.zeros((t[frame_start::frame_by].n_frames, len(pairs)))
for f in range(t[frame_start::frame_by].n_frames):
    scaling_factors[f,:] = scaling / avg_f**2

pair_time = time()

# Compute the RDF
print(f'Computing the RDF for {len(pairs):,d} pairs...')
r, g_r = compute_rdf(t[frame_start::frame_by], pairs, scaling_factors=scaling_factors, r_range=(0, 2.5), bin_width=0.01)
r = r*10
rdf_comp = time()

################################# RESULTS ########################################


print("Writing the RDF data to '%s'" %(filename))
data = np.vstack((r, g_r))
xvg = XVG(array=data, names=['r [Å]', 'g(r)'])
xvg.write(filename)
print('----------------------------------------------------------------')

rdf_time = time()
if timing:
    print('\n\n----------------------- TIMING BREAKDOWN -----------------------')
    print('\tLoading trajectory and inputs:\t\t\t%.2f s' %(load_time - start))
    print('\tFinding pairs:\t\t\t\t\t%.2f s' %(pair_time - load_time))
    print('\tRDF computations:\t\t\t\t%.2f s' %(rdf_time - pair_time))
    print('\n\tTotal time:\t\t\t\t\t%.2f s' %(rdf_time - start))
    print('----------------------------------------------------------------\n')

if plot:
    fig, ax = plt.subplots(1,1)
    plt.plot(r, g_r)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    plt.xlim(0,10)
    plt.ylim(0, g_r.max() + 1)
    plt.xlabel('r [Å]')
    plt.ylabel('g(r)')
    plt.show()

#################################################################################
