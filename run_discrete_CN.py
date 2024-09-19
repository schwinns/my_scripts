# Script to calculate the discrete CN free energies

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from MDAnalysis.analysis import distances
import pymbar

from solvation_shells_utils import *

if __name__ == '__main__':

    # inputs
    umb_centers = np.linspace(1,9,16)
    K_values = np.ones(16)*100
    n_sims = 16
    start = 0
    by = int(5/0.02) # frequency of saved configurations

    d_min = 2
    d_max = 8
    cn_range = (2,8)
    nbins = 200

    r0 = 3.15
    cation = 'resname NA'
    anion = 'resname CL'

    # initialize the UmbrellaAnalysis and do an initial FES calculation (which saves most of the necessary parameters)
    print('Loading COLVARs')
    umb = UmbrellaAnalysis(n_sims, start=start, by=by)
    print('\nCalculating FES')
    bins, fes = umb.calculate_FES(umb_centers, KAPPA=K_values, d_min=d_min, d_max=d_max, nbins=nbins, error=True)

    # load in trajectories
    print('\nLoading coordinates')
    xtcs = [f'prod_{i}.xtc' for i in range(n_sims)]
    umb.create_Universe(f'mda_readable.tpr', xtcs, cation=cation, anion=anion)
    biased = mda.AtomGroup([umb.cations[0]]) # NOTE: should change if not using biased cation

    # calculate the discrete free energies
    res = umb.calculate_discrete_FE(cn_range, biased, r0, n_bootstraps=50, filename='discrete_FE.dat')

    plt.bar(res.coordination_number, res.free_energy, width=0.25, fc='blue', ec='black', alpha=0.5, label='discrete CN')
    plt.errorbar(res.coordination_number, res.free_energy, yerr=res.error, fmt='none', color='blue')
    plt.plot(umb.bin_centers, umb.fes, c='blue', alpha=0.5, label='continuous CN')
    plt.fill_between(umb.bin_centers, umb.fes-umb.error, umb.fes+umb.error, fc='blue', alpha=0.5)
    plt.xlabel('coordination number')
    plt.ylabel('$\Delta$G (kJ/mol)')
    plt.legend()
    plt.savefig('compare_continuous_vs_discrete_FES.png')
    plt.show()