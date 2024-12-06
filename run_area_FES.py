# Script to calculate the discrete CN free energies

import matplotlib.pyplot as plt
import numpy as np

from solvation_shells_utils import *

if __name__ == '__main__':

    # inputs
    paths = [
        './C0.055M/P1bar/Na/total_CN/no_HREX/',
        './C0.2M/P1bar/Na/',
        './C0.6M/P1bar/Na/',
        './C1.8M/P1bar/Na/total_CN/',
    ]
    concs = ['dilute', '0.2 M', '0.6 M', '1.8 M']

    umb_centers = np.linspace(1,9,16)
    K_values = np.ones(16)*100
    n_sims = 16
    start = 0
    by = int(5/0.02) # frequency of saved configurations

    d_min = 2
    d_max = 8
    nbins = 200
    n_boots = 50

    cation = 'resname NA'
    anion = 'resname CL'

    area_range = None
    area_bins = 50

    for filepath in paths:
    
        # initialize the UmbrellaAnalysis and do an initial FES calculation (which saves most of the necessary parameters)
        print(f'Loading COLVARs for {filepath}')
        umb = UmbrellaAnalysis(n_sims, COLVAR_file=filepath+'COLVAR_', start=start, by=by)
        print('Calculating continuous FES')
        if n_boots == 0:
            bins, fes = umb.calculate_FES(umb_centers, KAPPA=K_values, d_min=d_min, d_max=d_max, nbins=nbins, error=False) # do not subsample time series if we are not calculating error in discrete FE
        else:
            bins, fes = umb.calculate_FES(umb_centers, KAPPA=K_values, d_min=d_min, d_max=d_max, nbins=nbins, error=True)

        # load in trajectories
        print('Loading coordinates')
        xtcs = [filepath+f'prod_{i}.xtc' for i in range(n_sims)]
        umb.create_Universe(filepath+f'mda_readable.tpr', xtcs, cation=cation, anion=anion)

        # load in polyhedron data
        poly = load_object(filepath+'polyhedrons.pl')
        umb.polyhedron_sizes = poly

        # calculate the FES in area
        print('Calculating area FES')
        bins, fes, err = umb.calculate_area_FES(area_range=area_range, nbins=area_bins, n_bootstraps=n_boots, filename=filepath+'fes_area.dat')

        fig, ax1 = plt.subplots(1,1)
        ax1.plot(bins, fes, c='tab:red')
        ax1.fill_between(bins, fes-err, fes+err, fc='tab:red', alpha=0.5)
        ax1.set_xlabel('polyhedron area', color='tab:red')
        ax1.set_ylabel('$\Delta$G (kJ/mol)')
        ax1.tick_params('x', labelcolor='tab:red')

        ax2 = ax1.twiny()

        ax2.plot(umb.bin_centers, umb.fes, c='tab:orange')
        ax2.fill_between(umb.bin_centers, umb.fes-umb.error, umb.fes+umb.error, fc='tab:orange', alpha=0.5)
        ax2.set_xlabel('continuous coordination number', color='tab:orange')
        ax2.set_ylabel('$\Delta$G (kJ/mol)')
        ax2.tick_params('x', labelcolor='tab:orange')
        fig.savefig(filepath+'compare_continuous_area_FES.png')