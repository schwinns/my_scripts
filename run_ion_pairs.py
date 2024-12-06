# Script to calculate the discrete CN free energies

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    njobs = 16

    n_sims = 16
    start = 0
    by = int(5/0.02) # frequency of saved configurations

    cation = 'resname NA'
    anion = 'resname CL'

    # regions for ion pairing events from standard MD
    ion_pair_cutoffs = {'CIP': (0, 3.75), 
                    'SIP': (3.75, 6.15), 
                    'DSIP': (6.15, 8.35), 
                    'FI': (8.35, np.inf)
                    }

    for filepath in paths:
    
        # initialize the UmbrellaAnalysis
        print(f'Loading COLVARs for {filepath}')
        umb = UmbrellaAnalysis(n_sims, COLVAR_file=filepath+'COLVAR_', start=start, by=by)

        # load in trajectories
        print('\nLoading coordinates')
        xtcs = [filepath+f'prod_{i}.xtc' for i in range(n_sims)]
        umb.create_Universe(filepath+f'mda_readable.tpr', xtcs, cation=cation, anion=anion)
        biased = mda.AtomGroup([umb.cations[0]]) # NOTE: should change if not using biased cation

        # calculate the ion pairing time series
        print('Calculating ion pair events')
        freq = umb.ion_pairing(biased, ion_pair_cutoffs, plot=False, njobs=njobs)

        df = pd.DataFrame(umb.ion_pairs.data)
        df.to_csv(filepath+'ion_pairing.csv', index=False)
