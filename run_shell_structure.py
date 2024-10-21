# Script to run all shell structure analyses on the umbrella simulations
# This includes RDFs by CN, water dipole distributions, and convex hull analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time

from solvation_shells_utils import *

if __name__ == '__main__':

    # some inputs
    n_sims = 16
    tpr = 'mda_readable.tpr'
    njobs = 32
    ci = 'NA'
    ai = 'CL'
    bi = 'NA'
    r0 = 3.15

    # intialize UmbrellaAnalysis
    print('Loading data...')
    start_time = time.perf_counter()
    umb = UmbrellaAnalysis(n_sims)

     # create a universe with trajectory data
    xtcs = [f'prod_{i}.xtc' for i in range(n_sims)]
    umb.create_Universe(tpr, xtcs, cation=f'resname {ci}', anion=f'resname {ai}')
    dt = umb.universe.trajectory[1].time - umb.universe.trajectory[0].time
    biased_ion = umb.universe.select_atoms(f'resname {bi}')[0]
    load_data_time = time.perf_counter()

    # calculate the discrete coordination nubers and save as csv
    cn = umb.get_coordination_numbers(mda.AtomGroup([biased_ion]), r0, njobs=njobs)
    df = pd.DataFrame()
    df['idx'] = np.arange(0,len(cn))
    df['time'] = df['idx']*dt
    df['coordination_number'] = cn
    df.to_csv('discrete_coordination_numbers.csv', index=False)
    cn_time = time.perf_counter()

    # calculate the RDFs by CN and save as csvs by CN
    print('Calculating the RDFs by discrete coordination number...')
    umb.rdfs_by_coordination(mda.AtomGroup([biased_ion]), np.arange(1,16))

    for i in umb.rdfs['i-w'].keys():
        df = pd.DataFrame()
        df['r'] = umb.rdfs['i-w'][i].bins
        for k in umb.rdfs.keys():
            df[k] = umb.rdfs[k][i].rdf

        df = df.fillna(0)
        df.to_csv(f'rdf_CN_{i}.csv', index=False)

    rdf_time = time.perf_counter()

    # calculate the water dipoles
    print('Calculating water dipoles...')
    res1 = umb.water_dipole_distribution(biased_ion, radius=r0, njobs=njobs)
    with open(f'water_dipoles.pl', 'wb') as output:
            pickle.dump(res1, output, pickle.HIGHEST_PROTOCOL)

    water_dipole_time = time.perf_counter()

    # calculate the polyhedron sizes
    print('Running polyhedron size analysis...')
    res2 = umb.polyhedron_size(biased_ion, r0=r0, njobs=njobs)
    with open(f'polyhedrons.pl', 'wb') as output:
            pickle.dump(res2, output, pickle.HIGHEST_PROTOCOL)

    polyhedron_time = time.perf_counter()

    print('Plotting some results...')    
    plt.figure()
    plt.hist(res1.angles.flatten(), bins=50, ec='white')
    plt.ylabel('counts')
    plt.xlabel('water dipoles (degrees)')
    plt.savefig('dipoles_hist.png')
    
    plt.figure()
    plt.hist(res2.areas, bins=50, ec='white')
    plt.ylabel('counts')
    plt.xlabel('area ($\AA^2$)')
    plt.savefig('area_hist.png')

    plot_time = time.perf_counter()

    print('\n' + '-'*20 + ' Timing ' + '-'*20)
    print(f'Loading data:       \t\t\t\t{load_data_time - start_time:.4f} s')
    print(f'Calculating coordination numbers: \t{cn_time - load_data_time:.4f} s')
    print(f'Calculating RDFs by CN: \t\t\t{rdf_time - cn_time:.4f} s')
    print(f'Calculating water dipoles: \t\t{water_dipole_time - rdf_time:.4f} s')
    print(f'Calculating polyhedron sizes: \t\t{polyhedron_time - water_dipole_time:.4f} s')
    print(f'Plotting:           \t\t\t\t{plot_time - polyhedron_time:.4f} s\n')
    print(f'Total:              \t\t\t\t{plot_time - start_time:.4f} s')
    print('-'*48)
