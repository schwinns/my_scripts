# Run all analysis for chlorinated membrane
# hydrogen bonding, RDFs, swelling, density profiles

from glob import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
from tqdm import tqdm

import MDAnalysis as mda
from ParallelMDAnalysis import ParallelInterRDF as InterRDF
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from diffusion_coefficient import DiffusionCoefficient


def hydrogen_bonds(universe):
    '''Calculate the number of hydrogen bonds over time'''

    donors_sel = f'(type n) or (type nh)'
    acceptors_sel = f'(type o)'
    hydrogens_sel = f'(type hn)'

    hbonds = HBA(universe=universe,
                donors_sel=donors_sel,
                acceptors_sel=acceptors_sel,
                hydrogens_sel=hydrogens_sel)

    hbonds.run(
        start=None,
        stop=None,
        step=None,
        verbose=True
    )

    return hbonds.count_by_time()


def calculate_volume(universe):
    '''Calculate the volume of the middle 50% of polymer over time'''

    polymer = universe.select_atoms('resname PA*')

    membrane_bounds = np.zeros((universe.trajectory.n_frames, 5))
    xy_area = np.zeros(universe.trajectory.n_frames)
    for i,ts in enumerate(tqdm(universe.trajectory)):

        polymer_min = np.nonzero(polymer.positions[:,2] < 10)[0]
        if len(polymer_min) > 0: # there are some extreme cases where the polymer jumps due to PBC
            # Create a mask to exclude indices in polymer_min
            mask = np.ones(polymer.positions.shape[0], dtype=bool)  # Start with all True
            mask[polymer_min] = False  # Set indices in polymer_min to False
            min_pos = polymer.positions[mask,2].min()

        else:
            min_pos = polymer.positions[:,2].min()

        membrane_bounds[i,:] = np.linspace(min_pos, polymer.positions[:,2].max(), num=5)
        xy_area[i] = (universe.dimensions[0]*universe.dimensions[1]) # A^2

    vol = xy_area * (membrane_bounds[:,3] - membrane_bounds[:,1])  # A^3 of the middle 50% of membrane
    return vol, xy_area, np.column_stack((membrane_bounds[:,1], membrane_bounds[:,3]))


def calculate_RDFs(universe, membrane_lower_bound=43, membrane_upper_bound=99, njobs=1):
    '''Calculate RDFs for water molecules in the bulk membrane'''

    waters_in_membrane = universe.select_atoms(f'type OW and prop z > {membrane_lower_bound} and prop z < {membrane_upper_bound}', updating=True)
    polymer = universe.select_atoms('resname PA*')
    cl = universe.select_atoms('type cl')
    cations = universe.select_atoms('resname NA')
    anions = universe.select_atoms('resname CL')
    cooh = universe.select_atoms('type oh')
    coo_c = universe.select_atoms('(type c) and (not bonded type n) and (not bonded type oh)')
    coo_o = universe.select_atoms('(type o) and (bonded group coo_c)', coo_c=coo_c)
    amide_n = universe.select_atoms('(type n) and (bonded type c)')
    amide_o = universe.select_atoms('(type o) and (bonded type n)')
    amine_n = universe.select_atoms('type nh')

    groups = [waters_in_membrane, polymer, polymer.select_atoms('not element H'), cl, cations, 
            anions, cooh, coo_c, coo_o, amide_n, amide_o, amine_n]
    labels = ['Water', 'Polymer', 'Polymer (no H)', 'polymer Cl', 'Na+', 
            'Cl-', 'COOH', 'COO-C', 'COO-O', 'Amide N', 'Amide O', 'Amine N']

    rdf_results = np.zeros((len(groups)+1, 150)) # 150 bins, 10 rdfs, and r values

    for i,g2 in enumerate(groups):
        print(f'\tCalculating RDF for {labels[i]}')
        irdf = InterRDF(waters_in_membrane, g2, nbins=150, range=(0.0, 15.0), verbose=True)
        irdf.run(njobs=njobs)
        rdf_results[i+1,:] = irdf.results.rdf

    rdf_results[0,:] = irdf.results.bins
    df = pd.DataFrame(rdf_results.T, columns=['r'] + labels)

    return df


def average_PSD(frames):
    '''Average pore size distributions from multiple frames'''

    psd_list = []
    for frame in frames:
        data = np.loadtxt(frame, skiprows=1)
        psd_list.append(data[:,1])

    psd_array = np.array(psd_list)
    bins = data[:,0]
    psd_mean = psd_array.mean(axis=0)
    psd_std = psd_array.std(axis=0)

    return bins, psd_mean, psd_std


if __name__ == "__main__":

    path = './'
    tpr = path + 'prod.tpr'
    xtc = path + 'prod_centered.xtc'

    print(f'Loading {tpr} with trajectory {xtc}')
    u = mda.Universe(tpr, xtc)

    # calculate percentage of hydrogen bonds over time
    print('\nCalculating hydrogen bonds')
    hbonds_count = hydrogen_bonds(u)

    total_hn = u.select_atoms('type hn').n_atoms
    hbonds_percentage = (hbonds_count / total_hn) * 100

    # save both count and percentage (two columns) to CSV
    hbonds_csv = path + 'hbonds.csv'
    hbonds_out = np.column_stack((hbonds_count, hbonds_percentage))
    np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
    print(f'Wrote hbonds count and percentage to {hbonds_csv}')

    print(f'{hbonds_percentage.mean():.2f}% out of {total_hn} possible hydrogen bonds')
    
    # calculate the volume over time
    print('\nCalculating bulk membrane volume')
    volume, xy_area, membrane_bounds = calculate_volume(u)

    # save to CSV
    volume_csv = path + 'membrane_volume.csv'
    np.savetxt(volume_csv, volume, delimiter=',', header='membrane_volume_A3', comments='', fmt='%.6f')
    print(f'Wrote membrane volume to {volume_csv}')

    print(f'Average membrane volume: {volume.mean():.2f} A^3')

    # calculate RDFs
    print('\nCalculating RDFs for water in membrane')
    rdfs = calculate_RDFs(u, membrane_lower_bound=membrane_bounds[:,1].mean(), membrane_upper_bound=membrane_bounds[:,3].mean(), njobs=16)

    # save to CSV
    rdfs_csv = path + 'rdfs.csv'
    rdfs.to_csv(rdfs_csv, index=False)
    print(f'Wrote RDFs to {rdfs_csv}')

    # calculate MSDs and diffusion coefficients
    print('\nCalculating water MSD and diffusion coefficients in membrane')
    diff = DiffusionCoefficient(tpr, xtc)
    frac, waters = diff.restrict_to_membrane(diff.water, membrane_fraction=(0.25, 0.75))
    D, D_ci = diff.run(waters, n_bootstraps=50, confidence=0.5)

    # save to MSD to CSV
    msd_csv = path + 'water_msd.csv'
    np.savetxt(msd_csv, np.column_stack((diff.lagtimes, diff.msd_ts, diff.msd_stderr, diff.msd_ci[0,:], diff.msd_ci[1,:])), delimiter=',', header='time,msd,stderr,lower_ci,upper_ci', comments='', fmt='%.6f')
    print(f'Wrote water MSD to {msd_csv}')

    print(f'Diffusion coefficient: {D:.4e} with CI ({D_ci[0]:.4e}, {D_ci[1]:.4e})')

    # merge pore size distribution data from all frames
    print('\nAveraging pore size distributions')
    frames = natsorted(glob(path + 'poreblazer/frame*/Total_psd.txt'))
    bins, psd, std = average_PSD(frames)

    # save to CSV
    psd_csv = path + 'average_psd.csv'
    np.savetxt(psd_csv, np.column_stack((bins, psd, std)), delimiter=',', header='pore_diameter_A,average_psd,std_dev', comments='', fmt='%.6f')
    print(f'Wrote average PSD to {psd_csv}')

    print(f'Max in the PSD is at {bins[np.argmax(psd)]:.2f} Angstroms')