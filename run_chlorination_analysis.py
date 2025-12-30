# Run all analysis for chlorinated membrane
# hydrogen bonding, RDFs, swelling, density profiles

from glob import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import MDAnalysis as mda
from ParallelMDAnalysis import ParallelInterRDF as InterRDF
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from diffusion_coefficient import DiffusionCoefficient
from solvation_analysis.solute import Solute


def save_object(obj, filename):
    '''Save object to pickle file'''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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
    amide_c = universe.select_atoms('(type c) and (bonded type n)')
    amide_o = universe.select_atoms('(type o) and (bonded group amide_c)', amide_c=amide_c)
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


def calculate_density(universe, membrane_lb=40, membrane_ub=100, frameby=10, bin_width=0.5):
    '''Calculate density profiles for different species and average density of bulk membrane'''

    dim = 'z' # z-dimension
    d = 2
    box = universe.dimensions[d]

    water = universe.select_atoms('resname SOL')
    polymer = universe.select_atoms('resname PA*')

    n_bins = int(box / bin_width)
    bins = np.linspace(0, box, num=n_bins)

    water_density_profile = np.zeros(n_bins-1)
    polymer_density_profile = np.zeros(n_bins-1)
    water_density = np.zeros(len(universe.trajectory[::frameby]))
    total_density = np.zeros(len(universe.trajectory[::frameby]))

    for i, ts in enumerate(tqdm(universe.trajectory[::frameby])):

        sel = universe.select_atoms(f'prop {dim} > {membrane_lb} and prop {dim} < {membrane_ub}')

        # calculate density profiles
        box = universe.dimensions
        for b in range(n_bins-1):
            lb = bins[b]
            ub = bins[b+1]

            box_dims = [box[i] for i in range(3) if i != d]
            dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
            
            bin_atoms = universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group water', water=water)
            mass = bin_atoms.masses.sum() / 6.022 / 10**23
            water_density_profile[b] += mass / dV

            bin_atoms = universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group polymer', polymer=polymer)
            mass = bin_atoms.masses.sum() / 6.022 / 10**23
            polymer_density_profile[b] += mass / dV

        # calculate the water density in the bulk region as a time series
        g = sel.select_atoms('resname SOL')
        xlo, xhi = g.positions[:,0].min(), g.positions[:,0].max() # NOTE: these limits will be slightly different than self.box
        ylo, yhi = g.positions[:,1].min(), g.positions[:,1].max() #       since it is using the min and max atom coordinates, but
        zlo, zhi = g.positions[:,2].min(), g.positions[:,2].max() #       the difference should only be at most 0.002

        total_mass = g.masses.sum() / 6.022 / 10**23 # [g/mol * mol/# = g]
        s1 = (xhi-xlo) * 10**-8 # [Ang * 10^8 cm/Ang = cm]
        s2 = (yhi-ylo) * 10**-8 # cm
        s3 = (zhi-zlo) * 10**-8 # cm
        vol = s1*s2*s3 # cm^3
        density = total_mass / vol
        water_density[i] = density

        # calculate the total density in the bulk region as a time series
        g = sel
        xlo, xhi = g.positions[:,0].min(), g.positions[:,0].max() # NOTE: these limits will be slightly different than self.box
        ylo, yhi = g.positions[:,1].min(), g.positions[:,1].max() #       since it is using the min and max atom coordinates, but
        zlo, zhi = g.positions[:,2].min(), g.positions[:,2].max() #       the difference should only be at most 0.002

        total_mass = g.masses.sum() / 6.022 / 10**23 # [g/mol * mol/# = g]
        s1 = (xhi-xlo) * 10**-8 # [Ang * 10^8 cm/Ang = cm]
        s2 = (yhi-ylo) * 10**-8 # cm
        s3 = (zhi-zlo) * 10**-8 # cm
        vol = s1*s2*s3 # cm^3
        density = total_mass / vol
        total_density[i] = density

    water_density_profile = water_density_profile / len(universe.trajectory[::frameby])
    polymer_density_profile = polymer_density_profile / len(universe.trajectory[::frameby])

    return bins, water_density_profile, polymer_density_profile, water_density, total_density


def coordination_analysis(universe):
    '''Get the coordination environment around water molecules in the membrane'''

    waters = universe.select_atoms('type OW') # only water oxygen atoms
    cations = universe.select_atoms('resname NA')
    anions = universe.select_atoms('resname CL')

    xlink_c = universe.select_atoms(f'(type c) and (bonded type n)')
    coo_c = universe.select_atoms(f'(type c) and (not bonded type oh n)')

    cooh_oh = universe.select_atoms(f'type oh')
    amide_o = universe.select_atoms(f'(type o) and (bonded group xlink_c)', xlink_c=xlink_c)
    coo_o = universe.select_atoms(f'(type o) and (bonded group coo_c)', coo_c=coo_c)
    nh2 = universe.select_atoms(f'type nh')
    cl = universe.select_atoms('type cl')

    solute = Solute.from_atoms(waters,
                        {
                            'amide_o' : amide_o,
                            'cooh_oh' : cooh_oh,
                            'coo_o' : coo_o,
                            'nh2' : nh2,
                            'anions' : anions,
                            'cations' : cations,
                            'cl' : cl
                        },
                        solute_name='Water', radii={'nh2' : 4.0, 'anions' : 3.95, 'cl' : 5.1})

    solute.run()

    return solute


if __name__ == "__main__":

    analysis_to_run = {'hydrogen_bonds': False,
                       'volume' : False,
                       'rdfs': False,
                       'diffusion_coefficient': False,
                       'pore_size_distribution': False,
                       'density_profiles': False, 
                       'coordination' : True}
    path = './'
    tpr = path + 'prod.tpr'
    xtc = path + 'prod_centered.xtc'

    print(f'Loading {tpr} with trajectory {xtc}')
    u = mda.Universe(tpr, xtc)
    dt = u.trajectory[1].time - u.trajectory[0].time
    time = np.arange(0, u.trajectory.n_frames*dt, dt)  # ps

    # calculate percentage of hydrogen bonds over time
    if analysis_to_run['hydrogen_bonds']:
        print('\nCalculating hydrogen bonds')
        hbonds_count = hydrogen_bonds(u)

        total_hn = u.select_atoms('type hn').n_atoms
        hbonds_percentage = (hbonds_count / total_hn) * 100

        # save both count and percentage (two columns) to CSV
        hbonds_csv = path + 'hbonds.csv'
        hbonds_out = np.column_stack((time, hbonds_count, hbonds_percentage))
        np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='time,hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
        print(f'Wrote hbonds count and percentage to {hbonds_csv}')

        print(f'{hbonds_percentage.mean():.2f}% out of {total_hn} possible hydrogen bonds')
    
    if analysis_to_run['volume']:
        # calculate the volume over time
        print('\nCalculating bulk membrane volume')
        volume, xy_area, membrane_bounds = calculate_volume(u)

        # save to CSV
        volume_csv = path + 'membrane_volume.csv'
        vol_out = np.column_stack((time, volume))
        np.savetxt(volume_csv, vol_out, delimiter=',', header='time,membrane_volume_A3', comments='', fmt='%.6f')
        print(f'Wrote membrane volume to {volume_csv}')

        bounds_csv = path + 'membrane_bounds.csv'
        bounds_out = np.column_stack((time, membrane_bounds))
        np.savetxt(bounds_csv, bounds_out, delimiter=',', header='time,membrane_lower_bound_A,membrane_upper_bound_A', comments='', fmt='%.6f')
        print(f'Wrote membrane bounds to {bounds_csv}')

        print(f'Average membrane volume: {volume.mean():.2f} A^3')
        print(f'Average membrane bounds: {membrane_bounds.mean(axis=0)} A')

    if analysis_to_run['rdfs']:
        # calculate RDFs
        print('\nCalculating RDFs for water in membrane')
        rdfs = calculate_RDFs(u, membrane_lower_bound=membrane_bounds[:,0].mean(), membrane_upper_bound=membrane_bounds[:,1].mean(), njobs=16)

        # save to CSV
        rdfs_csv = path + 'rdfs.csv'
        rdfs.to_csv(rdfs_csv, index=False)
        print(f'Wrote RDFs to {rdfs_csv}')

    if analysis_to_run['diffusion_coefficient']:
        # calculate MSDs and diffusion coefficients
        print('\nCalculating water MSD and diffusion coefficients in membrane')
        diff = DiffusionCoefficient(tpr, xtc, water='type OW')
        # frac, waters = diff.restrict_to_membrane(diff.water, membrane_fraction=(0.25, 0.75))
        # D, D_ci = diff.run(waters, n_bootstraps=50, confidence=0.5)

        D, D_ci = diff.run(diff.water, n_bootstraps=50, confidence=0.5)
        save_object(diff, path + 'diffusion_coefficient_unrestricted.pkl')

        # save to MSD to CSV
        msd_csv = path + 'water_msd_unrestricted.csv'
        np.savetxt(msd_csv, np.column_stack((diff.lagtimes, diff.msd_ts, diff.msd_stderr, diff.msd_ci[0], diff.msd_ci[1])), delimiter=',', header='time,msd,stderr,lower_ci,upper_ci', comments='', fmt='%.6f')
        print(f'Wrote water MSD to {msd_csv}')

        print(f'Diffusion coefficient: {D*10**-9:.4e} m^2/s with CI ({D_ci[0]*10**-9:.4e}, {D_ci[1]*10**-9:.4e})')

    if analysis_to_run['pore_size_distribution']:
        # merge pore size distribution data from all frames
        print('\nAveraging pore size distributions')
        frames = natsorted(glob(path + 'poreblazer/frame*/Total_psd.txt'))
        bins, psd, std = average_PSD(frames)

        # save to CSV
        psd_csv = path + 'average_psd.csv'
        np.savetxt(psd_csv, np.column_stack((bins, psd, std)), delimiter=',', header='pore_diameter_A,average_psd,std_dev', comments='', fmt='%.6f')
        print(f'Wrote average PSD to {psd_csv}')

        print(f'Max in the PSD is at {bins[np.argmax(psd)]:.2f} Angstroms')

    if analysis_to_run['density_profiles']:
        # calculate density profiles and average densities
        print('\nCalculating density profiles and average densities')
        bins, water_density_profile, polymer_density_profile, water_density, total_density = calculate_density(u, membrane_lb=membrane_bounds[:,0].mean(), membrane_ub=membrane_bounds[:,1].mean(), frameby=1, bin_width=0.5)

        # save both profiles and time series to CSV
        density_csv = path + 'density.csv'
        density_out = np.column_stack((time, water_density, total_density))
        np.savetxt(density_csv, density_out, delimiter=',', header='time,water_density,total_density', comments='', fmt='%.6f')
        print(f'Wrote average densities to {density_csv}')

        density_profile_csv = path + 'density_profiles.csv'
        density_profile_out = np.column_stack((bins[1:], water_density_profile, polymer_density_profile))
        np.savetxt(density_profile_csv, density_profile_out, delimiter=',', header='z,water_density_profile,polymer_density_profile', comments='', fmt='%.6f')
        print(f'Wrote density profiles to {density_profile_csv}')


    if analysis_to_run['coordination']:

        # calculate coordination environment around water molecules
        print('\nCalculating coordination environment around water molecules')
        solute = coordination_analysis(u)

        # save to pickle
        coord_pkl = path + 'water_coordination.pkl'
        save_object(solute, coord_pkl)
        print(f'Wrote water coordination analysis to {coord_pkl}')
    