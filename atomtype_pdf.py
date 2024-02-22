# Script to calculate the PDF weighted by scattering factors between MDAnalysis atom groups

# standard imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import MDAnalysis as mda
from time import time
from itertools import combinations

# my imports
from scattering_factors import *

def unwrap_PA(u):
    '''Unwrap the polyamide membrane and center it in the box using MDAnalysis'''

    # select only PA membrane
    PA = u.select_atoms('resname PA*')

    # get dimensions of the box
    box = u.dimensions[:3]

    # calculate the center of mass and move PA atoms that are > box/2 from the COM
    com = PA.center_of_mass()
    for atom in PA:
        dist = atom.position[2] - com[2]
        if abs(dist) >= box[2]/2: # if atom is on the opposite side of the box
            if dist < 0: # if atom is below COM
                atom.position[2] += box[2]
            elif dist > 0: # if atom is above COM
                atom.position[2] -= box[2]

    # center the atoms on the PA membrane
    u.atoms.translate(box/2 - PA.center_of_mass())

    # wrap so atoms fit within box
    u.atoms.wrap()


def rdf_to_data(irdf, savename=None):
    '''
    Convert an MDAnalysis InterRDF object to an array
    '''
    r = irdf.results.bins
    g_r = irdf.results.rdf

    if savename is not None:
        data = np.vstack((r, g_r)).T
        np.savetxt(savename, data, fmt='%.8f', delimiter='\t', header='r (A)\tg(r)')

    return r, g_r


def plot_g_r_from_array(r, g_r, label=None, alpha=1, ylims=None, xlims=None, color=None, ax=None, output=None):
    '''
    Plot PDF nicely from an r and g(r) array
    r : units Angstroms
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

    ax.plot(r, g_r, label=label, alpha=alpha, c=color)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    if ylims is None:
        plt.ylim(0, g_r.max() + 1)
    else:
        plt.ylim(ylims[0], ylims[1])

    if xlims is None:
        plt.xlim(0, r.max())
    else:
        plt.xlim(xlims[0], xlims[1])

    plt.ylabel('g(r)')
    plt.xlabel('r (A)')
    if label is not None:
        plt.legend()

    if output is not None:
        plt.savefig(output)

    return ax


if __name__ == '__main__':

    start_time = time()

    # inputs
    top = 'prod.tpr'
    trj = 'prod.xtc'
    frameby = 10
    r_range = np.array([0.0, 10.0])
    bin_width = 0.05
    atom_types = ['c', 'ca', 'n', 'nh', 'o', 'oh', 'OW']
    
    # initialize universe and scattering factors object
    print(f'Loading trajectory {trj} with topology {top}...')
    u = mda.Universe(top, trj)
    sf = ScatteringFactors()

    n_bins = int((r_range[1] - r_range[0]) / bin_width)
    n_frames = len(u.trajectory[::frameby])

    # unwrap the polymer and center in box
    print('Locating bulk membrane...')
    PA_zones = np.zeros((n_frames, 5))
    for i,ts in enumerate(u.trajectory[::frameby]):

        # unwrap this frame
        unwrap_PA(u)

        # select only polymer
        PA = u.select_atoms('resname PA*')

        # get the z limits and bulk zone of PA membrane
        PA_zlims = np.array([PA.positions[:,2].min(), PA.positions[:,2].max()])
        PA_zones[i,:] = np.linspace(PA_zlims[0], PA_zlims[1], 5)

        ts.dimensions[2] = PA_zones[i,3] - PA_zones[i,1]

    # create atom group for only bulk membrane
    bulk = u.select_atoms(f'prop z >= {PA_zones[:,1].mean()} and prop z <= {PA_zones[:,3].mean()}', updating=True)

    # calculate RDF for all combinations of atom types
    # for at1, at2 in combinations(atom_types, 2):
    for at1 in atom_types:

        at2 = at1
        # get atom group
        sel1 = bulk.select_atoms(f'type {at1}', updating=True)
        sel2 = bulk.select_atoms(f'type {at2}', updating=True)

        # run RDF calculation
        print(f'Calculating the weighted PDF for type {at1} to type {at2} over {n_frames} frames...')
        wrdf = WeightedRDF(sel1,sel2, range=r_range, nbins=n_bins, scattering_factors=sf, exclusion_block=(1,1))
        wrdf.run(step=frameby)

        # save RDF data and plot
        print(f"Saving data to {at1+'-'+at2+'.dat'} and saving plot to {at1+'-'+at2+'.png'}...")
        r, g_r = rdf_to_data(wrdf, savename=at1+'-'+at2+'.dat')
        ax = plot_g_r_from_array(r, g_r, output=at1+'-'+at2+'.png', xlims=(0,10))


    end_time = time()
    print(f'Finished after {end_time - start_time} s!')

    