# Class to calculate RDFs, coordination numbers, and cutoffs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

import MDAnalysis as mda
from MDAnalysis.analysis.base import Results
from ParallelMDAnalysis import ParallelCoordinationNumbers as CN
from ParallelMDAnalysis import ParallelInterRDF as InterRDF

plt.rcParams['font.size'] = 16


class IonAnalysis:

    def __init__(self, top, traj, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Initialize the equilibrium analysis object with a topology and a trajectory from
        a production simulation with standard MD
        
        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc)
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'
            
        '''

        self.universe = mda.Universe(top, traj)
        self.n_frames = len(self.universe.trajectory)
        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')


    def generate_rdfs(self, bin_width=0.05, range=(0,20), step=1, filename=None, njobs=1):
        '''
        Calculate radial distributions for the solution. This method calculates the RDFs for cation-water,
        anion-water, water-water, and cation-anion using MDAnalysis InterRDF. It saves the data in a 
        dictionary attribute `rdfs` with keys 'ci-w', 'ai-w', 'w-w', and 'ci-ai'.

        Parameters
        ----------
        bin_width : float
            Width of the bins for the RDFs, default=0.05 Angstroms
        range : array-like
            Range over which to calculate the RDF, default=(0,20)
        step : int
            Trajectory step for which to calculate the RDF, default=1
        filename : str
            Filename to save RDF data, default=None means do not save to file
        njobs : int
            Number of CPUs to run on, default=1

        Returns
        -------
        rdfs : dict
            Dictionary with all the results from InterRDF
        
        '''

        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {}

        print('\nCalculating cation-water RDF...')
        ci_w = InterRDF(self.cations, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_w.run(step=step, njobs=njobs)
        self.rdfs['ci-w'] = ci_w.results

        print('\nCalculating anion-water RDF...')
        ai_w = InterRDF(self.anions, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ai_w.run(step=step, njobs=njobs)
        self.rdfs['ai-w'] = ai_w.results

        print('\nCalculating water-water RDF...')
        w_w = InterRDF(self.waters, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True, exclusion_block=(1,1))
        w_w.run(step=step, njobs=njobs)
        self.rdfs['w-w'] = w_w.results

        print('\nCalculating cation-anion RDF...')
        ci_ai = InterRDF(self.cations, self.anions, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_ai.run(step=step, njobs=njobs)
        self.rdfs['ci-ai'] = ci_ai.results

        if filename is not None:
            data = np.vstack([ci_w.results.bins, ci_w.results.rdf, ai_w.results.rdf, w_w.results.rdf, ci_ai.results.rdf]).T
            np.savetxt(filename, data, header='r (Angstroms), cation-water g(r), anion-water g(r), water-water g(r), cation-anion g(r)')

        return self.rdfs


    def get_coordination_numbers(self, step=1, njobs=1):
        '''
        Calculate the water coordination number as a function of time for both cations and anions.
        
        Parameters
        ----------
        step : int
            Trajectory step for which to calculate coordination numbers
        njobs : int
            Number of CPUs to run on, default=1
        
        Returns
        -------
        avg_CN : tuple
            Tuple containing the average coordination number for cations and anions over the trajectory.
        
        '''

        if not hasattr(self, 'cation_cutoff') or not hasattr(self, 'anion_cutoff'):
            raise ValueError('Cutoffs have not been calculated. Please run get_ion_oxygen_distances() first.')

        counter_ci = CN(self.cations, self.waters, cutoff=self.cation_cutoff)
        counter_ci.run(step=step, njobs=njobs)

        counter_ai = CN(self.anions, self.waters, cutoff=self.anion_cutoff)
        counter_ai.run(step=step, njobs=njobs)
        
        self.coordination_numbers = Results()
        self.coordination_numbers.cations = counter_ci.results.cn
        self.coordination_numbers.anions = counter_ai.results.cn

        return self.coordination_numbers.cations.mean(), self.coordination_numbers.anions.mean()


    def get_ion_oxygen_distances(self):
        '''
        Calculate the distances between cations and anions to the nearest water oxygen from the RDFs.
        Additionally, this method finds the first minimum in the RDFs for cation-water and anion-water.
        
        Returns
        -------
        iod : dict
            Dictionary with keys 'cation' and 'anion' containing the distances to the nearest water oxygen
        
        '''

        if not self.rdfs:
            raise ValueError('RDFs have not been calculated. Please run generate_rdfs() first.')

        distances = {}

        # Find the first peak in the cation-water RDF
        peaks,_ = find_peaks(self.rdfs['ci-w'].rdf, distance=5)
        if len(peaks) > 0:
            distances['cation'] = self.rdfs['ci-w'].bins[peaks[0]]

        # now, the first minimum
        valleys,_ = find_peaks(-self.rdfs['ci-w'].rdf, distance=5)
        if len(valleys) > 0:
            self.cation_cutoff = self.rdfs['ci-w'].bins[valleys[0]]
        
        # Find the first peak in the anion-water RDF
        peaks,_ = find_peaks(self.rdfs['ai-w'].rdf, distance=5)
        if len(peaks) > 0:
            distances['anion'] = self.rdfs['ai-w'].bins[peaks[0]]

        # now, the first minimum
        valleys,_ = find_peaks(-self.rdfs['ai-w'].rdf, distance=5)
        if len(valleys) > 0:
            self.anion_cutoff = self.rdfs['ai-w'].bins[valleys[0]]

        if 'cation' not in distances or 'anion' not in distances:
            raise ValueError('Could not find peaks in RDFs for cation or anion. Check the RDFs.')

        self.iod = distances
        return self.iod

    def plot_RDFs(self):
        '''
        Plot the calculated RDFs for cation-water, anion-water, water-water, and cation-anion.

        Returns
        -------
        fig : matplotlib Figure
            The figure containing the RDF plots

        '''

        if not hasattr(self, 'rdfs'):
            raise ValueError('RDFs have not been calculated. Please run generate_rdfs() first.')

        axes = []
        for key, rdf in self.rdfs.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(rdf.bins, rdf.rdf, label=key, c='b')
            ax.set_xlabel('Distance (Angstroms)')
            ax.set_ylabel('g(r)')

            if key == 'ci-w' and hasattr(self, 'cation_cutoff'):
                ax.axvline(self.cation_cutoff, color='r', linestyle='--', label='Cation cutoff')
            elif key == 'ai-w' and hasattr(self, 'anion_cutoff'):
                ax.axvline(self.anion_cutoff, color='r', linestyle='--', label='Anion cutoff')

            if key == 'ci-w' and hasattr(self, 'iod'):
                ax.scatter(self.iod['cation'], np.max(rdf.rdf), color='g', marker='x')
            elif key == 'ai-w' and hasattr(self, 'iod'):
                ax.scatter(self.iod['anion'], np.max(rdf.rdf), color='g', marker='x')

            ax.legend(frameon=False)
            ax.set_xlim(0,10)
            axes.append(ax)
            plt.savefig(f'{key}.png', dpi=300)

        return axes