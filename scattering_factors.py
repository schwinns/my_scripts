# Contains all the scattering factor coefficients and functions to calculate the analytical approximation of the scattering factor

# Based on Debyer C functions in atomtables.c and atomtables.h

# Directly copied from Debyer atomtables.h
'''
/* Coefficients for approximation to the scattering factor called IT92.
* AFAIR the data was taken from ObjCryst. It can be also found
* in cctbx and in old atominfo program, and originaly comes from the paper
* publication cited below.
*/
/*
   International Tables for Crystallography
     Volume C
     Mathematical, Physical and Chemical Tables
     Edited by A.J.C. Wilson
     Kluwer Academic Publishers
     Dordrecht/Boston/London
     1992


   Table 6.1.1.4 (pp. 500-502)
     Coefficients for analytical approximation to the scattering factors
     of Tables 6.1.1.1 and 6.1.1.3


   [ Table 6.1.1.4 is a reprint of Table 2.2B, pp. 99-101,
     International Tables for X-ray Crystallography, Volume IV,
     The Kynoch Press: Birmingham, England, 1974.
     There is just one difference, see "Tl3+".
   ]
*/
'''

import numpy as np

def calculate_avg_sf(atom_group, scattering_factors):
    '''Calculate the average scattering factor from an MDAnalysis AtomGroup'''

    ag_elements = []
    [ag_elements.append(el) for el in atom_group.elements if el not in ag_elements]

    c = 0
    sf_sum = 0
    for el in ag_elements:
        count = len(atom_group.select_atoms(f'element {el}'))
        c += count
        sf_sum += count*scattering_factors.calculate_it92_factor(el)

    return sf_sum / c


class ScatteringFactors:
    def __init__(self):
        
        self.it92_coeff = {}
        self._parse_table('/home/nate/pkgs/my_scripts/atomtable.dat')


    def _parse_table(self, filename):
        '''Parse a file with the copied C data structure containing it92 coefficients'''

        dat = open(filename, 'r')

        lines = dat.readlines()

        for i,line in enumerate(lines):

            l = [strng.strip('}{,"') for strng in line.split() if len(strng.strip("}{,")) > 0]
            
            if len(l) == 5: # first line of scattering factor with element and a coefficients
                element = l[0]
                self.it92_coeff[element] = {}
                self.it92_coeff[element]['a'] = np.array([float(x) for x in l[1:]])
            elif len(l) == 4: # second line with b coefficients
                self.it92_coeff[element]['b'] = np.array([float(x) for x in l])
            elif len(l) == 1: # third line with c coefficient
                self.it92_coeff[element]['c'] = float(l[0])
            elif len(l) > 0:
                raise TypeError(f'Cannot parse line: {line}')
            
      
    def calculate_it92_factor(self, element, stol2=0):
        '''
        Calculate the scattering factor from the it92 coefficients
        
        stol2 is a variable used by Debyer, means (sin(theta) / lambda)^2 = (Q / 4pi)^2
        For the RDF calculation, Q is 0 so stol2 will always be 0
        If we were to calculate the diffraction pattern, we would need scattering factors for a range of Q
        '''

        if element == 'M':
            scattering_factor = 0
        else:
            scattering_factor = self.it92_coeff[element]['c']
            for i in range(4):
                scattering_factor += self.it92_coeff[element]['a'][i] * np.exp(-self.it92_coeff[element]['b'][i] * stol2)

        return scattering_factor


# PULLED STRAIGHT FROM MDANALYSIS
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis import distances
import warnings


class WeightedRDF(AnalysisBase):
    r"""Radial distribution function

    :class:`InterRDF` is a tool to calculate average radial distribution
    functions between two groups of atoms. Suppose we have two AtomGroups ``A``
    and ``B``. ``A`` contains atom ``A1``, ``A2``, and ``B`` contains ``B1``,
    ``B2``. Given ``A`` and ``B`` to :class:`InterRDF`, the output will be the
    average of RDFs between ``A1`` and ``B1``, ``A1`` and ``B2``, ``A2`` and
    ``B1``, ``A2`` and ``B2``. A typical application is to calculate the RDF of
    solvent with itself or with another solute.

    The :ref:`radial distribution function<equation-gab>` is calculated by
    histogramming distances between all particles in `g1` and `g2` while taking
    periodic boundary conditions into account via the minimum image
    convention.

    The `exclusion_block` keyword may be used to exclude a set of distances
    from the calculations.

    Results are available in the attributes :attr:`results.rdf`
    and :attr:`results.count`.

    Parameters
    ----------
    g1 : AtomGroup
        First AtomGroup
    g2 : AtomGroup
        Second AtomGroup
    nbins : int
        Number of bins in the histogram
    range : tuple or list
        The size of the RDF
    norm : str, {'rdf', 'density', 'none'}
          For 'rdf' calculate :math:`g_{ab}(r)`. For
          'density' the :ref:`single particle density<equation-nab>`
          :math:`n_{ab}(r)` is computed. 'none' computes the number of
          particles occurences in each spherical shell.

          .. versionadded:: 2.3.0

    exclusion_block : tuple
        A tuple representing the tile to exclude from the distance array.
    exclude_same : str
        Will exclude pairs of atoms that share the same "residue", "segment", or "chain".
        Those are the only valid values. This is intended to remove atoms that are
        spatially correlated due to direct bonded connections.
    verbose : bool
        Show detailed progress of the calculation if set to `True`

    Attributes
    ----------
    results.bins : numpy.ndarray
       :class:`numpy.ndarray` of the centers of the `nbins` histogram
       bins.

       .. versionadded:: 2.0.0

    bins : numpy.ndarray
       Alias to the :attr:`results.bins` attribute.

       .. deprecated:: 2.0.0
           This attribute will be removed in 3.0.0.
           Use :attr:`results.bins` instead.

    results.edges : numpy.ndarray

      :class:`numpy.ndarray` of the `nbins + 1` edges of the histogram
      bins.

       .. versionadded:: 2.0.0

    edges : numpy.ndarray

       Alias to the :attr:`results.edges` attribute.

       .. deprecated:: 2.0.0
           This attribute will be removed in 3.0.0.
           Use :attr:`results.edges` instead.

    results.rdf : numpy.ndarray
      :class:`numpy.ndarray` of the :ref:`radial distribution
      function<equation-gab>` values for the :attr:`results.bins`.

       .. versionadded:: 2.0.0

    rdf : numpy.ndarray
       Alias to the :attr:`results.rdf` attribute.

       .. deprecated:: 2.0.0
           This attribute will be removed in 3.0.0.
           Use :attr:`results.rdf` instead.

    results.count : numpy.ndarray
      :class:`numpy.ndarray` representing the radial histogram, i.e.,
      the raw counts, for all :attr:`results.bins`.

       .. versionadded:: 2.0.0

    count : numpy.ndarray
       Alias to the :attr:`results.count` attribute.

       .. deprecated:: 2.0.0
           This attribute will be removed in 3.0.0.
           Use :attr:`results.count` instead.

    Example
    -------
    First create the :class:`InterRDF` object, by supplying two
    AtomGroups then use the :meth:`run` method ::

      rdf = InterRDF(ag1, ag2)
      rdf.run()

    Results are available through the :attr:`results.bins` and
    :attr:`results.rdf` attributes::

      plt.plot(rdf.results.bins, rdf.results.rdf)

    The `exclusion_block` keyword allows the masking of pairs from
    within the same molecule. For example, if there are 7 of each
    atom in each molecule, the exclusion mask ``(7, 7)`` can be used.


    .. versionadded:: 0.13.0

    .. versionchanged:: 1.0.0
       Support for the `start`, `stop`, and `step` keywords has been
       removed. These should instead be passed to :meth:`InterRDF.run`.

    .. versionchanged:: 2.0.0
       Store results as attributes `bins`, `edges`, `rdf` and `count`
       of the `results` attribute of
       :class:`~MDAnalysis.analysis.AnalysisBase`.
    """
    def __init__(self,
                 g1,
                 g2,
                 nbins=75,
                 range=(0.0, 15.0),
                 norm="rdf",
                 exclusion_block=None,
                 exclude_same=None,
                 scattering_factors=None,
                 **kwargs):
        super(WeightedRDF, self).__init__(g1.universe.trajectory, **kwargs)
        self.g1 = g1
        self.g2 = g2
        self.norm = str(norm).lower()

        self.scattering_factors = scattering_factors

        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block
        if exclude_same is not None and exclude_same not in ['residue', 'segment', 'chain']:
            raise ValueError(
                "The exclude_same argument to InterRDF must be None, 'residue', 'segment' "
                "or 'chain'."
            )
        if exclude_same is not None and exclusion_block is not None:
            raise ValueError(
                "The exclude_same argument to InterRDF cannot be used with exclusion_block."
            )
        name_to_attr = {'residue': 'resindices', 'segment': 'segindices', 'chain': 'chainIDs'}
        self.exclude_same = name_to_attr.get(exclude_same)

        if self.norm not in ['rdf', 'density', 'none']:
            raise ValueError(f"'{self.norm}' is an invalid norm. "
                             "Use 'rdf', 'density' or 'none'.")

    def _prepare(self):
        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self.results.count = count
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        if self.norm == "rdf":
            # Cumulative volume for rdf normalization
            self.volume_cum = 0
        # Set the max range to filter the search radius
        self._maxrange = self.rdf_settings['range'][1]

    def _single_frame(self):
        pairs, dist = distances.capped_distance(self.g1.positions,
                                                self.g2.positions,
                                                self._maxrange,
                                                box=self._ts.dimensions)
        # Maybe exclude same molecule distances
        if self._exclusion_block is not None:
            idxA = pairs[:, 0]//self._exclusion_block[0]
            idxB = pairs[:, 1]//self._exclusion_block[1]
            mask = np.where(idxA != idxB)[0]
            dist = dist[mask]
            pairs = pairs[mask]

        if self.exclude_same is not None:
            # Ignore distances between atoms in the same attribute
            attr_ix_a = getattr(self.g1, self.exclude_same)[pairs[:, 0]]
            attr_ix_b = getattr(self.g2, self.exclude_same)[pairs[:, 1]]
            mask = np.where(attr_ix_a != attr_ix_b)[0]
            dist = dist[mask]
            pairs = pairs[mask]

        # new section to account for scattering lengths
        all_atoms = self.g1.union(self.g2)
        avg = calculate_avg_sf(all_atoms, self.scattering_factors)
        self._ts.dimensions[2] = all_atoms.bbox()[1,2] - all_atoms.bbox()[0,2]

        print(f'\tCalculating weights for frame {self._ts}...')
        weights = np.empty(pairs.shape[0])
        for i, (ai, aj) in enumerate(pairs):
            eli = self.g1.atoms[ai].element
            elj = self.g2.atoms[aj].element
            weights[i] = self.scattering_factors.calculate_it92_factor(eli)*self.scattering_factors.calculate_it92_factor(elj) / avg**2

        self.rdf_settings['weights'] = weights
        count, _ = np.histogram(dist, **self.rdf_settings)
        self.results.count += count

        if self.norm == "rdf":
            self.volume_cum += self._ts.volume

    def _conclude(self):
        norm = self.n_frames
        if self.norm in ["rdf", "density"]:
            # Volume in each radial shell
            vols = np.power(self.results.edges, 3)
            norm *= 4/3 * np.pi * np.diff(vols)

        if self.norm == "rdf":
            # Number of each selection
            nA = len(self.g1)
            nB = len(self.g2)
            N = nA * nB

            # If we had exclusions, take these into account
            if self._exclusion_block:
                xA, xB = self._exclusion_block
                nblocks = nA / xA
                N -= xA * xB * nblocks

            # Average number density
            box_vol = self.volume_cum / self.n_frames
            norm *= N / box_vol

        self.results.rdf = self.results.count / norm

    @property
    def edges(self):
        wmsg = ("The `edges` attribute was deprecated in MDAnalysis 2.0.0 "
                "and will be removed in MDAnalysis 3.0.0. Please use "
                "`results.bins` instead")
        warnings.warn(wmsg, DeprecationWarning)
        return self.results.edges

    @property
    def count(self):
        wmsg = ("The `count` attribute was deprecated in MDAnalysis 2.0.0 "
                "and will be removed in MDAnalysis 3.0.0. Please use "
                "`results.bins` instead")
        warnings.warn(wmsg, DeprecationWarning)
        return self.results.count

    @property
    def bins(self):
        wmsg = ("The `bins` attribute was deprecated in MDAnalysis 2.0.0 "
                "and will be removed in MDAnalysis 3.0.0. Please use "
                "`results.bins` instead")
        warnings.warn(wmsg, DeprecationWarning)
        return self.results.bins

    @property
    def rdf(self):
        wmsg = ("The `rdf` attribute was deprecated in MDAnalysis 2.0.0 "
                "and will be removed in MDAnalysis 3.0.0. Please use "
                "`results.rdf` instead")
        warnings.warn(wmsg, DeprecationWarning)
        return self.results.rdf
