# Class to extract clusters from MD simulations and calculate EXAFS spectra with FEFF

import numpy as np
import pandas as pd
from textwrap import dedent
from time import perf_counter as time
import subprocess

from ase.data import atomic_numbers
from MDAnalysis.analysis.base import Results

from larch import xafs
from ParallelMDAnalysis import ParallelAnalysisBase


def run(commands : list | str):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        out = subprocess.run(cmd, shell=True)

    return out


def list2str(lst):
    '''Convert a list to a delimited string'''
    lst = [str(i) for i in lst]
    return ' '.join(lst)


def write_feff(potentials, atoms, title='FEFF8.5 input',  edge='K', s02=1.0, kmax=20, 
               print_vals=[0,0,0,0,0,3], control=[1,1,1,1,1,1], exchange=[0,0.0,0.0], 
               cfaverage=[0,1,0], scf=[6,0,30,0.2], ion=[], tdlda=0, debye=[], rpath=None,
               nleg=8, criteria=[4,2.5], filename='feff.inp'):
    '''
    Write a FEFF8.5 input file. This function writes a typical FEFF input to generate EXAFS chi(k).
    More advanced options are not included here. Defaults are taken from the defaults for FEFF. The 
    exception is SCF, which uses a default rfms1 from the Pappalardo et al. 2021 example file.

    Pappalardo et al. 2021: https://doi.org/10.1021/acs.inorgchem.1c01888
    
    Parameters
    ----------
    potentials : list
        List of POTENTIALS lines, expects {ipot: >3} {atomic_number: >8} {tag: >8}\n
    atoms : list
        List of ATOMS lines, expects {pos_x: .8f} {pos_y: .8f} {pos_z: .8f} {ipot: <8} {tag: <8}\n for each line
    title : str
        TITLE for the file, default='FEFF8.5 input'
    edge : str
        Which EDGE to calculate (i.e. K, L1, L2, etc.), default='K'
    s02 : float
        Amplitude reduction factor, default=1.0
    kmax : float
        Maximum value of k to calculate, default=20
    print_vals : list
        List of length 6 with the output options, default=[0,0,0,0,0,3]
    control : list
        List of length 6 with the modules to run, default=[1,1,1,1,1,1] means run all
    exchange : list
        List of length 3 with [ixc, vr0, vi0] index of potential model, real, and imagninary corrections,
        default=[0,0.0,0.0]
    cfaverage : list
        List of length 3 with [iphabs, nabs, rclabs] potential index, average over nabs absorbers, radius
        of smaller atoms list, default=[0,1,0] means no configurational average
    scf : list
        List of length 4 with [rfms1,lfms1,nscmt,ca], default=[6,0,30,0.2]
    ion : list
        List of length 0 or 2 with [ipot,ionization], default=[] means no ionization
    tdlda : int
        Use time-dependent local density approximation to account for screening, options are 0 (static)
        and 1 (dynamic), default=0
    debye : list
        List of length 2 with [temperature, Debye-temperature], default=[] does not calculate the Debye-Waller factors
    rpath : float or None
        Maximum effective distance, default=None means use 2.2 times nearest neighbor distance
    nleg : int
        Limit the number of legs of each scattering path, default=8
    criteria : list
        List of length 2 with [critcw,critpw], default=[4,2.5]
    filename : str
        Name of the input file to write, default='feff.inp'

    Returns
    -------
    inp : str
        Name of the FEFF input file

    '''

    # decide whether to comment out sections
    if len(ion) == 0:
        ion_comment = '* '
    else:
        ion_comment = ''

    if rpath is None:
        rpath_comment = '* '
    else:
        rpath_comment = ''

    if len(debye) == 0:
        debye_comment = '* '
    else:
        debye_comment = ''

    f = dedent(f'''\
        TITLE     {title}

        EDGE      {edge}
        EXAFS     {kmax}
        S02       {s02}

        *         pot xsph fms path genfmt ff2x
        CONTROL   {list2str(control)}
        PRINT     {list2str(print_vals)}

        *          ixc vr0 vi0 ixc0
        EXCHANGE  {list2str(exchange)}

        *         iphabs nabs rclabs
        CFAVERAGE {list2str(cfaverage)}

        *         rfms1 lfms1 nscmt ca
        SCF       {list2str(scf)}

        *         ipot ionization
        {ion_comment}ION       {list2str(ion)}

        *         ixfc
        TDLDA     {tdlda}

        *         temperature Debye-temperature
        {debye_comment}DEBYE    {list2str(debye)}

        *         rpath
        {rpath_comment}RPATH     {rpath}

        *         nleg
        NLEG      {nleg}

        *         critcw critpw
        CRITERIA  {list2str(criteria)}        
        
    ''')

    inp = open(filename, 'w')
    inp.write(f) # write all the settings information above

    # now write POTENTIALS section
    inp.write('POTENTIALS\n')
    inp.write('* ipot    Z        tag      lmax1      lmax2      xnatph      spinph\n')
    inp.writelines(potentials)

    # now write ATOMS section
    inp.write('\nATOMS\n')
    inp.write('* x           y           z          ipot     tag\n')
    inp.writelines(atoms)

    # END
    inp.write('END')

    inp.close()

    return filename


def load_feff(chi_file):
    data = np.loadtxt(chi_file, comments='#')
    feff = pd.DataFrame(data, columns=['k', 'chi', 'mag', 'phase'])
    feff['k2chi'] = feff.k**2 * feff.chi
    return feff


class EXAFS(ParallelAnalysisBase):
    '''
    Parallelized EXAFS analysis class. Performs a simple parallelization over the frames of a trajectory.
    
    Parameters
    ----------
    absorber : MDAnalysis.AtomGroup
        AtomGroup containing the absorber atom, i.e. the atom around which to calculate the EXAFS.
    write_feff_kwargs : dict
        Dictionary of keyword arguments to pass to the `write_feff` function, which writes the FEFF input file.
        This can include parameters like `edge`, `s02`, `kmax`, etc. See the `write_feff` function for details.
    **kwargs : dict
        Additional keyword arguments passed to the base class.

    Results are available through the :attr:`results`.
    
    '''
    def __init__(self, absorber, write_feff_kwargs={}, **kwargs):
        super(EXAFS, self).__init__(absorber.universe.trajectory, **kwargs)
        self.absorber = absorber[0] # ensure we have a single atom
        self.u = absorber.universe

        self.feff_settings = write_feff_kwargs


    def _prepare(self):

        self.results = Results()
        self.results._per_frame = {}

        # save information for timing
        self.start_time = time()
        self.frame_times = []

    def _single_frame(self, frame_idx):
        self._frame_index = frame_idx
        self._ts = self.u.trajectory[frame_idx]

        frame_start = time()

        cluster = self.u.select_atoms(f'sphzone 5 index {self.absorber.index}') - self.absorber # get atoms in a 5 A sphere around the cation

        # first, create a dictionary of unique potential indices
        ipots = {}
        for atom in cluster:
            if not atom.element in ipots.keys():
                ipots[atom.element] = len(ipots.keys())+1

        # write the POTENTIAL lines with indices, tags, and atomic number
        potential_lines = []
        absorber_line = f"{0: >3} {atomic_numbers[self.absorber.element]: >8} {self.absorber.element+'0': >8}\n"
        potential_lines.append(absorber_line) # first, for the absorber
        for el,ipot in ipots.items():
            line = f'{ipot: >3} {atomic_numbers[el]: >8} {el: >8}\n'
            potential_lines.append(line)

        # write the ATOM lines with Cartesian coordinates, indices, tags
        atom_lines = []
        absorber_line = f' {self.absorber.position[0]: .8f} {self.absorber.position[1]: .8f} {self.absorber.position[2]: .8f} {0: <8} {self.absorber.element: <8}\n' # first, for the absorber
        atom_lines.append(absorber_line)
        for atom in cluster:
            line = f' {atom.position[0]: .8f} {atom.position[1]: .8f} {atom.position[2]: .8f} {ipots[atom.element]: <8} {atom.element: <8}\n'
            atom_lines.append(line)

        # write FEFF input and run
        run(f'mkdir ./frame{frame_idx:04d}')
        inp = write_feff(potential_lines, atom_lines, filename=f'./frame{frame_idx:04d}/feff.inp',
                         **self.feff_settings)
        feff = xafs.feff8l(folder=f'./frame{frame_idx:04d}', feffinp=inp)
        feff.run()

        # save results and time for this frame
        df = load_feff(f'./frame{frame_idx:04d}/chi.dat')
        self.results._per_frame[frame_idx] = df

        frame_end = time()
        self.frame_times.append(frame_end - frame_start)

        # return the results for this frame
        return df['k2chi'].values

    
    def _conclude(self):
        
        self.results.k = self.results._per_frame[0]['k'].values
        self.resuts.k2chi = np.zeros(self.results.k.shape)
        
        # take the average over all frames
        for res in self._result:
            self.results.k2chi += res

        self.results.k2chi /= len(self._result)

        # report timing
        self.end_time = time()
        print('\n')
        print('-'*20 + '  Timing  ' + '-'*20)
        for f,t in enumerate(self.frame_times):
            print(f'Frame {f:04d}'.ljust(25) + f'{t:.2f} s'.rjust(25))

        total_time = self.end_time - self.start_time # s
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f'\nTotal time:'.ljust(25) + f'{hours:d}:{minutes:d}:{seconds:d}'.rjust(25))
        print('-'*50)