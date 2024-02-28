# Script to calculate the diffusion coefficient from MD simulation

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.transformations import nojump
import MDAnalysis.analysis.msd as mdaMSD

import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

class DiffusionCoefficient:

    def __init__(self, top, traj):
        '''
        Class to calculate the diffusion coefficient from a trajectory.

        Parameters
        ----------
        top : str
            Name of the topology file (e.g. gro, tpr, pdb)
        traj : str
            Name of the trajectory file (e.g. xtc, trr, dcd)
        
        '''

        # Load in underlying MDAnalysis Universe
        # Apply the nojump transformation, which is necessary for MSD
        self.universe = mda.Universe(top, traj, transformations=nojump.NoJump())
        

    def run(self, atom_group, n_bootstraps=0, confidence=0.95, msd_kwargs={}, fit_kwargs={}):
        '''
        Run the calculation to get the MSD and perform the linear fit.

        Parameters
        ----------
        atom_group : str or MDAnalysis.AtomGroup
            Atoms to calculate the diffusion coefficient for
        n_bootstraps : int
            Number of bootstraps to perform to calculate error in the diffusion coefficient, default=0
        confidence : float
            Confidence interval for bootstrapping, should be [0,1), default=0.95
        msd_kwargs : dict
            Keyword arguments for the DiffusionCoefficient.msd method. 
            Options are 'msd_type', 'step'. Default={}
        fit_kwargs : dict
            Keyword arguments for the DiffusionCoefficient.fit_msd method. 
            Options are 'msds', 'start_idx', 'end_idx'. Default={}

        Returns
        -------
        D : float
            Diffusion coefficient for the selection
        ci : float
            Confidence interval estimate if bootstrapping is performed

        '''

        if isinstance(atom_group, str):
            ag = self.universe.select_atoms(atom_group)
        else:
            ag = atom_group

        # Calculate MSD and fit to get diffusion coefficient
        msd_ts, lagtimes = self.msd(ag, **msd_kwargs)
        self.fit_msd(**fit_kwargs)

        # boostrap to get a confidence interval
        if n_bootstraps > 0:
            self.msd_boots = np.zeros((msd_ts.shape[0], n_bootstraps))
            diff_boots = np.zeros(n_bootstraps)

            # bootstrap over randomly chosen particles with replacement
            for b in tqdm(range(n_bootstraps)):
                rng = np.random.default_rng()
                indices = rng.integers(0, len(ag.residues), len(ag.residues)) 
                self.msd_boots[:,b] = self.msds_by_particle[:,indices].mean(axis=1)
                diff_boots[b],_ = self.fit_msd(msds=self.msd_boots[:,b], **fit_kwargs)

            # calculate mean, standard error, and confidence interval for diffusion coefficient
            if n_bootstraps <= 30: # use t-distribution
                self.ci = stats.t.interval(confidence, n_bootstraps, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
                self.msd_ci = stats.t.interval(confidence, n_bootstraps, loc=self.msd_boots.mean(axis=1), scale=stats.sem(self.msd_boots, axis=1))
            else: # use normal distribution
                self.ci = stats.norm.interval(confidence, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
                self.msd_ci = stats.norm.interval(confidence, loc=self.msd_boots.mean(axis=1), scale=stats.sem(self.msd_boots, axis=1))

            self.msd_ts = self.msd_boots.mean(axis=1)
            self.msd_stderr = stats.sem(self.msd_boots, axis=1)
            self.D = diff_boots.mean()
            self.stderr = stats.sem(diff_boots)
            
            return self.D, self.ci
        
        else:
            return self.D


    def msd(self, atom_group, msd_type='xyz', step=1):
        '''
        Use MDAnalysis EinsteinMSD to calculate the MSD for a given group.

        Parameters
        ----------
        atom_group : MDAnalysis.AtomGroup
            Atom group to calculate MSD for
        msd_type : str
            Which dimensions to calculate the MSD for, default='xyz'
        step : int
            Trajectory step for which to run the MSD calculation, default=1

        Returns
        -------
        msd_ts : np.array
            Time series with the MSD averaged over all the particles with respect to lag-time
        lagtimes : np.array
            Lag-time in nanoseconds

        '''

        MSD = mdaMSD.EinsteinMSD(atom_group, msd_type=msd_type, fft=True)
        MSD.run(step=step)
    
        self.msds_by_particle = MSD.results.msds_by_particle / 10**2 # save the per particle MSDs for bootstrapping, converted to nm^2
        self.msd_ts = MSD.results.timeseries / 10**2 # convert to nm^2
        dt = self.universe.trajectory[step].time - self.universe.trajectory[0].time
        self.lagtimes = np.arange(MSD.n_frames)*dt / 1000 # convert to ns
        self.dimensionality = MSD.dim_fac
        self.msd_ci = (np.zeros(MSD.n_frames), np.zeros(MSD.n_frames))
    
        return self.msd_ts, self.lagtimes
    

    def fit_msd(self, msds=None, start_idx=-1, end_idx=-1):
        '''
        Fit the MSD vs time lags to extract the diffusion coefficient.

        Parameters
        ----------
        msds : np.array
            Mean squared displacement over the time lags for the linear fit, default=None means use
            previously calculated MSD saved in the object
        start_idx : int
            Starting index for the region to fit the MSD, default=-1 means start at 10% of the time lags,
            which is the same as Gromacs gmx msd defaults
        end_idx : int
            Ending index for the region to fit the MSD, default=-1 means end at 90% of the time lags,
            which is the same as Gromacs gmx msd defaults

        Returns
        -------
        D : float
            Diffusion coefficient for the calculated MSD
        stderr : float
            Standard error for D in the linear fit

        '''

        if msds is None:
            msds = self.msd_ts

        # get the region of the MSD to fit
        if start_idx == -1:
            val = np.quantile(self.lagtimes, 0.1)
            start_idx = np.where(self.lagtimes >= val)[0][0]
        if end_idx == -1:
            val = np.quantile(self.lagtimes, 0.9)
            end_idx = np.where(self.lagtimes <= val)[0][-1]

        # set up the matrices for the linear regression
        X = self.lagtimes[start_idx:end_idx]
        Y = msds[start_idx:end_idx]
        X = sm.add_constant(X)

        # fit the ordinary least squares model and get the slope
        ols = sm.OLS(Y, X)
        self.fit_results = ols.fit()
        b, m = self.fit_results.params
        be, me = self.fit_results.bse # standard error in parameters

        self.D = m / 2 / self.dimensionality
        self.stderr = me / 2 / self.dimensionality

        return self.D, self.stderr # nm^2 / ns = 1e-9 m^2 / s
    

    def plot_msd(self, fraction=0.5, filename='msd.dat'):
        '''
        Plot the MSD versus the time lag.

        Parameters
        ----------
        fraction : float
            Fraction of the MSD curve to show, should be between [0,1], default=0.5
        filename : str
            Filename for saving the data, default='msd.dat'

        Returns
        -------
        ax : matplotlib.axes.Axes

        '''

        end_idx = int(fraction * self.lagtimes.shape[0])

        fig, ax = plt.subplots(1,1)
        
        ax.plot(self.lagtimes[:end_idx], self.msd_ts[:end_idx])
        ax.fill_between(self.lagtimes[:end_idx], 
                        self.msd_ts[:end_idx]-self.msd_ci[0][:end_idx], 
                        self.msd_ts[:end_idx]+self.msd_ci[1][:end_idx], alpha=0.5)

        ax.set_xlabel('time lag (ns)')
        ax.set_ylabel('MSD (nm$^2$)')

        if filename is not None:
            np.savetxt(filename, np.vstack([self.lagtimes, self.msd_ts, self.msd_ci[0], self.msd_ci[1]]).T,
                       header='lag times (ns), MSD (nm^2), lower confidence bound (nm^2), upper confidence bound (nm^2)')

        return ax

    



        

        
