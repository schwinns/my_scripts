# Script to optimize the switching function parameter for coordination number analysis

import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.integrate import simpson
from scipy.optimize import minimize
import subprocess
from textwrap import dedent

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def write_plumed(options, filename='plumed.dat'):
    '''Write plumed.dat file for metadynamics simulation'''

    f = dedent(f'''\
    water_group: GROUP ATOMS=1-3000:3   # oxygen atom of the water molecules
    n: COORDINATION GROUPA=3001 GROUPB=water_group SWITCH={{Q REF={options['R_0']} BETA={options['a']} LAMBDA=1 R_0={options['R_0']}}}

    t: MATHEVAL ARG=n FUNC=1000-x PERIODIC=NO

    PRINT STRIDE=10 ARG=* FILE=COLVAR
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def run_plumed(plumed_input, traj, dt=0.002, stride=250, output='COLVAR'):
    '''Run plumed driver on plumed input file for a given trajectory (as an xtc) and read output COLVAR'''
    cmd = f'plumed driver --plumed {plumed_input} --ixtc {traj} --timestep {dt} --trajectory-stride {stride}'
    # cmd = f'plumed driver --plumed {plumed_input} --igro {traj}'
    run(cmd)

    COLVAR = np.loadtxt(output, comments='#')
    return COLVAR


def objective_func(a, options, traj, CN):
    '''
    Objective function for the minimization
    
    a : array-like, parameters for the switching function
    options : dict, options for the plumed input file
    traj : str, name of the standard MD trajectory
    CN : float, coordination number calculated from the integrated RDF at the cutoff
    '''

    options['a'] = a
    dat = write_plumed(options)
    COLVAR = run_plumed(dat, traj)
    run('rm COLVAR')
    n = COLVAR.shape[0]

    print(f'myPLUMED: Mean CN = {COLVAR[:,2].mean()}')

    return np.abs(COLVAR[:,2].mean() - CN) # absolute distance
    # return 1/n * np.sum( (COLVAR[:,2].mean() - CN)**2 ) # mean squared error


def func(x, options, gro, tol):
    '''Function to solve numerically
    
    Will solve func(a) = 0, where func(a) = switching function with parameter a at r=0.455 - tol
    '''

    options['a'] = x[0]
    dat = write_plumed(options)
    COLVAR = run_plumed(dat, gro)
    return COLVAR[1] - tol



if __name__ == '__main__':

    # inputs
    xtc = '../prod.xtc'
    tpr = '../mda_readable.tpr'
    ion_sel = 'resname NA'
    plumed_input = 'plumed.dat'
    dt = 0.002
    stride = 250

    options = {
        'a' : 0.4,
        'R_0' : 0.315,
        'SIGMA' : 0.05,
        'HEIGHT' : 10,
        'PACE' : 250,
        'TEMP' : 300,
        'BIASFACTOR' : 20
    }

    # calculate the "correct" CN value from the integrated RDF

    ### load in universe and select groups
    u = mda.Universe(tpr, xtc)

    ##### solute atoms
    ion = u.atoms.select_atoms(ion_sel)

    ##### solvent atoms
    water = u.atoms.select_atoms('resname SOL')
    OW = water.select_atoms('name OW')
    HW = water.select_atoms('name HW*')

    ### calculate the RDF between ion and water
    bin_width = 0.05 # Angstroms
    n_bins = int((20 - 0) / bin_width)

    ion_OW_rdf = rdf.InterRDF(ion, OW, nbins=n_bins, range=(0.0, 20.0), norm='rdf')
    ion_OW_rdf.run()

    ### calculate the density of OW around ion and use n(r) = rho*g(r) to calculate density rho
    n_r = rdf.InterRDF(ion, OW, nbins=n_bins, range=(0.0, 20.0), norm='density')
    n_r.run()
    rho = (n_r.results.rdf / ion_OW_rdf.results.rdf)[-1]

    ### numerically integrate the g(r) at different values of r to get G(r)
    n_bins = len(ion_OW_rdf.results.bins)
    G_r = np.zeros(n_bins)
    for i in range(1,n_bins):
        r = ion_OW_rdf.results.bins[:i]
        g_r = ion_OW_rdf.results.rdf[:i]
        y = 4*np.pi*np.power(r,2)*g_r
        G_r[i] = simpson(y, r)

    ### get the coordination number N(r) from G(r) and get CN for cutoff distance
    N_r = rho*G_r
    CN = N_r[ion_OW_rdf.results.bins > options['R_0']*10 - bin_width/2][:2].mean()
    print(f'The coordination number from the integrated RDF: {CN}')

    # fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
    # ax3.plot(ion_OW_rdf.results.bins, N_r, label='N(r)')
    # ax3.set_xlabel('r (A)')
    # ax3.set_ylabel('N(r)')
    # ax3.set_xlim(0,5)
    # ax3.set_ylim(0,20)
    # ax3.legend()

    # run optimization on the parameters of the switching function
    # from scipy.optimize import fsolve
    # result = fsolve(func, 0.1, args=(options, '../r_0.455nm.gro', 10**-6))
    # CN = 5.7883085292571295

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective_func, bracket=(-800,-600), args=(options, xtc, CN))
    # result = minimize(objective_func, [-500], args=(options, xtc, CN), tol=10**-8)
    
    # get the final CN value from plumed
    COLVAR = run_plumed('plumed.dat', xtc)
    CN = COLVAR[:,2].mean()
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].scatter(COLVAR[:,0]/1000, COLVAR[:,2], s=1)
    ax[0].set_xlabel('time (ns)', fontsize=14)
    ax[0].set_ylabel('CN', fontsize=14)
    ax[0].set_yticks(np.arange(0,13, step=1))
    ax[0].axhline(CN, ls='dashed', c='r')
    ax[0].text((COLVAR[:,0]/1000).mean(), 9, f'CN = {CN:.3f}')

    ax[1].hist(COLVAR[:,2], bins=50, edgecolor='k')
    ax[1].set_ylabel('counts', fontsize=14)
    ax[1].set_xlabel('CN', fontsize=14)
    ax[1].axvline(CN, ls='dashed', c='r')
    plt.savefig('tmp.png')
    plt.show()

    print(f'\nMinimization results:\n{result}')

    print(f'\nOptimal value for a = {result.x}')
    print(f'The coordination number from the switching function: {CN}')
