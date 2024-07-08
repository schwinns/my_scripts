# Script to slowly introduce the ion coordination bias (to prevent ions from entering hydration shell)

import numpy as np
import subprocess
from textwrap import dedent
import MDAnalysis as mda

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run grompp with mdp file on gro file with topology top
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if tpr is None:
        tpr = gro.split('.gro')[0] + '.tpr'
    cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {gro} -o {tpr}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return tpr


def mdrun(tpr, output=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run GROMACS with tpr file
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if output is None:
        output = tpr.split('.tpr')[0]
    cmd = [f'{gmx} mdrun -s {tpr} -deffnm {output}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return output + '.gro'


def write_plumed(KAPPA, options, filename='plumed.dat'):
    '''Write plumed input file for umbrella sampling simulation'''

    f = dedent(f'''\
    # groups from NDX file
    biased: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['biased_group']}
    waters: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['waters_group']}
    ions: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['ions_group']}

    # ion coordination
    n2: COORDINATION GROUPA=biased GROUPB=ions SWITCH={{Q REF={options['R_0']} BETA=-200 LAMBDA=1 R_0={options['R_0']}}} # very high switching function parameter ensures strict cutoff
    t2: MATHEVAL ARG=n2 FUNC={options['n_ions']}-x PERIODIC=NO
    r2: RESTRAINT ARG=t2 KAPPA={KAPPA} AT=0 # very high force constant prevents ion-ion coordination

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


if __name__ == '__main__':

    # Gromacs inputs
    equil_gro = 'prod.gro'
    top = 'solution.top'
    mdp = 'increment.mdp'
    ntasks = 16

    # create MDAnalysis groups for plumed and write to Gromacs ndx file
    u = mda.Universe('mda_readable.tpr', equil_gro)
    biased = u.select_atoms('resname NA')[0]
    waters = u.select_atoms('resname SOL and not element H')
    ions = u.select_atoms('resname NA CL') - biased 

    mda.AtomGroup([biased]).write('index.ndx', mode='w', name='biased')
    waters.write('index.ndx', mode='a', name='waters')
    ions.write('index.ndx', mode='a', name='ions')

    # PLUMED inputs
    KAPPAs = [1000, 2500, 5000, 10000]

    plumed_options = {
        'ndx'           : 'index.ndx',
        'biased_group'  : 'biased',
        'waters_group'  : 'waters',
        'ions_group'    : 'ions',
        'R_0'           : 0.315,
        'n_waters'      : len(waters),
        'n_ions'        : len(ions),
        'STRIDE'        : 10,
        'FILE'          : 'COLVAR' 
    }

    # increment through increasing KAPPA
    gro = equil_gro
    for k in KAPPAs:
        plumed_options['FILE'] = f'COLVAR_K{k}'
        plumed_input = write_plumed(k, plumed_options, filename=f'plumed_K{k}.dat')
        tpr = grompp(gro, mdp, top, tpr=f'run_K{k}.tpr', gmx='mpirun -np 1 gmx_mpi')
        gro = mdrun(tpr, output=f'run_K{k}', flags={'plumed' : f'plumed_K{k}.dat'}, gmx=f'mpirun -np {ntasks} gmx_mpi')