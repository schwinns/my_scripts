# Script to run production in python

import subprocess
import argparse

def run(commands, cwd=None):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, cwd=cwd)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False, **kwargs):
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
        run(cmd, **kwargs)
    return tpr


def mdrun(tpr, output=None, gmx='gmx', flags={}, dry_run=False, **kwargs):
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
        run(cmd, **kwargs)
    return output + '.gro'


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coords', type=str, help='Coordinate file')
    parser.add_argument('-p', '--top', type=str, help='Topology file')
    parser.add_argument('-f', '--mdp', type=str, help='MDP file')
    args = parser.parse_args()

    gmx = lambda n : f'mpirun -np {n} gmx_mpi' # GROMACS callable
    input_gro = args.coords # initial gro file
    input_top = args.top # topology for the system
    min_file = args.mdp

    # minimization
    min_tpr = grompp(input_gro, min_file, top=input_top, tpr=f'min.tpr', gmx=gmx(1), flags={'maxwarn' : 1}, cwd='./minimization/')
    min_gro = mdrun(min_tpr, output=f'min', gmx=gmx(128), cwd='./minimization/')