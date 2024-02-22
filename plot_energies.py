# Script to properly order energy files and plot a given thermodynamic output

import subprocess
from glob import glob
from natsort import natsorted
import gromacs as gro
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--energy', nargs='+', help='Quantity from the energy file to plot (numerical selection)')
    parser.add_argument('-nvt', '--nvt', action='store_true', help='Whether this quantity is calculated during NVT')
    parser.add_argument('-npt', '--npt', action='store_true', help='Whether this quantity is calculated during NPT')
    parser.add_argument('-o', '--output', default='energies.png', help='Name for the output figure')
    parser.add_argument('-d', '--dir', help='Directory with edr energy files')
    args = parser.parse_args()

    # collect all edr files
    nvt_edr = glob(args.dir + '/nvt*.edr')
    npt_edr = glob(args.dir + '/npt*.edr')

    nvt_edr = natsorted(nvt_edr)
    npt_edr = natsorted(npt_edr)

    if len(nvt_edr) != len(npt_edr):
        nvt_edr = nvt_edr[:-1]

    # loop through and generate xvg files
    energies = {}
    for e in args.energy:
        energies[e] = []
    
    if args.nvt and args.npt:
        for nvt, npt in tqdm(zip(nvt_edr, npt_edr)):

            # add all data for NVT
            cmd = 'echo '
            for e in energies:
                cmd += f'{e} '
            
            cmd += f'| gmx energy -f {nvt} -o tmp.xvg'
            run(cmd)
            
            xvg = gro.fileformats.XVG('tmp.xvg')
            for i,e in enumerate(energies):
                [energies[e].append(a) for a in xvg.array[i+1,:]]
            
            run('rm tmp.xvg')

            # add all data for NPT
            cmd = 'echo '
            for e in energies:
                cmd += f'{e} '
            
            cmd += f'| gmx energy -f {npt} -o tmp.xvg'
            run(cmd)
            
            xvg = gro.fileformats.XVG('tmp.xvg')
            for i,e in enumerate(energies):
                [energies[e].append(a) for a in xvg.array[i+1,:]]
            
            run('rm tmp.xvg')

    elif args.nvt:
        for nvt in nvt_edr:
            
            # add all data for NVT
            cmd = 'echo '
            for e in energies:
                cmd += f'{e} '
            
            cmd += f'| gmx energy -f {nvt} -o tmp.xvg'
            run(cmd)
            
            xvg = gro.fileformats.XVG('tmp.xvg')
            for i,e in enumerate(energies):
                [energies[e].append(a) for a in xvg.array[i+1,:]]

            run('rm tmp.xvg')

    elif args.npt:
        for npt in npt_edr:
            
            # add all data for NPT
            cmd = 'echo '
            for e in energies:
                cmd += f'{e} '
            
            cmd += f'| gmx energy -f {npt} -o tmp.xvg'
            run(cmd)
            
            xvg = gro.fileformats.XVG('tmp.xvg')
            for i,e in enumerate(energies):
                [energies[e].append(a) for a in xvg.array[i+1,:]]

            run('rm tmp.xvg')

    else:
        raise TypeError('Please specify whether to use NVT, NPT, or all simulations with the -nvt, -npt arguments')

        
    # plot full time series
    time = np.arange(0, len(energies[args.energy[0]]))/1000*2

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    
    data = time.copy()
    for e in energies:
        data = np.vstack((data, np.array(energies[e])))
        plt.plot(time, energies[e])

    np.savetxt(args.output.split('.png')[0] + '.csv', data, delimiter=',', fmt='%.8f')

    plt.xlabel('time (ns)')
    plt.savefig(args.output)
    plt.show()