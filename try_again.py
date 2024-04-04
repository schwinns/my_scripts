# Script to try batch submission of umbrellas again

import subprocess
from glob import glob

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


if __name__ == '__main__':

    n_sims = 16

    for i in range(n_sims):
        if len(glob(f'./COLVAR_{i}')) == 0:
            run(f'sbatch submit_umbrella_{i}.job')