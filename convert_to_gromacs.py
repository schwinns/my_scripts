# Script to convert to GROMACS

from polym_analysis import PolymAnalysis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='LAMMPS data file to convert to Gromacs')
args = parser.parse_args()

data = PolymAnalysis(args.data)

data._reassign_atom_numbers('min_renumbered.data')

mapping = {
    1 : 'c',
    2 : 'ca',
    3 : 'ha',
    4 : 'hn',
    5 : 'ho',
    6 : 'cl',
    7 : 'n',
    8 : 'nh',
    9 : 'o',
    10 : 'oh',
    11 : 'OW',
    12 : 'HW'
}

data._atomtype_mapping(mapping)
data.write_GROMACS(output=args.data.split('.data')[0])