# Script to convert coordinate files with MDAnalysis

import MDAnalysis as mda
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='file to convert')
parser.add_argument('-o', '--output', help='name of the output file')
args = parser.parse_args()

U = mda.Universe(args.file)

# for ts in U.trajectory:
#     with mda.Writer(args.output) as W:
#         W.write(U.atoms)