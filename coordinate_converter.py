# Script to convert between different coordinate files with Open Babel or MD Analysis

import argparse
from openbabel import pybel
import MDAnalysis as mda

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input file to convert')
parser.add_argument('-o', '--output', required=True, help='output file')
parser.add_argument('-n', '--names', help='yaml file mapping atom types in input to atom names')
args = parser.parse_args()

in_file = args.input
in_ext = in_file.split('.')[-1]

out_file = args.output
out_ext = out_file.split('.')[-1]

try:
    print('Trying OpenBabel...')
    mols = pybel.readfile(in_ext, in_file)
    out = pybel.Outputfile(out_ext, out_file, overwrite=True)

    for mol in mols:
        out.write(mol)

except:
    print('Trying MDAnalysis...')
    u = mda.Universe(in_file)
    try:
        u.atoms.names
    except:
        if args.names is None:
            raise TypeError('Please provide a yaml mapping atom types to atom names')

        import yaml
        with open(args.names, 'r') as file:
            names_dict = yaml.safe_load(file)

        names = [names_dict[t] for t in u.atoms.types]
        u.add_TopologyAttr('names', names)
    
    u.atoms.write(args.output)    


print(f'Converted {in_file} to {out_file}!')