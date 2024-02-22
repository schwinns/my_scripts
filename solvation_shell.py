# Script to calculate shell cutoffs and most probable shells from equilibrium simulations

import MDAnalysis as mda
import matplotlib.pyplot as plt
import argparse

from solvation_analysis.solute import Solute
from scipy.signal import find_peaks

# wrapper to determine the first minimum in the RDF, more robust than default
def find_peaks_wrapper(bins, data, **kwargs):
    peaks, _  = find_peaks(-data, **kwargs)
    radii = bins[peaks[0]]
    return radii


# command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--top', type=str, default='mda_readable.tpr', help='TPR or other topology for MDAnalysis')
parser.add_argument('-x', '--traj', type=str, default='prod.xtc', help='XTC or other trajectory for MDAnalysis')
parser.add_argument('-ci', '--cation', type=str, default='resname NA', help='MDAnalysis selection language for the cations')
parser.add_argument('-ai', '--anion', type=str, default='resname CL', help='MDAnalysis selection language for the anions')
parser.add_argument('-s', '--step', type=int, default=1, help='trajectory step for calculations')
parser.add_argument('--plot', type=bool, default=False, help='whether to plot and show results')
args = parser.parse_args()

# hard-coded inputs
water_selection = 'resname SOL'
water_oxygen_selection = 'type OW'

# initialize universe and make selections
u = mda.Universe(args.top, args.traj)

# solute atoms
ci = u.atoms.select_atoms(args.cation)
ai = u.atoms.select_atoms(args.anion)

# solvent atoms
water = u.atoms.select_atoms('resname SOL')
OW = water.select_atoms('name OW')


# run SolvationAnalysis on the cation and anion
cation = Solute.from_atoms(ci, {'water' : OW, 'coion' : ai}, solute_name='Cation', 
                              rdf_kernel=find_peaks_wrapper, kernel_kwargs={'distance':5})
anion = Solute.from_atoms(ai, {'water' : OW, 'coion' : ci}, solute_name='Anion',
                              rdf_kernel=find_peaks_wrapper, kernel_kwargs={'distance':5})

cation.run(step=args.step)
anion.run(step=args.step)

# combine the speciation results into shells for each anion and cation
df1 = cation.speciation.speciation_fraction

shell = []
for i in range(df1.shape[0]):
    row = df1.iloc[i]
    shell.append(f'{row.coion:.0f}-{row.water:.0f}')

df1['shell'] = shell

df2 = anion.speciation.speciation_fraction

shell = []
for i in range(df2.shape[0]):
    row = df2.iloc[i]
    shell.append(f'{row.coion:.0f}-{row.water:.0f}')

df2['shell'] = shell

# output the results
print(f"\nHydration shell cutoff for cation-water = {cation.radii['water']:.6f}")
print(f"Hydration shell cutoff for anion-water = {anion.radii['water']:.6f}")
print(f"\nHighest probability cation shell (anions-waters): {df1.iloc[df1['count'].argmax()].shell}")
print(f"Highest probability anion shell (cations-waters): {df2.iloc[df2['count'].argmax()].shell}")
print(f"\nCoordination number for waters around cation = {cation.coordination.coordination_numbers['water']:.6f}")
print(f"Coordination number for waters around anion = {anion.coordination.coordination_numbers['water']:.6f}")

if args.plot:
    for solvent in cation.solvents.keys():
        cation.plot_solvation_radius('Cation', solvent)

    for solvent in anion.solvents.keys():
        anion.plot_solvation_radius('Anion', solvent)

    df = df1.merge(df2, on='shell', how='outer')
    df.plot(x='shell', y=['count_x', 'count_y'], kind='bar', legend=False)
    plt.legend(['Cation', 'Anion'])
    plt.ylabel('probability')

    