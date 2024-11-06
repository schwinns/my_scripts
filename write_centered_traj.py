# Function to center a trajectory on a given selection usign MDAnalysis

from tqdm import tqdm

import MDAnalysis as mda
import MDAnalysis.transformations as trans

def write_centered(top, traj, selection, output='centered.xtc'):
    '''
    Write trajectory centered on a given trajectory

    Parameters
    ----------
    top : str
        Topology file for the system, should be a .tpr or .gro file
    traj : str, list
        Trajectory file(s) for the system, should be a .xtc or .gro or list of those files
    selection : str
        MDAnalysis selection language for the centering group, all atoms will be centered on this selection
    output : str
        Name of the output file, default='centered.xtc'

    '''

    u = mda.Universe(top, traj)
    workflow = [trans.unwrap(u.atoms),
                trans.center_in_box(u.select_atoms(selection)),
                trans.wrap(u.select_atoms(selection))]

    u.trajectory.add_transformations(*workflow)

    with mda.Writer(output, u.atoms.n_atoms) as w:
        for ts in tqdm(u.trajectory):
            w.write(u.atoms)


if __name__ == '__main__':

    # get the index of the first cation
    u = mda.Universe('mda_readable.tpr')
    cations = u.select_atoms('resname NA')
    idx = cations[0].index
    print(f'Index of the first cation is {idx}')

    # center trajectory on first cation
    write_centered('mda_readable.tpr', 'prod_0.xtc', f'index {idx}', output='prod_0_centered.xtc')