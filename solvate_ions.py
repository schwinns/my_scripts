# Script to run simulations of solvated ions

import subprocess
import argparse
from textwrap import dedent

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def update_topology(n_waters, n_cations, n_anions, cation='NA', anion='CL', filename='solution.top'):
    '''Update the top file with new numbers of waters and ions'''

    f = dedent(f'''\
    [ defaults ]
    ; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
    1               2               yes             0.5     0.8333

    [ atomtypes ]
    ;type, bondingtype, atomic_number, mass, charge, ptype, sigma, epsilon
    OW   OW     8     0.00000000  0.00000000  A     3.17427e-01  6.8369e-01
    HW   HW     1     0.00000000  0.00000000  A     0.00000e+00  0.00000e+00
    Li   Li  	3	6.94000000	0.00000000	A	2.26288274e-01	1.36251960e-02
    Na   Na  	11	22.99000000	0.00000000	A	2.61746043e-01	1.26028774e-01
    K    K   	19	39.10000000	0.00000000	A	3.03440103e-01	5.86672238e-01
    Rb   Rb  	37	85.47000000	0.00000000	A	3.20723539e-01	8.91730690e-01
    Cs   Cs  	55	132.91000000	0.00000000	A	3.50123196e-01	1.49632371e+00
    Tl   Tl  	81	204.38000000	0.00000000	A	2.23259219e-01	1.06570246e-02
    Cu   Cu  	29	63.55000000	0.00000000	A	2.07757581e-01	2.54399752e-03
    Ag   Ag  	47	107.87000000	0.00000000	A	2.28782791e-01	1.65544981e-02
    F    F   	9	19.00000000	0.00000000	A	3.23930774e-01	9.53762046e-01
    Cl   Cl  	17	35.45000000	0.00000000	A	4.10882489e-01	2.68724396e+00
    Br   Br  	35	79.90000000	0.00000000	A	4.45627539e-01	3.18294026e+00
    I    I   	53	126.90000000	0.00000000	A	4.95339687e-01	3.64743606e+00
    Be   Be  	4	9.01000000	0.00000000	A	1.66598060e-01	8.53536000e-06
    Ni   Ni  	28	58.69000000	0.00000000	A	2.10073918e-01	3.21502744e-03
    Pt   Pt  	78	195.08000000	0.00000000	A	2.12924794e-01	4.24537928e-03
    Zn   Zn  	30	65.40000000	0.00000000	A	2.13815692e-01	4.62034936e-03
    Co   Co  	27	58.93000000	0.00000000	A	2.19695624e-01	7.87449720e-03
    Pd   Pd  	46	106.42000000	0.00000000	A	2.20408343e-01	8.37582408e-03
    Cr   Cr  	24	52.00000000	0.00000000	A	2.16844748e-01	6.11382816e-03
    Fe   Fe  	26	55.85000000	0.00000000	A	1.99739493e-01	1.06478616e-03
    Mg   Mg  	12	24.30500000	0.00000000	A	2.32702745e-01	2.21841538e-02
    V    V   	23	50.94000000	0.00000000	A	2.33415464e-01	2.33573474e-02
    Mn   Mn  	25	54.94000000	0.00000000	A	2.41077193e-01	3.94048283e-02
    Hg   Hg  	80	200.59000000	0.00000000	A	2.41077193e-01	3.94048283e-02
    Cd   Cd  	48	112.41000000	0.00000000	A	2.41789912e-01	4.12549931e-02
    Yb   Yb  	70	173.05000000	0.00000000	A	2.73327727e-01	2.11920981e-01
    Ca   Ca  	20	40.08000000	0.00000000	A	2.74574985e-01	2.23042764e-01
    Sn   Sn  	50	118.71000000	0.00000000	A	2.77247681e-01	2.48150739e-01
    Pb   Pb  	82	207.20000000	0.00000000	A	2.89363904e-01	3.84128270e-01
    Eu   Eu  	64	151.96000000	0.00000000	A	2.69764132e-01	1.82199058e-01
    Sr   Sr  	38	87.62000000	0.00000000	A	2.98807430e-01	5.14970025e-01
    Sm   Sm  	62	150.36000000	0.00000000	A	2.72793187e-01	2.07269461e-01
    Ba   Ba  	56	137.33000000	0.00000000	A	3.27850728e-01	1.03137546e+00
    Ra   Ra  	88	226.03000000	0.00000000	A	3.27850728e-01	1.03137546e+00
    Al   Al  	13	26.98000000	0.00000000	A	1.79070642e-01	6.82410400e-05
    In   In  	49	114.82000000	0.00000000	A	2.21121062e-01	8.90363568e-03
    Y    Y   	39	88.91000000	0.00000000	A	2.59786066e-01	1.14434534e-01
    La   La  	57	138.91000000	0.00000000	A	2.93283858e-01	4.35863890e-01
    Ce   Ce  	58	140.12000000	0.00000000	A	2.44106249e-01	4.77413646e-02
    Pr   Pr  	59	140.91000000	0.00000000	A	2.82058534e-01	2.97781891e-01
    Nd   Nd  	60	144.24000000	0.00000000	A	2.78316760e-01	2.58684879e-01
    Gd   Gd  	64	157.25000000	0.00000000	A	2.68160514e-01	1.69800402e-01
    Tb   Tb  	65	158.93000000	0.00000000	A	2.63527841e-01	1.37276454e-01
    Dy   Dy  	66	162.50000000	0.00000000	A	2.61389684e-01	1.23860751e-01
    Er   Er  	68	167.26000000	0.00000000	A	2.55331573e-01	9.09820842e-02
    Tm   Tm  	69	168.93000000	0.00000000	A	2.53549775e-01	8.26723673e-02
    Lu   Lu  	71	174.97000000	0.00000000	A	2.53549775e-01	8.26723673e-02
    Hf   Hf  	72	178.49000000	0.00000000	A	2.02055829e-01	1.38323040e-03
    Zr   Zr  	40	91.22000000	0.00000000	A	2.08292120e-01	2.68700664e-03
    U    U   	92	238.03000000	0.00000000	A	2.22190140e-01	9.74746480e-03
    Pu   Pu  	94	244.06000000	0.00000000	A	2.33059105e-01	2.27646419e-02
    Th   Th  	90	232.04000000	0.00000000	A	2.62815122e-01	1.32695309e-01
    
    #include "OPC3.itp"
    #include "ions.itp"

    [ system ]
    water and salt

    [ molecules ]
    ; Compound        nmols
    SOL              {n_waters}
    {cation}               {n_cations}
    {anion}				 {n_anions}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_packmol(C, cation, anion, cation_charge=1, anion_charge=-1, n_waters=1000, water='water.pdb', filename='solution.inp', top='solution.top', packmol_options=None):
    '''
    Write packmol input file for standard MD simulation
    
    Parameters
    ----------
    C : float
        Concentration of the solution
    cation : str
        Filename of the cation in solution
    anion : str
        Filename of the anion in solution
    cation_charge : int
        Charge on the cation, default=+1
    anion_charge : int
        Charge on the anion, default=-1
    n_waters : int
        Number of water molecules to pack in system, default=1000
    water : str
        Filename of the water molecule in solution, default='water.pdb'
    filename : str
        Filename for the packmol input file, default='solution.inp'
    top : str
        Filename for the topology file, default='solution.top'
    packmol_options : dict
        Additional options to put in the packmol input file, default=None uses some presets.
        If specified, should have 'seed', 'tolerance', 'filetype', 'output', 'box'.

    Returns
    -------
    filename : str
        Filename of the packmol input file

    '''

    if packmol_options is None:
        packmol_options = {
            'seed' : 123456,
            'tolerance' : 2.0,
            'filetype' : 'pdb',
            'output' : filename.split('.inp')[0] + '.pdb',
            'box' : 32
        }

    # (n_cations / salt molecule) * (N_A salt molecules / mol salt) * (C mol salt / L water) * (L water / rho g water) * (18.02 g / mol) * (mol / N_A molecules) * (n_water molecules)
    n_cations = (-anion_charge) * C / 997 * 18.02 * n_waters
    n_anions = (cation_charge) * C / 997 * 18.02 * n_waters

    print(f'For a box of {n_waters} waters, would need {n_cations} cations and {n_anions} anions.')

    n_cations = round(n_cations)
    n_anions = round(n_anions)
    print(f'So, adding {n_cations} cations and {n_anions} anions...')

    top = update_topology(n_waters, n_cations, n_anions, filename=top, cation=cation.split('.pdb')[0].upper(), anion=anion.split('.pdb')[0].upper())

    f = dedent(f'''\
    #
    # A mixture of water and salt
    #

    # All the atoms from diferent molecules will be separated at least {packmol_options['tolerance']}
    # Anstroms at the solution.

    seed {packmol_options['seed']}
    tolerance {packmol_options['tolerance']}
    filetype {packmol_options['filetype']}

    # The name of the output file

    output {packmol_options['output']}

    # {n_waters} water molecules and {n_cations} cations, {n_anions} anions will be put in a box
    # defined by the minimum coordinates x, y and z = 0. 0. 0. and maximum
    # coordinates {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. That is, they will be put in a cube of side
    # {packmol_options['box']} Angstroms. (the keyword "inside cube 0. 0. 0. {packmol_options['box']}.") could be used as well.

    structure {water}
    number {n_waters}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. 
    end structure

    structure {cation}
    number {n_cations}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure

    structure {anion}
    number {n_anions}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure


    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


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


if __name__ == '__main__':

    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conc', type=float, default=0.2, help='Concentration to which to solvate')
    parser.add_argument('-ci', '--cation', type=str, default='Na.pdb', help='Name of pdb file with cation')
    parser.add_argument('-ai', '--anion', type=str, default='Cl.pdb', help='Name of pdb file with anion')
    parser.add_argument('-nw', '--waters', type=int, default=1000, help='Number of waters to pack in the box')
    parser.add_argument('-g', '--gro', type=str, default='solution.gro', help='Name of the coordinate file to generate')
    parser.add_argument('-mm', '--min', type=str, default='min.mdp', help='Input mdp file for minimization')
    parser.add_argument('-mv', '--nvt', type=str, default='nvt.mdp', help='Input mdp file for NVT equilibration')
    parser.add_argument('-mp', '--npt', type=str, default='npt.mdp', help='Input mdp file for NPT equilibration')
    parser.add_argument('-md', '--prod', type=str, default='prod.mdp', help='Input mdp file for production')
    parser.add_argument('-p', '--top', type=str, default='solution.top', help='Topology file')
    parser.add_argument('-i', '--inp', type=str, default='solution.inp', help='Packmol input file for creating initial box')
    parser.add_argument('-n', '--ntasks', type=int, default=16, help='Number of processors to use with mpirun')
    args = parser.parse_args()    

    # run packmol to place ions in water
    inp = write_packmol(args.conc, args.cation, args.anion, cation_charge=2, anion_charge=-1,
                        n_waters=args.waters, water='water.pdb', filename=args.inp, top=args.top)
    cmd = f'packmol < {inp}'
    run(cmd)

    # run gmx editconf to convert pdb to gro
    cmd = f'mpirun -np 1 gmx_mpi editconf -f solution.pdb -o {args.gro} -bt cubic -box 3.2 3.2 3.2'
    run(cmd)

    # minimization
    min_tpr = grompp(args.gro, args.min, top=args.top, tpr='min.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    min_gro = mdrun(min_tpr, output='min', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NVT equilibration (50 ps)
    nvt_tpr = grompp(min_gro, args.nvt, top=args.top, tpr='nvt.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    nvt_gro = mdrun(nvt_tpr, output='nvt', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NPT equilibration (1 ns)
    npt_tpr = grompp(nvt_gro, args.npt, top=args.top, tpr='npt.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    npt_gro = mdrun(npt_tpr, output='npt', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NPT production (20 ns)
    prod_tpr = grompp(npt_gro, args.prod, top=args.top, tpr='prod.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    prod_gro = mdrun(prod_tpr, output='prod', gmx=f'mpirun -np {args.ntasks} gmx_mpi')