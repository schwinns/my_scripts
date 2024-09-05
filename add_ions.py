# Script to add ions near the carboxylate groups

from polym_analysis import PolymAnalysis
import MDAnalysis as mda
import subprocess


def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def update_topology(gro, top, n_ions, ion_name, output='PA_ions.top'):
    '''Update the topology with newly added ions'''

    # create an MDAnalysis universe
    u = mda.Universe(gro)

    # read original topology file
    orig = open(top, 'r')
    lines = orig.readlines()
    orig.close()

    # write new file
    new = open(output, 'w')

    # update number of waters (since some may have been removed)
    n_waters = len(u.select_atoms('resname SOL').residues)
    lines[-1] = f'SOL\t{n_waters}\n'

    # add ions to [ molecules ] directive
    new_line = f'{ion_name}\t{n_ions}\n'
    lines.append(new_line)
    for line in lines:
        new.write(line)

    return output


def generate_tpr(top, coord, mdp='min.mdp', tpr='out.tpr', gmx='gmx'):
    '''Use Gromacs to generate a tpr with grompp'''

    cmd = f'{gmx} grompp -f {mdp} -c {coord} -p {top} -o {tpr} -maxwarn 1'
    run(cmd)

    return tpr



if __name__ == '__main__':

    # input desired concentration in water reservoir and names/charges of ions, also number of extra ions to insert in membrane
    cation = 'NA'
    anion = 'CL'
    cation_charge = 1
    anion_charge = -1
    extra_added = 65 # 65 is the number for 50% deprotonated, adding for statistics and comparability even though fully protonated
    C = 0.2 # M = mol / L

    # for now, manually add the atomtypes for the ions from ions.itp to beginning of top file and include ions.itp at bottom of top file
    # SAVING AS: pre_ions.top
    ion_top = 'pre_ions.top'
    input_gro = 'hydrated.gro'

    # generate tpr that MDAnalysis can read
    tpr = generate_tpr(ion_top, input_gro, mdp='min.mdp', tpr='hydrated.tpr', gmx='gmx')

    # define proper types for Gromacs system
    xlink_c = 'c and bonded type n'
    xlink_n = 'n'
    term_n = 'nh'
    cl_type = 'cl'
    oh_type = 'oh'
    ow_type = 'OW'
    hw_type = 'HW'

    # initialize PolymAnalysis
    gro = PolymAnalysis(input_gro, frmt='GRO', tpr_file=tpr, 
                        xlink_c=xlink_c, xlink_n=xlink_n, term_n=term_n, cl_type=cl_type,
                        oh_type=oh_type, ow_type=ow_type, hw_type=hw_type)
    gro.calculate_density('prop z >= 0 and prop z <= 100')

    # calculate how many ions are needed to reach desired concentration
    bins, hist = gro.density_profile('resname PA*')
    bins = bins[1:]
    wat_z = (bins[hist <= 10][0], bins[hist <= 10][-1]) # water reservoir defined as region where there are at most 10 PA atoms
    wat_vol = (gro.box[1,0]-gro.box[0,0]) * (gro.box[1,1]-gro.box[0,1]) * (wat_z[1] - wat_z[0]) / 10**3
    n_salt = round(6.022*10**23 * C * 10**3 / 10**27 * wat_vol) # salt molecules / volume water
    print(f'Need {n_salt} salt molecules to reach {C} M')

    # add cations and update topology for cations near de/protonated groups
    n_cations, new_gro, excess_charge = gro.insert_cations_in_membrane(ion_name=cation, ion_charge=cation_charge, 
                                                                       extra_inserted=extra_added, output='PA_ions.gro')
    new_top = update_topology(gro=new_gro, top=ion_top, n_ions=n_cations, ion_name=cation)
    new_tpr = generate_tpr(top=new_top, coord=new_gro, mdp='min.mdp', tpr='PA_ions.tpr', gmx='gmx')

    # reinitialize PolymAnalysis
    gro = PolymAnalysis(new_gro, frmt='GRO', tpr_file=new_tpr, 
                        xlink_c=xlink_c, xlink_n=xlink_n, term_n=term_n, cl_type=cl_type,
                        oh_type=oh_type, ow_type=ow_type, hw_type=hw_type)
    gro.gmx = 'gmx'

    # balance excess charge and add ions to concentration
    print(f'WARNING: current implementation assumes cations and anions are either monovalent or divalent, cannot handle different combinations')
    if cation_charge == -anion_charge:
        c = 1
        a = 1
    elif cation_charge == 2 and anion_charge == -1:
        c = 1
        a = 2
    elif cation_charge == 1 and anion_charge == -2:
        c = 2
        a = 1

    if excess_charge < 0: 
        n_cations = int(excess_charge / cation_charge + n_salt * c)
        n_anions = n_salt * a
    elif excess_charge > 0:
        n_cations = n_salt * c
        n_anions = int(-excess_charge / anion_charge + n_salt * a)

    total_charge = excess_charge + n_cations*cation_charge + n_anions*anion_charge
    if abs(total_charge) == 1: # this should only happen with one ion being +/-1 and the other being +/-2
        n_cations += 1
        n_anions += 1
        
    total_charge = excess_charge + n_cations*cation_charge + n_anions*anion_charge
    print(f'Adding {n_cations} cations and {n_anions} anions for a total charge of {total_charge}')
    gro.insert_ions(n_anions=n_anions, anion_name=anion, anion_charge=anion_charge,
                    n_cations=n_cations, cation_name=cation, cation_charge=cation_charge, 
                    top=new_top, output=new_gro, water_sel=18)
    new_tpr = generate_tpr(top=new_top, coord=new_gro, mdp='min.mdp', tpr='PA_ions.tpr', gmx='gmx')
