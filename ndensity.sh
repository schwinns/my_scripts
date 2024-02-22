#!/bin/bash

xtc=$1
tpr=$2

source /usr/local/gromacs/bin/GMXRC

echo '1' | gmx density -f $xtc -s $tpr -n mono_gly.ndx -sl 100 -dens number -o ndensity_mono.xvg

echo '2' | gmx density -f $xtc -s $tpr -n mono_gly.ndx -sl 100 -dens number -o ndensity_ions.xvg

echo '3' | gmx density -f $xtc -s $tpr -n mono_gly.ndx -sl 100 -dens number -o ndensity_gly.xvg

echo '6' | gmx density -f $xtc -s $tpr -n mono_gly.ndx -sl 100 -dens number -o ndensity_head.xvg

echo '9' | gmx density -f $xtc -s $tpr -n mono_gly.ndx -sl 100 -dens number -o ndensity_tail.xvg
