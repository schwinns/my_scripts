#!/usr/bin/env python

import numpy as np
import gromacs as gro
import matplotlib.pyplot as plt

# Plot number densities of everything
files = ['ndensity_mono.xvg','ndensity_gly.xvg',
         'ndensity_head.xvg','ndensity_tail.xvg','ndensity_ions.xvg']
colors = ['navy','seagreen','dimgray','orange','indianred']
natoms = [138,14,16,36,2]

fig = plt.figure()
plt.xlabel('z box length')
plt.ylabel('density')

for file,c,n in zip(files,colors,natoms):
    xvg = gro.fileformats.XVG(file)
    x = xvg.array[0,:]
    y = xvg.array[1,:]/n

    print(file)
    print('Mean density: %.4f\n' % (np.mean(y)))
    
    plt.plot(x,y,color=c)

plt.legend(['Monomer','Glycerol','Heads','Tails','Ions'],loc=0)
plt.savefig('ndensity_all.png')

# Zoom in on monomers and ions
files = ['ndensity_mono.xvg','ndensity_head.xvg','ndensity_tail.xvg','ndensity_ions.xvg']
colors = ['navy','dimgray','orange','indianred']
natoms = [138,16,36,2]


fig = plt.figure()
plt.xlabel('z box length')
plt.ylabel('density')

for file,c,n in zip(files,colors,natoms):
    xvg = gro.fileformats.XVG(file)
    x = xvg.array[0,:]
    y = xvg.array[1,:]/n

    print(file)
    print('Mean density: %.4f\n' % (np.mean(y)))
    
    plt.plot(x,y,color=c)

plt.legend(['Monomer','Heads','Tails','Ions'],loc=0)
plt.savefig('ndensity_mono_ions.png')