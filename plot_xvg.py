# Script to quickly plot .xvg files from GROMACS
# Uses GromacsWrapper so should be in LLC-env or gro_wrap

# Use as:
#           python plot_xvg.py <filename> -x xmin xmax -y ymin ymax

import numpy as np
import gromacs as gro
import sys
import matplotlib.pyplot as plt

# Get the .xvg file to plot from the command line
filename = sys.argv[1]

# Should be done better -- getting ylims and xlims from command line
ylims = False
xlims = False

if len(sys.argv) > 2:
    if sys.argv[2] == '-y':
        ymin = sys.argv[3]
        ymax = sys.argv[4]
        ylims = True
    elif len(sys.argv) > 5 and sys.argv[5] == '-y':
        ymin = sys.argv[6]
        ymax = sys.argv[7]
        ylims = True

    if sys.argv[2] == '-x':
        xmin = sys.argv[3]
        xmax = sys.argv[4]
        xlims = True
    elif len(sys.argv) > 5 and sys.argv[5] == '-x':
        xmin = sys.argv[6]
        xmax = sys.argv[7]
        xlims = True

# Import the .xvg file
xvg = gro.fileformats.XVG(filename)

# Plot using the GromacsWrapper and save as 'filename'.png
fig = plt.figure()
plot = xvg.plot(color='Dark2', maxpoints=None)

if ylims:
    plot.set_ylim(ymin, ymax)
if xlims:
    plot.set_xlim(xmin, xmax)

name = filename[0:-4]
plot.set_ylabel(name)
#plot.set_xlabel('time (ps)')
plt.savefig(name + '.png')
