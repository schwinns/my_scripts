# Some utility functions for running pdfgetx3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import mean_squared_error

import subprocess
from textwrap import dedent

def run(commands, cwd='./'):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, cwd=cwd, shell=True)


def get_data(file, comments='#', skiprows=0):
    '''
    Get data from a dat file
    file : str, file
    '''

    data = np.loadtxt(file, comments=comments, skiprows=skiprows)
    x = data[:,0]
    y = data[:,1]

    return x, y


def g2G(r, g_r, rho0):
    '''
    Convert g(r) to G(r), the weighted radial distribution function to the reduced pair distribution function

    G(r) = 4 pi r / rho0 ( g(r) - 1 )

    r : array-like, Angstroms
    g_r : array-like, same shape as r
    rho0 : float, number density = N/V
    '''
    return 4*np.pi*r/rho0 * (g_r-1)


def G2g(r, G_r, rho0):
    '''
    Convert G(r) to g(r), inverse of g2G()
    
    g(r) = G(r) rho0 / (4 pi r) + 1

    r : array-like, Angstroms
    G_r : array-like, same shape as r
    rho0 : float, number density = N/V
    '''
    return G_r*rho0/4/np.pi/r + 1


def write_G_r(r, G_r, filename='output.gr'):
    '''
    Write data to file
    r : array-like, Angstroms
    G_r : array-like, same shape as r
    filename : str, file
    '''

    data = np.vstack((r, G_r)).T
    np.savetxt(filename, data, fmt='%.8f', delimiter='\t', header='r (A)\tG(r)')


def plot_G_r(r, G_r, label=None, xticks=(1,0.25), yticks=(0.1,0.025), xlims=None, ylims=None, alpha=1, savename=None, ax=None):
    '''Plot G(r) nicely with tick marks'''

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

    plt.plot(r, G_r, label=label, alpha=alpha)
    plt.xlabel('r ($\mathrm{\AA}$)', fontsize=14)
    plt.ylabel('G(r)', fontsize=14)
    plt.legend(fontsize=14)

    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)

    ax.xaxis.set_major_locator(MultipleLocator(xticks[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(xticks[1]))
    ax.yaxis.set_major_locator(MultipleLocator(yticks[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(yticks[1]))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if savename is not None:
        plt.savefig(savename)

    return ax


def plot_g_r(r, g_r, label=None, alpha=1, xticks=(1,0.25), yticks=(0.5,0.1), ylims=None, xlims=None, color=None, ax=None, output=None):
    '''
    Plot PDF nicely from r and g(r)
    r : units Angstroms
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

    ax.plot(r, g_r, label=label, alpha=alpha, c=color)

    ax.xaxis.set_major_locator(MultipleLocator(xticks[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(xticks[1]))
    ax.yaxis.set_major_locator(MultipleLocator(yticks[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(yticks[1]))

    if ylims is None:
        plt.ylim(0, g_r.max() + 1)
    else:
        plt.ylim(ylims[0], ylims[1])

    if xlims is None:
        plt.xlim(0, r.max())
    else:
        plt.xlim(xlims[0], xlims[1])

    plt.ylabel('g(r)')
    plt.xlabel('r ($\mathrm{\AA}$)')
    if label is not None:
        plt.legend()

    if output is not None:
        plt.savefig(output)

    return ax


def objective_function_wet(x, input_wet='avgd_PA_wet_11p5min_getx3.qq.txt', input_dry='avgd_PA_dry_11p5min_getx3.qq.txt', bg_wet='avgd_water_11p5min_getx3.qq.txt', bg_dry=None, comp_wet='C4451.85H5687.94O2132.77N731.62', comp_dry='C17.7N2.9H11.8O3.1', output_wet='tmp_wet.qq.gr', output_dry='tmp_dry.qq.gr'):
    '''
    Objective function for the optimization problem
    
    Input (x) : bgscale, rpoly, qmin, qmax
    Output : MSE between G_r_wet and G_r_dry for r values >= rpoly

    '''

    # bgscale, rpoly, qmin, qmax = x
    bgscale = float(x)
    rpoly = 2
    qmin = 0.3
    qmax = 16

    # standard configuration settings
    cfg = Config()
    cfg.outputtype = ['gr']
    # cfg.outputtype = ['iq']
    cfg.plot = ['none']
    cfg.verbose = 0

    # variable configuration settings
    cfg.bgscale = bgscale
    cfg.rpoly = rpoly
    cfg.qmin = qmin
    cfg.qmax = qmax

    # system specific configuration settings
    cfg.backgroundfile = bg_wet
    cfg.composition = comp_wet
    cfg.output = output_wet
    cfg.run(inputs=[input_wet], config_name='tmp_wet.cfg')
    r_wet, G_r_wet = get_data(output_wet, skiprows=27)
    # q_wet, I_q_wet = get_data(output_wet, skiprows=27)
    
    cfg.backgroundfile = bg_dry
    cfg.composition = comp_dry
    cfg.output = output_dry
    cfg.run(inputs=[input_dry], config_name='tmp_dry.cfg')
    r_dry, G_r_dry = get_data(output_dry, skiprows=27)
    # q_dry, I_q_dry = get_data(output_dry, skiprows=27)

    G_r_wet = G_r_wet[r_wet >= rpoly]
    G_r_dry = G_r_dry[r_dry >= rpoly]
    # I_q_wet = I_q_wet[1:]

    run('rm tmp*')

    return mean_squared_error(G_r_dry, G_r_wet)
    # return mean_squared_error(I_q_dry, I_q_wet)


class Config:
    def __init__(self):

        # initialize all the options
        self.filename = 'pdfgetx3.cfg'

        # input and output options
        self.parameters = 'DEFAULT'
        self.dataformat = 'QA'
        self.inputfile = []    
        self.backgroundfile = None
        self.formfactorfile = None
        self.datapath = []        
        self.output = '@b.@o'
        self.outputtype = ['none']
        self.force = False

        # PDF parameters
        self.bgscale = 1
        self.mode = 'xray'
        self.wavelength = 0.2116
        self.twothetazero = 0
        self.composition = 'C4451.85H5687.94O2132.77N731.62' # default is for wet PA from simulation
        self.rpoly = 2
        self.qmaxinst = 23.0
        self.qmin = 0.3
        self.qmax = 16
        self.rmin = 0.0
        self.rmax = 10
        self.rstep = 0.02
    
        # other parameters
        self.plot = ['none']
        self.interact = 'no'
        self.verbose = 5


    def write(self, filename=None):
        '''Write out the pdfgetx3 config file'''

        if filename is None:
            filename = self.filename

        self.filename = filename
        out = open(filename, 'w')

        header = dedent('''\
        ## Template configuration file for pdfgetx3.

        ## pdfgetx3 searches for default configurations in ~/.pdfgetx3.cfg,
        ## .pdfgetx3.cfg, pdfgetx3.cfg and then loads a custom configuration
        ## if specified via the "-c" option.  You can run
        ##
        ##     pdfgetx3 --verbose=info
        ##
        ## to verify how and from what file are the parameters set.

        ## The default section -------------------------------------------------------

        ''')
        out.write(header)

        defaults = dedent(f'''\
        ## Parameters defined here are also available in custom sections below.
        [{self.parameters}]
        
        ''')
        out.write(defaults)

        dataformat = dedent(f'''\
        ## Format of input files.  Available formats are: "twotheta", "QA", "Qnm"
        ## corresponding to a 2-column text data where the first column is either
        ## twotheta in degrees, Q in inverse Angstroms or Q in inverse nanometers.
        dataformat = {self.dataformat}

        ''')
        out.write(dataformat)

        input_intensities = dedent(f'''\
        ## One or more input xray intensities from the sample.  This setting is
        ## ignored if there are any files provided on the command line.  Several
        ## files need to be specified as one file per line.
        
        ''')
        for inp in self.inputfile:
            input_intensities += f'{inp}\n'

        out.write(input_intensities)

        backgroundfile = dedent(f'''\
        ## Optional background intensities from container and air scattering
        ''')
        if self.backgroundfile is not None:
            backgroundfile += f'backgroundfile = {self.backgroundfile}\n'
        
        backgroundfile += '\n'
        out.write(backgroundfile)

        add_dir = dedent(f'''\
        ## Additional directories to be searched for input files, one per line.
        ''')
        for dp in self.datapath:
            add_dir += f'{dp}\n'

        out.write(add_dir)
    
        bgscale = dedent(f'''\
        ## Optional scaling of the background intensities.  By default 1.
        bgscale = {self.bgscale}

        ''')
        out.write(bgscale)
    
        formfactors = dedent(f'''\
        ## Form factor intensities of the scatterers. This is required for `sas`
        ## mode. The form factor file is expected to be in two-column format
        ## with (Q, f2avg) data or three-column format with
        ## (Q, f2avg, favg2) data. The unit of Q is required to be A^-1.
        ''')
        if self.formfactorfile is not None:
            formfactors += f'formfactorfile = {self.formfactorfile}\n'
        
        formfactors += '\n'
        out.write(formfactors)
    
        outputs = dedent(f'''\
        ## Output file name, write to the standard output when "-".
        ## This may contain @f, @h, @r, @e, @t, @b, @o tokens which expands as follows:
        ##
        ##   @f  dir1/dir2/filename.dat    input file path
        ##   @h  dir1/dir2                 input file head directory or '.'
        ##   @r  dir1/dir2/filename        input path with extension removed
        ##   @e  dat                       input file extension without '.'
        ##   @t  filename.dat              tail component of the input file
        ##   @b  filename                  tail component with extension removed
        ##   @o  gr                        output extension iq, sq, fq or gr
        ##
        ## An empty value works the same as "@b.@o", i.e., saves the data
        ## in the current directory with a proper data type extension.
        output = {self.output}
        ''')
        out.write(outputs)

        outputtypes = dedent(f'''\
        ## Types of output files to be saved.  Possible values are
        ## "iq", "sq", "fq", "gr", also used as filename extensions.
        ## No files are saved when empty, "none" or "NONE".
        ''')
        line = 'outputtype ='
        for ot in self.outputtype:
            line += f' {ot}'
        
        outputtypes += line + '\n'
        out.write(outputtypes)
    
        forces = dedent(f'''\
        ## Flag for overwriting existing output files.  By default False.
        ## It is probably safer to use it from command line.
        force = {self.force}

        ''')
        out.write(forces)

        mode = dedent(f'''\
        ## The PDF calculator configuration mode or a name of the calculator
        ## setup.  The available modes correspond to the radiation type.
        ## The supported modes are:
        ## 'xray', 'neutron', 'sas'
        mode = {self.mode}

        ''')
        out.write(mode)

        wavelength = dedent(f'''\
        ## X-ray, neutron, or electron wavelength in Angstroms.
        ## Required for the "twotheta" dataformat.
        wavelength = {self.wavelength}
        
        ''')
        out.write(wavelength)

        twothetazero = dedent(f'''\
        ## Position of the zero scattering angle in diffractometer degrees.
        ## Applies only for the "twotheta" dataformat.
        twothetazero = {self.twothetazero}

        ''')
        out.write(twothetazero)

        composition = dedent(f'''\
        ## Chemical composition of the measured sample.  Supported formats are
        ## "PbTi0.5Zr0.5O3", "Pb (TiZr)0.5 O3" or "Pb 1 (Ti Zr) 1/2 O 3".
        ## Space characters are ignored, unit counts can be omitted, but it is
        ## important to use a proper upper and lower case in atom symbols.
        ## Elements can appear several times in the formula, e.g., "CH3 CH3",
        ## and the formula may contain parentheses or fractional counts.
        composition = {self.composition}
        
        ''')
        out.write(composition)

        rpoly = dedent(f'''\
        ## r-limit for the maximum frequency in the F(Q) correction polynomial.
        ## The PDF is unreliable at shorter r, however a too small rpoly
        ## disables polynomial correction and yields noisy PDF.  Too large
        ## values may smooth-out useful signal in the data.
        rpoly = {self.rpoly}

        ''')
        out.write(rpoly)

        qmaxinst = dedent(f'''\
        ## The Q cutoff for the meaningful input intensities in inverse Angstroms.
        ## This is the upper boundary for the qmax parameter.  It is also used as
        ## the upper boundary for the polynomial fits in S(Q) corrections.
        qmaxinst = {self.qmaxinst}
        
        ''')
        out.write(qmaxinst)

        qmin = dedent(f'''
        ## Lower Q cutoff for Fourier transformation in inverse Angstroms.
        ## Use 0.0 when not specified.
        qmin = {self.qmin}

        ''')
        out.write(qmin)

        qmax = dedent(f'''\
        ## Upper Q cutoff for Fourier transformation in inverse Angstroms.
        ## Use maximum Q in the data when not specified.
        qmax = {self.qmax}
        
        ''')
        out.write(qmax)

        rrange = dedent(f'''\
        ## Limits and spacing for the calculated PDF r-grid.
        ## All values in Angstroms.
        rmin = {self.rmin}
        rmax = {self.rmax}
        rstep = {self.rstep}

        ''')
        out.write(rrange)
    
        plot = dedent(f'''\
        ## Plot the specified results and activate interactive mode.
        ## A comma separated list with items such as "iq", "sq", "fq", "gr".
        ## No plot is produced when empty, "none" or "NONE".
        ''')
        line = 'plot ='
        for p in self.plot:
            line += f' {p}'
        
        plot += line + '\n'
        out.write(plot)

        interact = dedent(f'''\
        ## Start an IPython interactive session after processing all files.
        ## Useful for tuning the configuration parameters or interactive plotting.
        ## This is always on when plot option has been set.
        interact = {self.interact}
        
        ''')
        out.write(interact)
    
        verbosity = dedent(f'''\
        ## Program verbosity - the minimum priority of the messages to be displayed.
        ## Possible values are (error, warning, info, debug, all) or an integer from
        ## 0 to 5, where 1 corresponds to error, 2 to warning and so forth.
        verbose = {self.verbose}
        
        ''')
        out.write(verbosity)
    
        if self.verbose != 0:
            print(f'Wrote config file to {filename}!')


    def get_composition_from_MD(self, coord):
        '''Use MD coordinate file to calculate the composition for the experimental PDF'''
        pass


    def run(self, inputs=[], config_name='tmp.cfg'):
        '''Run pdfgetx3 with the parameters in self'''
        
        self.write(config_name)
        
        cmd = f'pdfgetx3 -c {config_name}'
        for inp in inputs:
            cmd += f' {inp}'

        run(cmd)

