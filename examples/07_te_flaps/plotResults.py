## Original by Ben Mertz, 2022
## Modified by Gerrit Motes, beginning 11/16/2022, in order to use with SNL
##  modifications of WEIS.
## Original file location : https://github.com/Ben-Mertz/BAR_Start/blob/main/BAR_Start/plot_results.py
## *****************************************************************************


# Python Modules
import os
import matplotlib.pyplot as plt
import glob
# ROSCO toolbox modules
from ROSCO_toolbox.ofTools.fast_io.output_processing import output_processing
import ROSCO_toolbox

# Instantiate fast_IO and FAST_Plots
op = output_processing()

# Define openfast directory and output filenames
outb_filename_1 = 'BAR10_0.outb'

openfast_dir_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp', 'BAR10')

fullPath_1 = os.path.join(openfast_dir_1, outb_filename_1)


outfiles = []
outfiles = [fullPath_1]

print('Plotting results from {}'.format(outfiles[::-1]))

# Load output info and data
fast_out = op.load_fast_out(outfiles, tmin=0.0)

#  Define Plot cases
#  --- Comment,uncomment, create, and change these as desired...
cases = {}
#cases['Baseline'] = ['Wind1VelX','BldPitch1', 'TipDxc1', 'GenPwr', 'RootMyb1','RotSpeed']
#cases['Baseline'] = ['Wind1VelX','Wind1VelY', 'BldPitch1', 'BLFLAP1', 'TipClrnc1', 'GenPwr', 'RootMyb1','RotSpeed']
#cases['Baseline'] = ['Wind1VelX','Wind1VelY', 'BLFLAP1', 'GenPwr', 'RootMyb1','RotSpeed']
cases['FlapTest'] = ['BldPitch1','BLFLAP1', 'BLFLAP2', 'BLFLAP3', 'RootMyb1', 'RootMyb2', 'RootMyb3', 'TipDxc1', 'TipDxc2', 'TipDxc3']

# Plot, woohoo!
fig, ax = op.plot_fast_out(fast_out, cases)
fig[0].tight_layout()
plt.show()
