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
#fn = 'BAR_DRC_LE0_SS50SE95_DLC11_IEC_11.outb'
#fn = 'BAR_DRC_clean_vout25_DLC11_IEC_07.outb'
#fn2 = 'BAR_DRC_LE_DLC51_IEC_1.outb'
#fn2 = 'BAR_DRC_LE_SS50SE95_DLC11_IEC_07.outb'
#fn3 = 'BAR_DRC_LE_SS75SE95_DLC11_IEC_07.outb'
outb_filename_1 = 'BAR10_0.outb'

openfast_dir_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp', 'BAR10_AlwaysOn')
openfast_dir_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp', 'BAR10_AlwaysOff')
#openfast_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'BAR_DRC_LE0_SS50SE95_DLC11')
#openfast_dir2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'BAR_DRC_LE_DLC51')
#openfast_dir2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'BAR_DRC_LE_SS50SE95_DLC11')
#openfast_dir3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'BAR_DRC_LE_SS75SE95_DLC11')

fullPath_1 = os.path.join(openfast_dir_1, outb_filename_1)
fullPath_2 = os.path.join(openfast_dir_2, outb_filename_1)
#filenames2 = os.path.join(openfast_dir2, fn2)
#filenames3 = os.path.join(openfast_dir3, fn3)

outfiles = []
outfiles = [fullPath_1, fullPath_2]
#outfiles = [fullPath_1]
#outfiles = [filenames, filenames2, filenames3]
#outfiles = [os.path.split(fname)[-1:][0] for fname in filenames]
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
