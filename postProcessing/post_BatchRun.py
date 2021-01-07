'''
A script to post process a batch run and generate stats and load rankings
'''

# Python Modules and instantiation
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
import os
import pathlib
# ROSCO toolbox modules 
# from ROSCO_toolbox import utilities as ROSCO_utilites
# from ofTools.fast_io
# fast_io = ROSCO_utilites.FAST_IO()
# fast_pl = ROSCO_utilites.FAST_Plots()
# WEIS modules
from weis.aeroelasticse.Util import FileTools
from ROSCO_toolbox.ofTools.fast_io.output_processing import output_processing
# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis

if __name__ == '__main__':
    # Define input files paths
    output_dir      = '/Users/dzalkind/Tools/WEIS-2/outputs/iea15mw/PC_sweep_play'
    results_dir     = 'results'
    save_results    = True


    # Load case matrix into dataframe
    fname_case_matrix = os.path.join(output_dir,'case_matrix.yaml')
    case_matrix = FileTools.load_yaml(fname_case_matrix, package=1)
    cm = pd.DataFrame(case_matrix)

    # Find all outfiles
    outfiles = []
    for file in os.listdir(output_dir):
        if file.endswith('.outb'):
            outfiles.append(os.path.join(output_dir,file))
        elif file.endswith('.out') and not file.endswith('.MD.out'):
            outfiles.append(os.path.join(output_dir,file))


    # Initialize processing classes
    fp = Processing.FAST_Processing()
    fa = Analysis.Loads_Analysis()


    # Set some processing parameters
    fp.OpenFAST_outfile_list = outfiles
    fp.t0 = 0
    fp.parallel_analysis = False
    fp.results_dir = os.path.join(output_dir, 'stats')
    fp.verbose=True

    if save_results:
        fp.save_LoadRanking = True
        fp.save_SummaryStats = True

    # Load and save statistics and load rankings
    stats, load_rankings = fp.batch_processing()

    
    

    # Get wind speeds for processed runs
    windspeeds, seed, IECtype, cm_wind = Processing.get_windspeeds(cm, return_df=True)
    stats_df = pdTools.dict2df(stats)

    print('here')


# # Get AEP
# pp = Analysis.Power_Production()
# Vavg = 10   # Average wind speed of cite
# Vrange = [2,26] # Range of wind speeds being considered
# # bnums = int(len(set(windspeeds))/len(fp.namebase)) # Number of wind speeds per dataset for binning data
# bnums = len(fp.OpenFAST_outfile_list)
# pp.windspeeds = list(set(windspeeds))
# p = pp.gen_windPDF(Vavg, bnums, Vrange)
# AEP = pp.AEP(stats)
# print('AEP = {}'.format(AEP))


# # Plot some spectral cases
# spec_cases = [('RootMyb1', 0), ('TwrBsFyt', 0)]
# twrfreq = .0716
# fig,ax = fast_pl.plot_spectral(fast_dict, spec_cases, show_RtSpeed=True, 
#                         add_freqs=[twrfreq], add_freq_labels=['Tower'],
#                         averaging='Welch')
# ax.set_title('DLC1.1')

# # Plot a data distribution
# channels = ['RotSpeed']
# caseid = [0, 1]
# an_plts.distribution(fast_dict, channels, caseid, names=['DLC 1.1', 'DLC 1.3'])

# # --- Batch Statistical analysis ---
# # Bar plot
# fig,ax = an_plts.stat_curve(windspeeds, stats, 'RotSpeed', 'bar', names=['DLC1.1', 'DLC1.3'])

# # Turbulent power curve
# fig,ax = an_plts.stat_curve(windspeeds, stats, 'GenPwr', 'line', stat_idx=0, names=['DLC1.1'])

# plt.show()



