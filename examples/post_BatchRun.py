__author__ = ["Nikhar Abbas", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Nikhar Abbas", "Jake Nunemaker"]
__email__ = ["nikhar.abbas@nrel.gov", "jake.nunemaker@nrel.gov"]


import os
from fnmatch import fnmatch

#import numpy as np
import pandas as pd
try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError("No module named ruamel.yaml or ruamel_yaml")

from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
#from pCrunch.io import load_FAST_out
from pCrunch.utility import save_yaml, get_windspeeds, convert_summary_stats


def valid_extension(fp):
    return any([fnmatch(fp, ext) for ext in ["*.outb", "*.out"]])


# Define input files paths
output_dir = "/Users/gbarter/devel/WEIS/examples/05_IEA-3.4-130-RWT/temp/iea34"
results_dir = os.path.join(output_dir, "results")
save_results = True


# Find outfiles
outfiles = [
    os.path.join(output_dir, f)
    for f in os.listdir(output_dir)
    if valid_extension(f)
]

# Configure pCrunch
magnitude_channels = {
    "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
    "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
    "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
}

fatigue_channels = {"RootMc1": FatigueParams(lifetime=25, slope=10),
                    "RootMc2": FatigueParams(lifetime=25, slope=10),
                    "RootMc3": FatigueParams(lifetime=25, slope=10),
                    }

channel_extremes = [
    "RotSpeed",
    "RotThrust",
    "RotTorq",
    "RootMc1",
    "RootMc2",
    "RootMc3",
]

# Run pCrunch
la = LoadsAnalysis(
    outfiles,
    magnitude_channels=magnitude_channels,
    fatigue_channels=fatigue_channels,
    extreme_channels=channel_extremes,
    trim_data=(0,),
)
la.process_outputs(cores=1)

if save_results:
    save_yaml(
        results_dir,
        "summary_stats.yaml",
        convert_summary_stats(la.summary_stats),
    )

# Load case matrix into dataframe
fname_case_matrix = os.path.join(output_dir, "case_matrix.yaml")
with open(fname_case_matrix, "r") as f:
    case_matrix = ry.load(f, Loader=ry.Loader)
cm = pd.DataFrame(case_matrix)

# Get wind speeds for processed runs
windspeeds, seed, IECtype, cm_wind = get_windspeeds(cm, return_df=True)

# Get AEP
turbine_class = 1
pp = PowerProduction(turbine_class)
AEP, perf_data = pp.AEP(
    la.summary_stats,
    windspeeds,
    ["GenPwr", "RtAeroCp", "RotSpeed", "BldPitch1"],
)
print(f"AEP: {AEP}")

# # ========== Plotting ==========
# an_plts = Analysis.wsPlotting()
# #  --- Time domain analysis ---
# filenames = [outfiles[0][2], outfiles[1][2]] # select the 2nd run from each dataset
# cases = {'Baseline': ['Wind1VelX', 'GenPwr', 'BldPitch1', 'GenTq', 'RotSpeed']}
# fast_dict = fast_io.load_FAST_out(filenames, tmin=30)
# fast_pl.plot_fast_out(cases, fast_dict)

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
