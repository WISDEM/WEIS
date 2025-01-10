#!/usr/bin/env python3
import os
from weis import weis_main
import sys


## File management
run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-22-280-RWT_Floater.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_openfast.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_of_ptfm_opt.yaml")

# Change optimizer and output folder
optimizer = sys.argv[1]
# optimizer = "SLSQP"
print(f"Optimizer: {optimizer}")

analysis_override = {}
analysis_override["general"] = {}
analysis_override["general"]["folder_output"] = f"17_IEA22_OptStudies/of_{optimizer}"
analysis_override["driver"] = {}
analysis_override["driver"]["optimization"] = {}
analysis_override["driver"]["optimization"]["solver"] = optimizer

wt_opt, modeling_options, analysis_options = weis_main(
    fname_wt_input,
    fname_modeling_options,
    fname_analysis_options,
    analysis_override=analysis_override
)

print("Tower mass (kg) =", wt_opt["towerse.tower_mass"])
print("Floating platform mass (kg) =", wt_opt["floatingse.platform_mass"])

