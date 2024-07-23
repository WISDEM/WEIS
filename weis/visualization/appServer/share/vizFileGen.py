# This file generates an inputfile for the vizulaization tool.
# the user passes the modeling and analysis option files, the tools generated what
# we are currently calling the vizInputFile.yaml. This provides the viz tool
# with the information it needs to generate the visualizations for the specific run.
# The generation of the vizInputFile.yaml will eventually be offloaded to WEIS.

import os
import numpy as np
import pandas as pd
import argparse
import yaml

from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS


class WEISVizInputFileGenerator:

    def __init__(self, fname_modeling_options, fname_opt_options, fname_wt_input):
        self.fname_modeling_options = fname_modeling_options
        self.fname_opt_options = fname_opt_options
        self.fname_wt_input = fname_wt_input
        self.vizInput = {}


    def fetchWEISinputs(self):
        wt_initial = WindTurbineOntologyPythonWEIS(self.fname_wt_input, self.fname_modeling_options, self.fname_opt_options)
        self.wt_init, self.modeling_options, self.opt_options = wt_initial.get_input_data()

    def userOptions(self):
        # Add the user options to the vizInput file
        self.vizInput['userOptions'] = {}
        # Run type
        self.vizInput['userOptions']['optimization'] = self.opt_options['driver']['optimization']['flag']
        self.vizInput['userOptions']['deisgn_of_experiments'] = self.opt_options['driver']['design_of_experiments']['flag']
        self.vizInput['userOptions']['inverse_design'] = False if self.opt_options['inverse_design'] == {} else True

        # SQL recorder context
        self.vizInput['userOptions']['sql_recorder'] = self.opt_options['recorder']['flag']
        self.vizInput['userOptions']['sql_recorder_file'] = self.opt_options['recorder']['file_name']

        # Outputs context
        self.vizInput['userOptions']['output_folder'] = os.path.join(os.path.split(self.fname_opt_options)[0], self.opt_options['general']['folder_output']) 
        self.vizInput['userOptions']['output_fileName'] = self.opt_options['general']['fname_output']

    def setDefaultUserPreferencs(self):
        '''
        userPreferences:
            openfast:
                file_path:
                file1: of-output/NREL5MW_OC3_spar_0.out
                file2: of-output/IEA15_0.out
                file3: of-output/IEA15_1.out
                file4: of-output/IEA15_2.out
                file5: None
                graph:
                xaxis: Time
                yaxis:
                - Wind1VelX
                - Wind1VelY
                - Wind1VelZ
            optimization:
                convergence:
                channels:
                - raft.pitch_period
                - floatingse.constr_draft_heel_margin
                - floating.jointdv_0
                - floating.memgrp1.outer_diameter_in
                dlc:
                xaxis: Wind1VelX
                xaxis_stat: mean
                yaxis:
                - Wind1VelY
                - Wind1VelZ
                yaxis_stat: max
                timeseries:
                channels:
                - Wind1VelX
                - Wind1VelY
                - Wind1VelZ
            wisdem:
                blade:
                shape_yaxis:
                - rotorse.rc.chord_m
                - rotorse.re.pitch_axis
                - rotorse.theta_deg
                struct_yaxis:
                - rotorse.rhoA_kg/m
                struct_yaxis_log:
                - rotorse.EA_N
                - rotorse.EIxx_N*m**2
                - rotorse.EIyy_N*m**2
                - rotorse.GJ_N*m**2
                xaxis: rotorse.rc.s
                output_path: /Users/sryu/Desktop/FY24/WEIS/WISDEM/examples/02_reference_turbines/outputs/
        '''

        # Set the default user preferences based on the comment above
        self.vizInput['userPreferences'] = {}
        self.vizInput['userPreferences']['openfast'] = {}
        self.vizInput['userPreferences']['openfast']['file_path'] = {}
        self.vizInput['userPreferences']['openfast']['file_path']['file1'] = 'None'
        self.vizInput['userPreferences']['openfast']['file_path']['file2'] = 'None'
        self.vizInput['userPreferences']['openfast']['file_path']['file3'] = 'None'
        self.vizInput['userPreferences']['openfast']['file_path']['file4'] = 'None'
        self.vizInput['userPreferences']['openfast']['file_path']['file5'] = 'None'

        self.vizInput['userPreferences']['openfast']['graph'] = {}
        self.vizInput['userPreferences']['openfast']['graph']['xaxis'] = 'Time'
        self.vizInput['userPreferences']['openfast']['graph']['yaxis'] = ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ']

        self.vizInput['userPreferences']['optimization'] = {}
        self.vizInput['userPreferences']['optimization']['convergence'] = {}
        self.vizInput['userPreferences']['optimization']['convergence']['channels'] = ['raft.pitch_period', 'floatingse.constr_draft_heel_margin', 'floating.jointdv_0', 'floating.memgrp1.outer_diameter_in']
        self.vizInput['userPreferences']['optimization']['dlc'] = {}
        self.vizInput['userPreferences']['optimization']['dlc']['xaxis'] = 'Wind1VelX'
        self.vizInput['userPreferences']['optimization']['dlc']['xaxis_stat'] = 'mean'
        self.vizInput['userPreferences']['optimization']['dlc']['yaxis'] = ['Wind1VelY', 'Wind1VelZ']
        self.vizInput['userPreferences']['optimization']['dlc']['yaxis_stat'] = 'max'
        self.vizInput['userPreferences']['optimization']['timeseries'] = {}
        self.vizInput['userPreferences']['optimization']['timeseries']['channels'] = ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ']

        self.vizInput['userPreferences']['wisdem'] = {}
        self.vizInput['userPreferences']['wisdem']['blade'] = {}
        self.vizInput['userPreferences']['wisdem']['blade']['shape_yaxis'] = ['rotorse.rc.chord_m', 'rotorse.re.pitch_axis', 'rotorse.theta_deg']
        self.vizInput['userPreferences']['wisdem']['blade']['struct_yaxis'] = ['rotorse.rhoA_kg/m']
        self.vizInput['userPreferences']['wisdem']['blade']['struct_yaxis_log'] = ['rotorse.EA_N', 'rotorse.EIxx_N*m**2', 'rotorse.EIyy_N*m**2', 'rotorse.GJ_N*m**2']
        self.vizInput['userPreferences']['wisdem']['blade']['xaxis'] = 'rotorse.rc.s'
        self.vizInput['userPreferences']['wisdem']['output_path'] = os.path.join(os.path.split(self.fname_opt_options)[0], self.opt_options['general']['folder_output'])

    def getOutputDirStructure(self):
        # self.vizInput['outputDirStructure'] = path_to_dict(self.vizInput['userOptions']['output_folder'])
        self.vizInput['outputDirStructure'] = path_to_dict(self.vizInput['userOptions']['output_folder'],d = {'dirs':{},'files':[]})
        # print(self.vizInput['outputDirStructure'])

    def writeVizInputFile(self, fname_output):
        with open(fname_output, 'w') as f:
            yaml.dump(self.vizInput, f, default_flow_style=False)


# def path_to_dict(path):
#     d = {'name': os.path.basename(path)}
#     if os.path.isdir(path):
#         d['type'] = "folder"
#         d['content'] = [path_to_dict(os.path.join(path, x)) for x in os.listdir(path)]
#     else:
#         d['type'] = "file"
#     return d

def path_to_dict(path, d):

    name = os.path.basename(path)

    if os.path.isdir(path):
        if name not in d['dirs']:
            d['dirs'][name] = {'dirs':{},'files':[]}
        for x in os.listdir(path):
            path_to_dict(os.path.join(path,x), d['dirs'][name])
    else:
        d['files'].append(name)
    return d

# the mian function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WEIS Visualization App')
    parser.add_argument('--modeling_options', type=str, default='modeling_options.yaml', help='Modeling options file')
    parser.add_argument('--analysis_options', type=str, default='analysis_options.yaml', help='Analysis options file')
    parser.add_argument('--wt_input', type=str, default='wt_input.yaml', help='Wind turbine input file')
    parser.add_argument('--output', type=str, default='vizInputFile.yaml', help='Output file name')

    args = parser.parse_args()

    # generate the viz input file
    viz = WEISVizInputFileGenerator(args.modeling_options, args.analysis_options, args.wt_input)
    viz.fetchWEISinputs()

    # Process the user options of importance to the visualization tool
    viz.userOptions()

    # Set the default user preferences
    viz.setDefaultUserPreferencs()

    # Get the output directory structure
    viz.getOutputDirStructure()

    # Write the viz input file
    viz.writeVizInputFile(args.output)
