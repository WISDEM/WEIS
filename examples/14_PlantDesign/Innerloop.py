#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import weis.inputs as sch
import os
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.control.LinearModel           import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse.CaseGen_General import case_naming
import pickle
from weis.control.dtqp_wrapper          import dtqp_wrapper
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams

import numpy as np

class dict2class(object):
    
    def __init__(self,my_dict):
        
        for key in my_dict:
            setattr(self,key,my_dict[key])
            
        self.A_ops = self.A
        self.B_ops = self.B
        self.C_ops = self.C
        self.D_ops = self.D

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == "__main__":
    # read WEIS options:
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    modeling_options = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input   = mydir + os.sep + "IEA-15-floating.yaml"
    wt_init          = sch.load_geometry_yaml(fname_wt_input)
    
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)

    # Wind turbine inputs 
    ws_cut_in               = wt_init['control']['supervisory']['Vin']
    ws_cut_out              = wt_init['control']['supervisory']['Vout']
    ws_rated                = 11.2
    wind_speed_class        = wt_init['assembly']['turbine_class']
    wind_turbulence_class   = wt_init['assembly']['turbulence_class']

    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']
    
    # Initialize the generator
    fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(ws_cut_in, ws_cut_out, ws_rated, wind_speed_class, wind_turbulence_class, fix_wind_seeds, fix_wave_seeds, metocean)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)


    # generate wind files
    FAST_namingOut = 'oloc'
    wind_directory = 'outputs/oloc/wind'
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init['assembly']['rotor_diameter']
    hub_height = wt_init['assembly']['hub_height']

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases

    level2_disturbance = []

    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)

        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(WindFile_name[i_case])
        ts_file.compute_rot_avg(rotorD/2)
        u_h         = ts_file['rot_avg'][0,:]
        
        
        off = max(u_h) - 25
        ind = u_h > 25;
        
        # remove any windspeeds > 25 m/s
        if ind.any():
            u_h[ind] = u_h[ind] - off
        
        
        tt = ts_file['t']
        level2_disturbance.append({'Time':tt, 'Wind': u_h})
        
        
    # Linear Model
    
    pkl_file = mydir + os.sep + "outputs" + os.sep + "ABCD_matrices.pkl"
    
    with open(pkl_file,"rb") as handle:
        ABCD_list = pickle.load(handle)
    breakpoint()    
    fst_vt = {}
    fst_vt['DISCON_in'] = {}
    fst_vt['DISCON_in']['PC_RefSpd'] = 0.7853192931562493

    la = LoadsAnalysis(
            outputs=[],
        )

    magnitude_channels = {
        "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
        "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
        "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
        }

    run_directory = modeling_options['General']['openfast_configuration']['OF_run_dir']
    
        
    n_cases = len(ABCD_list)
    
    for n in range(n_cases):
        ABCD = ABCD_list[n]
        
        LinearTurbine = dict2class(ABCD)
        
        summary_stats, extreme_table, DELs, Damage = dtqp_wrapper(
        LinearTurbine, 
        level2_disturbance, 
        analysis_options, 
        fst_vt, 
        la, 
        magnitude_channels, 
        run_directory
        )
            
        
        
        
        
