
import weis.inputs as sch
import os
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.control.LinearModel           import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse.CaseGen_General import case_naming
from weis.control.dtqp_wrapper          import dtqp_wrapper
import pickle
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
import pandas as pd


import numpy as np

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':

    # read WEIS options:
    mydir                       = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    fname_modeling_options      = mydir + os.sep + "modeling_options_mhk.yaml"
    modeling_options            = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input              = mydir + os.sep + "mhk_temp_geom.yaml"
    wt_init                     = sch.load_geometry_yaml(fname_wt_input)

    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)

    # Wind turbine inputs 
    ws_cut_in               = wt_init['control']['supervisory']['Vin']
    ws_cut_out              = wt_init['control']['supervisory']['Vout']
    ws_rated                = 1.25
    wind_speed_class        = wt_init['assembly']['turbine_class']
    wind_turbulence_class   = wt_init['assembly']['turbulence_class']

    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']
    
    # Initialize the generator
    fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(
        metocean,
        **{
            "ws_cut_in": ws_cut_in, 
            "ws_cut_out": ws_cut_out, 
            "ws_rated": ws_rated, 
            "wind_speed_class": wind_speed_class, 
            "wind_turbulence_class": wind_turbulence_class, 
            "fix_wind_seeds": fix_wind_seeds, 
            "fix_wave_seeds": fix_wave_seeds, 
            "MHK": wt_init['assembly']['marine_hydro']
        }
        )

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

     # Build case table for RAFT
    raft_cases = [[]]*dlc_generator.n_cases
    for i, icase in enumerate(dlc_generator.cases):
        if 'EWM' in icase.IEC_WindType:
            wind_type = 'EWM'
        else:
            wind_type = icase.IEC_WindType[-3:]
        turb_class = wt_init['assembly']['turbulence_class']
        turbStr = f'{dlc_generator.wind_speed_class}{turb_class}_{wind_type}'
        opStr = 'operating' if icase.turbine_status.lower() == 'operating' else 'parked'
        waveStr = 'JONSWAP' #icase.wave_spectrum
        raft_cases[i] = [icase.URef,
                            icase.wind_heading,
                            turbStr,
                            opStr,
                            icase.yaw_misalign,
                            waveStr,
                            max(1.0, icase.wave_period),
                            max(1.0, icase.wave_height),
                            icase.wave_heading,
                            icase.current]

    with open('/Users/dzalkind/Tools/WEIS-4/examples/16_MHK_DLCs/raft_cases.txt','w') as f:
        raft_dlc_keys = [
            'wind_speed', 
            'wind_heading', 
            'turbulence',
            'turbine_status', 
            'yaw_misalign', 
            'wave_spectrum',
            'wave_period', 
            'wave_height', 
            'wave_heading',
            'current_speed'
            ]
        f.write("{}\n".format('\t'.join(raft_dlc_keys)))
        for rc in raft_cases:
            f.write("{}\n".format('\t'.join([str(r) for r in rc])))

    # # generate wind files
    # FAST_namingOut = 'oloc'
    # wind_directory = '/Users/dzalkind/Tools/WEIS-1/examples/13_DTQP/outputs/oloc/wind'
    # if not os.path.exists(wind_directory):
    #     os.makedirs(wind_directory)
    # rotorD = wt_init['assembly']['rotor_diameter']
    # hub_height = wt_init['assembly']['hub_height']

    # # from various parts of openmdao_openfast:
    # WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    # WindFile_name = [''] * dlc_generator.n_cases

    # level2_disturbance = []

    # for i_case in range(dlc_generator.n_cases):
    #     dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
    #     WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
    #         dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)