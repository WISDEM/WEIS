
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
    fname_modeling_options      = mydir + os.sep + "modeling_options.yaml"
    modeling_options            = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input              = mydir + os.sep + "IEA-15-floating.yaml"
    wt_init                     = sch.load_geometry_yaml(fname_wt_input)

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
    wind_directory = '/Users/dzalkind/Tools/WEIS-1/examples/13_DTQP/outputs/oloc/wind'
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
        ind = u_h > 25
        
        # remove any windspeeds > 25 m/s
        u_h[ind] = 25
        
        
        tt          = ts_file['t']
        level2_disturbance.append({'Time':tt, 'Wind': u_h})


    # Linear Model
    
    # number of linear wind speeds and other options below assume that you haven't changed 
    # the Level2: linearization: options from example 12_linearization
    # if the modelling input was changed, or different between 12_ and 13_, these options will produce unexpected results
    n_lin_ws = len(modeling_options['Level2']['linearization']['wind_speeds'])
    lin_case_name = case_naming(n_lin_ws,'lin')
    OutputCon_flag = False
    
    lin_pickle = mydir + os.sep + "LinearTurbine_22.pkl"

    if True and os.path.exists(lin_pickle):
        with open(lin_pickle,"rb") as pkl_file:
            LinearTurbine = pickle.load(pkl_file)
    
    else:   # load the model and run MBC3
        LinearTurbine = LinearTurbineModel(
                    os.path.join(weis_dir,'outputs/IEA_level2_dtqp_full'),  # directory where linearizations are
                    lin_case_name,
                    nlin=modeling_options['Level2']['linearization']['NLinTimes'],
                    reduceControls = True
                    )

        with open(lin_pickle,"wb") as pkl_file:
            pickle.dump(LinearTurbine,pkl_file)
    
    
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

    if os.path.isabs(modeling_options['General']['openfast_configuration']['OF_run_dir']):
        run_directory = modeling_options['General']['openfast_configuration']['OF_run_dir']
    else:
        run_directory = os.path.join(weis_dir,modeling_options['General']['openfast_configuration']['OF_run_dir'])
    run_directory = modeling_options['General']['openfast_configuration']['OF_run_dir']
    
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

    # save disturbances
    for i_dist, dist in enumerate(level2_disturbance):
        pd.DataFrame(dist).to_pickle(os.path.join(run_directory,f'dist_{i_dist}.p'))

    summary_stats, extreme_table, DELs, Damage, chan_time = dtqp_wrapper(
        LinearTurbine, 
        level2_disturbance, 
        analysis_options, 
        modeling_options,
        fst_vt, 
        la, 
        magnitude_channels, 
        run_directory,
        cores = 1
        )

    # Save each timeseries as a pickled dataframe
    for i_ts, timeseries in enumerate(chan_time):
        timeseries.df.to_pickle(os.path.join(run_directory,FAST_namingOut + '_' + str(i_ts) + '.p'))
    
