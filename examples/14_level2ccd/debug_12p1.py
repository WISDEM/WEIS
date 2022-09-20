import os
import numpy as np
import matplotlib.pyplot as plt 


# import specific weis classes and functions
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.turbsim_file import TurbSimFile 
from weis.dlc_driver.dlc_generator      import DLCGenerator

from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS

if __name__ == "__main__":
    
    # get directory 
    weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    # read modeling and analysis options
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # modeling options
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    #modeling_options = sch.load_modeling_yaml(fname_modeling_options)
    
    # geometry options
    run_dir                = mydir + os.sep
    wisdem_examples        = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
    fname_wt_input         = os.path.join(wisdem_examples,'09_floating/IEA-15-240-RWT_VolturnUS-S.yaml')
    #wt_init          = sch.load_geometry_yaml(fname_wt_input)
    
    # analysis options
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    #analysis_options            = sch.load_analysis_yaml(fname_analysis_options)
    
    wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
    wt_init,modeling_options,analysis_options = wt_ontology.get_input_data()
    
    # load and generate dlc
    Turbine_class = wt_init["assembly"]["turbine_class"]
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
    wind_directory = 'outputs/debug_12p1'
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init['assembly']['rotor_diameter']
    hub_height = wt_init['assembly']['hub_height']

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases
    
    plotflag = True;wind_saveflag = 1
    level2_disturbance = []

    # plot
    if plotflag:
        
        fig1,ax1 = plt.subplots(1,1)
        ax1.set_ylabel('Time [s]',fontsize = 16)
        ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)

    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time

        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)

        # Compute rotor average wind speed as level2_disturbances
        
        # fname = WindFile_name[i_case]
        
        # #with open(fname,'rb') as handle:
            
        
        # ts_file     = TurbSimFile(WindFile_name[i_case])
        # ts_file.compute_rot_avg(rotorD/2)
        # u_h         = ts_file['rot_avg'][0,:]
        
        # off = 3 - min(u_h)
        # ind = u_h < 3
        
        # # if ind.any():
        # #     u_h[ind] = u_h[ind]+off
        
        # off = max(u_h) - 25
        # ind = u_h > 25;
        
        # # remove any windspeeds > 25 m/s
        # # if ind.any():
        # #     u_h[ind] = u_h[ind] - off
        
        # # save DLC
        # tt = ts_file['t']
        # level2_disturbance.append({'Time':tt, 'Wind': u_h})
        # #print(np.min(u_h)); print(np.max(u_h))
        # # plot
        # if plotflag:
        #     ax1.plot(tt,u_h)