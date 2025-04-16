
import weis.inputs as sch
import os
from weis.aeroelasticse.utils import generate_wind_files
from openfast_io.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.control.LinearModel           import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse.CaseGen_General import case_naming
from weis.control.dtqp_wrapper          import dtqp_wrapper
import pickle
from pCrunch import Crunch, FatigueParams
import pandas as pd
import shutil
import matplotlib.pyplot as plt


import numpy as np

this_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_dir))

def compute_rot_avg(u,y,z,t,R,HubHt):
    ''' 
    Compute rotor average wind speed, where R is the rotor radius
    '''

    rot_avg = np.zeros((3,len(t)))
    
    for i in range(3):
        u_      = u[i,:,:,:]
        yy, zz = np.meshgrid(y,z)
        rotor_ind = np.sqrt(yy**2 + (zz - HubHt)**2) < R

        u_rot = []
        for u_plane in u_:
            u_rot.append(u_plane[rotor_ind].mean())

        rot_avg[i,:] = u_rot

    return rot_avg

if __name__ == "__main__":

    # read WEIS options:
    mydir                       = this_dir  # get path to this file
    fname_modeling_options      = os.path.join(mydir, "modeling_options.yaml")
    modeling_options            = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input              = os.path.join(mydir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S.yaml")
    wt_init                     = sch.load_geometry_yaml(fname_wt_input)

    fname_analysis_options      = os.path.join(mydir, "analysis_options.yaml")
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)

    # Wind turbine inputs 
    ws_cut_in               = wt_init["control"]["supervisory"]["Vin"]
    ws_cut_out              = wt_init["control"]["supervisory"]["Vout"]
    ws_rated                = 11.2
    wind_speed_class        = wt_init["assembly"]["turbine_class"]
    wind_turbulence_class   = wt_init["assembly"]["turbulence_class"]

    # Extract user defined list of cases
    DLCs = modeling_options["DLC_driver"]["DLCs"]
    
    # Initialize the generator
    fix_wind_seeds = modeling_options["DLC_driver"]["fix_wind_seeds"]
    fix_wave_seeds = modeling_options["DLC_driver"]["fix_wave_seeds"]
    metocean = modeling_options["DLC_driver"]["metocean_conditions"]
    dlc_generator = DLCGenerator(ws_cut_in, ws_cut_out, ws_rated, wind_speed_class, wind_turbulence_class, fix_wind_seeds, fix_wave_seeds, metocean)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt["DLC"], DLCopt)

    if os.path.isabs(modeling_options["General"]["openfast_configuration"]["OF_run_dir"]):
        run_directory = modeling_options["General"]["openfast_configuration"]["OF_run_dir"]
    else:
        run_directory = os.path.join(weis_dir,modeling_options["General"]["openfast_configuration"]["OF_run_dir"])
    run_directory = modeling_options["General"]["openfast_configuration"]["OF_run_dir"]
    
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

    
    # generate wind files
    FAST_namingOut = "oloc"
    wind_directory = os.path.join(this_dir,"oloc/wind")
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init["assembly"]["rotor_diameter"]
    hub_height = wt_init["assembly"]["hub_height"]

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [""] * dlc_generator.n_cases

    level2_disturbance = []
    turbsim_exe = shutil.which('turbsim')

    fig,ax = plt.subplots(1)
    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
            dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
            dlc_generator.cases[i_case].IEC_WindType = 'NTM'
        else:
            dlc_generator.cases[i_case].IECturbc = wind_turbulence_class


        dlc_generator.cases[i_case].GridHeight = 2. * np.abs(hub_height) - 1.e-3
        dlc_generator.cases[i_case].GridWidth = dlc_generator.cases[i_case].GridHeight
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height,turbsim_exe, i_case)
        
        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(this_dir+os.sep+'oloc' + os.sep +WindFile_name[i_case])
        

        rot_avg = compute_rot_avg(ts_file['u'],ts_file['y'],ts_file['z'],ts_file['t'],rotorD,hub_height)
        
        u_h         = rot_avg[0,:]
        
        
        off = max(u_h) - 25
        ind = u_h > 25
        
        # remove any windspeeds > 25 m/s
        u_h[ind] = 25
        
        
        tt          = ts_file["t"]
        level2_disturbance.append({"Time":tt, "Wind": u_h})

        ax.plot(tt,u_h)


        

    fig.savefig(run_directory + os.sep + 'u_h')
    plt.close(fig)
    

    # Linear Model
    
    # number of linear wind speeds and other options below assume that you haven't changed 
    # the OpenFAST_Linear: linearization: options from example 12_linearization
    # if the modelling input was changed, or different between 12_ and 13_, these options will produce unexpected results
    n_lin_ws = len(modeling_options['OpenFAST_Linear']['linearization']['wind_speeds'])
    lin_case_name = case_naming(n_lin_ws,'lin')
    OutputCon_flag = False
    
    lin_pickle = mydir + os.sep + "LinearTurbine.pkl"

    if True and os.path.exists(lin_pickle):
        with open(lin_pickle,"rb") as pkl_file:
            LinearTurbine = pickle.load(pkl_file)
    
    else:   # load the model and run MBC3
        LinearTurbine = LinearTurbineModel(
                    os.path.join(weis_dir,"outputs/IEA_level2_dtqp_full"),  # directory where linearizations are
                    lin_case_name,
                    nlin=modeling_options['OpenFAST_Linear']['linearization']['NLinTimes'],
                    reduceControls = True
                    )

        with open(lin_pickle,"wb") as pkl_file:
            pickle.dump(LinearTurbine,pkl_file)
    
    
    fst_vt = {}
    fst_vt["DISCON_in"] = {}
    fst_vt["DISCON_in"]["PC_RefSpd"] = 0.7853192931562493


    magnitude_channels = {
        "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
        "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
        "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
        }

    

    # save disturbances
    for i_dist, dist in enumerate(level2_disturbance):
        pd.DataFrame(dist).to_pickle(os.path.join(run_directory,f"dist_{i_dist}.p"))


    Crunch, chan_time = dtqp_wrapper(
        LinearTurbine, 
        level2_disturbance, 
        analysis_options, 
        modeling_options,
        fst_vt,  
        magnitude_channels, 
        run_directory,
        cores = 1
        )
    
    # Save each timeseries as a pickled dataframe
    for i_ts, timeseries in enumerate(chan_time):
        fig, ((ax1,ax2,ax3)) = plt.subplots(3,1,)
        T = timeseries['Time']
        t0 = T[0];tf = T[-1]

        # wind
        ax1.plot(T,timeseries['RtVAvgxh'])
        ax1.set_title('Wind Speed [m/s]')
        ax1.set_xlim([t0,tf])

        # torue
        ax2.plot(T,timeseries['GenTq'])
        #ax2.set_ylim([1.8,2])
        ax2.set_title('Gen Torque [MWm]')
        ax2.set_xlim([t0,tf])

        # blade pitch
        ax3.plot(T,timeseries['BldPitch1'])
        #ax3.set_ylim([0.2, 0.3])
        ax3.set_title('Bld Pitch [deg]')
        ax3.set_xlim([t0,tf])

        fig.subplots_adjust(hspace = 0.65)
        fig.savefig(run_directory + os.sep +'oloc_results_'+str(i_ts)+'.png')
        plt.close(fig)


        timeseries.df.to_pickle(os.path.join(run_directory,FAST_namingOut + "_" + str(i_ts) + ".p"))
    
