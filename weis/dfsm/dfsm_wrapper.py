import os
import numpy as np
import matplotlib.pyplot as plt
import time as timer
import pickle
import shutil
import pandas as pd
import logging

from rosco.toolbox import control_interface as ROSCO_ci
from weis.aeroelasticse.CaseGen_General import case_naming
from openfast_io.FAST_writer          import InputWriter_OpenFAST
from pCrunch import Crunch,FatigueParams, AeroelasticOutput
from weis.dlc_driver.dlc_generator    import DLCGenerator
from openfast_io.turbsim_file   import TurbSimFile
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.CaseGen_General import CaseGen_General

from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,calculate_time,valid_extension_DISCON,compile_dfsm_results
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.ode_algorithms import RK4,ABM4
from weis.dfsm.generate_wave_elev import generate_wave_elev
from wisdem.inputs import load_yaml, write_yaml

from scipy.interpolate import CubicSpline, interp1d

logger = logging.getLogger("wisdem/weis")


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

def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    t1 = timer.time()
    if case_data['ode_method'] == 'RK4':

        output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

    elif case_data['ode_method'] == 'ABM4':

        output = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
    t2 = timer.time()
    print(t2-t1)
    # shut down controller
    case_data['param']['controller_interface'].kill_discon()


    return output


def run_serial(case_data_all):

    # initialize storage array for results
    outputs = []

    # loop through and evaluate simulations    
    for case_data in case_data_all:
        
        # initialize controller interface
        case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])
        t1 = timer.time()
        if case_data['ode_method'] == 'RK4':
            t,x,u,y = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

        elif case_data['ode_method'] == 'ABM4':
            t,x,u,y = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        t2 = timer.time()
        print(t2-t1)
        
        # shut down controller
        case_data['param']['controller_interface'].kill_discon()

        # initialize
        sim_results = {}
        sim_results['case_no'] = case_data['case']
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)
    
    return outputs

def run_mpi(case_data_all,mpi_options):

    from openmdao.utils.mpi import MPI

    # mpi comm management
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_ranks = mpi_options['mpi_comm_map_down'][rank]

    size = len(sub_ranks)

    N_cases = len(case_data_all)
    N_loops = int(np.ceil(float(N_cases)/float(size)))

    sim_results = []
    for i in range(N_loops):
        idx_s    = i*size
        idx_e    = min((i+1)*size, N_cases)

        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            data   = [evaluate_multi, case_data]
            rank_j = sub_ranks[j]
            comm.send(data, dest=rank_j, tag=0)

        # for rank_j in sub_ranks:
        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            rank_j = sub_ranks[j]
            data_out = comm.recv(source=rank_j, tag=1)
            sim_results.append(data_out)

    # compile and store results
    outputs = []
    
    for icase,result_ in enumerate(sim_results):
        t = result_[0]
        x = result_[1]
        u = result_[2]
        y = result_[3]

        # initialize
        sim_results = {}

        # store
        sim_results['case_no'] = icase
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)



    return outputs




def  generate_wind_files_local(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory, FAST_namingOut, wind_directory):

    DLCs = modopt['DLC_driver']['DLCs']

    # Initialize the DLC generator
    cut_in = float(inputs['V_cutin'])
    cut_out = float(inputs['V_cutout'])
    rated = float(inputs['Vrated'])
    ws_class = discrete_inputs['turbine_class']
    wt_class = discrete_inputs['turbulence_class']
    hub_height = float(inputs['hub_height'])
    rotorD = float(inputs['Rtip'])*2.
    PLExp = float(inputs['shearExp'])
    fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
    metocean = modopt['DLC_driver']['metocean_conditions']

    if modopt['OpenFAST']['from_openfast']:
        reg_traj = os.path.join(modopt['OpenFAST']['regulation_trajectory'])
        if os.path.isfile(reg_traj):
            data = load_yaml(reg_traj)
            cases = data['cases']
            U_interp = [case["configuration"]["wind_speed"] for case in cases]
            pitch_interp = [case["configuration"]["pitch"] for case in cases]
            rot_speed_interp = [case["configuration"]["rotor_speed"] for case in cases]
            Ct_aero_interp = [case["outputs"]["integrated"]["ct"] for case in cases]
        else:
            logger.warning("A yaml file with rotor speed, pitch, and Ct is required in modeling options->OpenFAST->regulation_trajectory.",
                    " This file does not exist. Check WEIS example 02 for a template file")
            U_interp = np.arange(cut_in, cut_out)
            pitch_interp = np.ones_like(U_interp) * 5. # fixed initial pitch at 5 deg
            rot_speed_interp = np.ones_like(U_interp) * 5. # fixed initial omega at 5 rpm
            Ct_aero_interp = np.ones_like(U_interp) * 0.7 # fixed initial ct at 0.7
    else:
        U_interp = inputs['U']
        pitch_interp = inputs['pitch']
        rot_speed_interp = inputs['Omega']
        Ct_aero_interp = inputs['Ct_aero']
    
    tau1_const_interp = np.zeros_like(Ct_aero_interp)
    for i in range(len(Ct_aero_interp)):
        a = 1. / 2. * (1. - np.sqrt(1. - np.min([Ct_aero_interp[i],1])))    # don't allow Ct_aero > 1
        tau1_const_interp[i] = 1.1 / (1. - 1.3 * np.min([a, 0.5])) * inputs['Rtip'][0] / U_interp[i]

    initial_condition_table = {}
    initial_condition_table['U'] = U_interp
    initial_condition_table['pitch_initial'] = pitch_interp
    initial_condition_table['rot_speed_initial'] = rot_speed_interp
    initial_condition_table['Ct_aero'] = Ct_aero_interp
    initial_condition_table['tau1_const'] = tau1_const_interp
    
    # DLC generator
    dlc_generator = DLCGenerator(
        cut_in, 
        cut_out, 
        rated, 
        ws_class, 
        wt_class, 
        fix_wind_seeds, 
        fix_wave_seeds, 
        metocean, 
        modopt['DLC_driver'],
        initial_condition_table,
        )


    # dlc_generator = DLCGenerator(cut_in, cut_out, rated, ws_class, wt_class, fix_wind_seeds, fix_wave_seeds, metocean)
    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

    # Initialize parametric inputs
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases

    TMax = np.zeros(dlc_generator.n_cases)
    TStart = np.zeros(dlc_generator.n_cases)

    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].turbulent_wind:
            # Assign values common to all DLCs
            # Wind turbulence class
            if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
                dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
                dlc_generator.cases[i_case].IEC_WindType = 'NTM'
            else:
                dlc_generator.cases[i_case].IECturbc = wt_class
            # Reference height for wind speed
            if not dlc_generator.cases[i_case].RefHt:   # default RefHt is 0, use hub_height if not set
                dlc_generator.cases[i_case].RefHt = hub_height
            # Center of wind grid (TurbSim confusingly calls it HubHt)
            if not dlc_generator.cases[i_case].HubHt:   # default HubHt is 0, use hub_height if not set
                dlc_generator.cases[i_case].HubHt = hub_height

            if not dlc_generator.cases[i_case].GridHeight:   # default GridHeight is 0, use hub_height if not set
                dlc_generator.cases[i_case].GridHeight =  2. * hub_height - 1.e-3

            if not dlc_generator.cases[i_case].GridWidth:   # default GridWidth is 0, use hub_height if not set
                dlc_generator.cases[i_case].GridWidth =  2. * hub_height - 1.e-3
            # Power law exponent of wind shear
            if dlc_generator.cases[i_case].PLExp < 0:    # use PLExp based on environment options (shear_exp), otherwise use custom DLC PLExp
                dlc_generator.cases[i_case].PLExp = PLExp
            # Length of wind grids
            dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].total_time
    
    turbsim_exe = shutil.which('turbsim')
    print('generating wind files')
    for i_case in range(dlc_generator.n_cases):
            WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
                dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, turbsim_exe,i_case)
            
    # Set initial rotor speed and pitch if the WT operates in this DLC and available,
    # otherwise set pitch to 90 deg and rotor speed to 0 rpm when not operating
    # set rotor speed to rated and pitch to 15 deg if operating

    # Parameteric inputs
    case_name = []
    case_list = []
    FAST_InputFile = modopt['General']['openfast_configuration']['OF_run_fst']
    for i_case, case_inputs in enumerate(dlc_generator.openfast_case_inputs):
            # Generate case list for DLC i
            dlc_label = DLCs[i_case]['DLC']
            case_list_i, case_name_i = CaseGen_General(case_inputs, FAST_runDirectory, FAST_InputFile, filename_ext=f'_DLC{dlc_label}_{i_case}')
            # Add DLC to case names
            case_name_i = [f'DLC{dlc_label}_{i_case}_{cni}' for cni in case_name_i]   # TODO: discuss case labeling with stakeholders
            
            # Extend lists of cases
            case_list.extend(case_list_i)
            case_name.extend(case_name_i)

    # Apply wind files to case_list (this info will be in combined case matrix, but not individual DLCs)
    for case_i, wt, wf in zip(case_list,WindFile_type,WindFile_name):
        case_i[('InflowWind','WindType')] = wt
        case_i[('InflowWind','FileName_Uni')] = wf
        case_i[('InflowWind','FileName_BTS')] = wf

    # Save some case info
    TMax = [c.total_time for c in dlc_generator.cases]
    TStart = [c.transient_time for c in dlc_generator.cases]
    dlc_label = [c.label for c in dlc_generator.cases]

    # Merge various cases into single case matrix
    case_df = pd.DataFrame(case_list)
    case_df.index = case_name
    # Add case name and dlc label to front for readability
    case_df.insert(0,'DLC',dlc_label)
    case_df.insert(0,'case_name',case_name)
    text_table = case_df.to_string(index=False)

    # Write the text table to a yaml, text file
    write_yaml(case_df.to_dict(),os.path.join(FAST_runDirectory,'case_matrix_combined.yaml'))
    with open(os.path.join(FAST_runDirectory,'case_matrix_combined.txt'), 'w') as file:
        file.write(text_table)

    for case in case_list:
        for key in list(case):
            if key[0] in ['DLC','TurbSim']:
                del case[key]
            
    return dlc_generator, case_list, case_name,TMax, TStart,rotorD,hub_height



def write_FAST_files(fst_vt, FAST_runDirectory, FAST_namingOut):
    writer = InputWriter_OpenFAST()
    writer.fst_vt = fst_vt 
    writer.FAST_runDirectory = FAST_runDirectory
    writer.FAST_namingOut    = FAST_namingOut

    writer.execute()



def extract_simulation_results(modeling_data_path, model_options):

    # Get the path to the outb files in the folder, and arrange them
    outfiles = [os.path.join(modeling_data_path,f) for f in os.listdir(modeling_data_path) if valid_extension(f)]
    outfiles = sorted(outfiles)

    # Get the states, controls and outputs of the DFSM model
    reqd_states = model_options['reqd_states']
    reqd_controls = model_options['reqd_controls']
    reqd_outputs = model_options['reqd_outputs']


    # Get scaling and filtering arguments
    # --these arguments are used to scale and filter specific signals before constructing the DFSM model
    scale_args = model_options['scale_args']
    filter_args = model_options['filter_args']

    # Get additional options

    # 1. file name: Referes to file that has the linear models are available
    filename = model_options['linear_model_file']
    if filename == 'none':
        filename = None

    # 2. region: The region for which the linear model stored in the file is constructed
    region = model_options['region']

    # 3. add_dx2: Flag to add the first time/second time state derivatives to the model
    add_dx2 = model_options['add_dx2']

    # 4. tmin and tmax: If specified, the simulation data corresponding to t \in [tmin,tmax] is used to construct the DFSM
    # -- tmin is by default 0, and tmax is none
    tmin = model_options['tmin']; tmax = model_options['tmax']

    # Extract the simulation
    # instantiate class
    sim_detail = SimulationDetails(outfiles,
                                reqd_states,reqd_controls,reqd_outputs,
                                scale_args, filter_args,
                                tmin = tmin, add_dx2 = add_dx2,
                                linear_model_file = filename, region = region)

    # load and process data
    sim_detail.load_openfast_sim()

    return sim_detail

def construct_dfsm_helper(sim_detail, construction_options):

    # Extract options
    n_samples = construction_options['n_samples']

    # Sampling method
    sampling_method = construction_options['sampling_method']

    # Linear model type
    L_type = construction_options['L_type']

    # Nonlinear model type 
    N_type = construction_options['N_type']

    if N_type == 'None':
        N_type = None

    # Train split
    train_split = construction_options['train_split']


    # instantiate DFSM class
    dfsm = DFSM(sim_detail,
                n_samples = n_samples,
                sampling_method = sampling_method,
                L_type = L_type,
                N_type = N_type,
                train_split = train_split)
    
    # construct
    dfsm.construct_surrogate()

    return dfsm



def dfsm_wrapper(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory = None, FAST_namingOut = None,mpi_options = None):

    print('----------------------------------------------------')
    print('Running DFSM')
    print('Derivative Function Surrogate Model')
    print('This program is licensed under Apache License Version 2.0 and comes with ABSOLUTELY NO WARRANTY.')
    print('----------------------------------------------------')

    # Load the stored DFSM model and run simulations with it
    general_options = modopt['DFSM']['general_options']
    model_options = modopt['DFSM']['model_options']


    for chan in ['TTDspFA','TTDspSS','NcIMURAys','YawBrTAxp']:
        fst_vt['outlist']['ElastoDyn'][chan] = True
    
    fst_vt['outlist']['SeaState']['Wave1Elev'] = True
    # set run dir. THis is the directory where OpenFAST files are stored
    general_options['run_dir'] = modopt['General']['openfast_configuration']['OF_run_dir']
    modopt_dir = os.path.dirname(modopt['fname_input_modeling'])


    # Extract datapath
    # --this is the folder in which the .outb files are available

    # load DFSM model
    dfsm_file = general_options['dfsm_file']

    with open(os.path.join(modopt_dir, dfsm_file),'rb') as handle:
        dfsm = pickle.load(handle)

    reqd_states =  dfsm.reqd_states
    reqd_controls = dfsm.reqd_controls
    reqd_outputs = dfsm.reqd_outputs
    
    ode_method = model_options['ode_method']

    # setup interpolation method for LPV model
    #dfsm.setup_LPV(interp_type)

    
    # If the test data is online, this implies, the DFSM will be used to run simulations
    # and save them in the current openfast run directory 
    testing_data_path = FAST_runDirectory

    # folder to save dfsm results
    dfsm_save_folder = testing_data_path + os.sep + 'dfsm_results'


    #------------------------------------------------------
    # Use case
    #------------------------------------------------------

    # Get usecase
    usecase = general_options['usecase']

    if usecase == 'closed-loop-simulation' :

        # generate wind files
        wind_directory = FAST_runDirectory + os.sep + 'wind'
        dlc_generator, case_list, case_name, TMax, TStart,rotorD,hub_height = generate_wind_files_local(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory, FAST_namingOut, wind_directory)
        fst_vt['Fst']['TMax'] = TMax[0]
        fst_vt['Fst']['TStart'] = TStart[0]
        # Extract disturbance(s)
        test_dataset = []

        # initialize random number generator
        rng = np.random.default_rng(12345)

        wv_files = []

        for i_case,case in enumerate(case_list):
            ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
            rot_avg = compute_rot_avg(ts_file['u'],ts_file['y'],ts_file['z'],ts_file['t'],rotorD,hub_height)
            u_h         = rot_avg[0,:]
            t          = ts_file['t']
           
            if fst_vt['SeaState']['WaveMod'] == 5:
                DT = fst_vt['SeaState']['WaveDT']
                tmax_ss = fst_vt['SeaState']['WaveTMax']

                tt = np.arange(0,tmax_ss + DT,DT)

                eta = generate_wave_elev(tt,dlc_generator.cases[i_case].wave_height,dlc_generator.cases[i_case].wave_period,rng)
                c_name = case_name[i_case]
                wvkinfile = FAST_runDirectory + os.sep + c_name +'_Wave.Elev'
                wvkinfile1 = FAST_runDirectory + os.sep + c_name +'_Wave'
                wv_files.append(wvkinfile1)

                with open(wvkinfile, 'w') as file:
                    file.write('Time Wave1Elev\n')  # Write the header
                    for t_, elev in zip(tt, eta):
                        file.write(f'{t_:.6f} {elev:.6f}\n')

                eta = CubicSpline(tt,eta)(t)

            else:
                
                eta = generate_wave_elev(t,dlc_generator.cases[i_case].wave_height,dlc_generator.cases[i_case].wave_period,rng)

            dt = fst_vt['Fst']['DT']
            test_dataset.append({'time':t, 'wind_speed': u_h,'wave_elev':eta})

        # number of test cases
        test_ind = np.arange(len(case_list))

        # generate OpenFAST files
        # This step generates the DISCON.IN and cp-ct-cq.txt files which are need to run closed-loop simulations
        
        for i_case,case in enumerate(case_name):

            fst_vt['InflowWind']['WindType'] = case_list[i_case][('InflowWind','WindType')]
            fst_vt['InflowWind']['FileName_BTS'] = case_list[i_case][('InflowWind','FileName_BTS')]
            fst_vt['InflowWind']['WindType'] = case_list[i_case][('InflowWind','WindType')]
            if fst_vt['SeaState']['WaveMod'] == 5:
                fst_vt['SeaState']['WvKinFile'] = wv_files[i_case]
                write_FAST_files(fst_vt, FAST_runDirectory, case)

            else:

                write_FAST_files(fst_vt, FAST_runDirectory, case)
        
        case_names = case_naming(len(case_name),'dfsm')

        
        # initialize storage lists
        output_list = []
        ct = []
        
        # required maginitude and fatigue channels
        #-- note: Some of these quantities are not always modeled using the DFSM. The corresponding outputs for these quantities will be 'Nan'
        magnitude_channels = {'LSShftF': ['RotThrust', 'LSShftFys', 'LSShftFzs'], 
                'LSShftM': ['RotTorq', 'LSSTipMys', 'LSSTipMzs'],
                'RootMc1': ['RootMxc1', 'RootMyc1', 'RootMzc1'],
                'RootMc2': ['RootMxc2', 'RootMyc2', 'RootMzc2'],
                'RootMc3': ['RootMxc3', 'RootMyc3', 'RootMzc3'],
                'TipDc1': ['TipDxc1', 'TipDyc1', 'TipDzc1'],
                'TipDc2': ['TipDxc2', 'TipDyc2', 'TipDzc2'],
                'TipDc3': ['TipDxc3', 'TipDyc3', 'TipDzc3'], 
                'TwrBsM': [ 'TwrBsMyt'],
                'PtfmOffset': ['PtfmSurge', 'PtfmSway'],
                'NcIMUTA': ['NcIMUTAxs', 'NcIMUTAys', 'NcIMUTAzs']}

        fatigue_channels = {
                    'RootMc1': FatigueParams(slope=10),
                    'RootMc2': FatigueParams(slope=10),
                    'RootMc3': FatigueParams(slope=10),
                    'RootMyb1': FatigueParams(slope=10),
                    'RootMyb2': FatigueParams(slope=10),
                    'RootMyb3': FatigueParams(slope=10),
                    'TwrBsM': FatigueParams(slope=4),
                    'LSShftM': FatigueParams(slope=4),
                                        }

        DISCON_file = [os.path.join(testing_data_path,f) for f in os.listdir(testing_data_path) if valid_extension_DISCON(f)]
        GB_ratio = fst_vt['DISCON_in']['WE_GearboxRatio']
        case_data_all = []

        for idx,ind in enumerate(test_ind):

            test_data = test_dataset[ind]

            time = test_data['time']
            wind_speed = test_data['wind_speed']
            wind_fun = CubicSpline(time, wind_speed)

            x0 = np.zeros((dfsm.n_deriv,))

            t0 = time[0]; 
            tf = time[-1]
            tspan = [t0,tf]

            args = {'DT':dt,
                        'num_blade':3,'pitch':15}


            if 'Wave1Elev' in reqd_controls:
                wave_elev = test_data['wave_elev']
                wave_fun = CubicSpline(time,wave_elev)

            # initialize param dict
            param = {}

            # populate dictonary with relevant info
            param['VS_GenEff'] = fst_vt['DISCON_in']['VS_GenEff']
            param['WE_GearboxRatio'] = fst_vt['DISCON_in']['WE_GearboxRatio']
            param['VS_RtPwr'] = fst_vt['DISCON_in']['VS_RtPwr']
            param['time'] = [t0]
            param['dt']= dt
            param['blade_pitch'] = [15]
            param['gen_torque'] = [19000]
            param['t0'] = t0
            param['tf'] = tf 
            param['gen_speed_scaling'] = 1
            param['lib_name'] = fst_vt['ServoDyn']['DLL_FileName']
            param['num_blade'] = 3
            param['ny'] = dfsm.n_outputs
            param['args'] = args
            param['param_filename'] = DISCON_file[idx]
            param['w_fun'] = wind_fun
            if 'Wave1Elev' in reqd_controls:
                param['wave_fun'] = wave_fun
            else:
                param['wave_fun'] = None



            # # start timer and solve for the states and controls
            # t1 = timer.time()
            # T_dfsm, states_dfsm, controls_dfsm,outputs_dfsm = RK4(x0, dt, tspan, dfsm, param)
            # t2 = timer.time()
            # dfsm.simulation_time = (t2-t1)

            case_data = {}
            case_data['case'] = idx
            case_data['param'] = param 
            case_data['dt'] = dt 
            case_data['x0'] = x0 
            case_data['tspan'] = tspan 
            case_data['dfsm'] = dfsm
            case_data['ode_method'] = ode_method


            case_data_all.append(case_data)
        
        if mpi_options['mpi_run']:

            # evaluate the closed loop simulations in parallel using MPI
            sim_outputs = run_mpi(case_data_all,mpi_options)

        else:

            # evaluate the closed loop simulations serially
            sim_outputs = run_serial(case_data_all)

        # plot properties
        markersize = 10
        linewidth = 1.5
        fontsize_legend = 16
        fontsize_axlabel = 18
        fontsize_tick = 12

        # save results
        if not os.path.exists(dfsm_save_folder):
            os.makedirs(dfsm_save_folder)

        for icase,sim_result in enumerate(sim_outputs):
            
            # extract results
            T_dfsm = sim_result['T_dfsm']
            states_dfsm = sim_result['states_dfsm']
            controls_dfsm = sim_result['controls_dfsm']
            outputs_dfsm = sim_result['outputs_dfsm']

        

            for iu,control in enumerate(reqd_controls):
        
                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,controls_dfsm[:,iu],label = 'DFSM')
                
                ax.set_title(control,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

                if general_options['save_results']:
                    fig.savefig(os.path.join(dfsm_save_folder, f"{icase}_{control}_comp.pdf"))

                plt.close(fig)

            

            #------------------------------------------------------
            # Plot States
            #------------------------------------------------------
            for ix,state in enumerate(reqd_states):
                    
                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,states_dfsm[:,ix],label = 'DFSM')
                
                ax.set_title(state,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if general_options['save_results']:

                    fig.savefig(os.path.join(dfsm_save_folder, f"{icase}_{state}_comp.pdf"))

                plt.close(fig)

            #------------------------------------------
            # Plot Outputs
            #------------------------------------------
            for iy,output_ in enumerate(reqd_outputs):

                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
                
                
                ax.set_title(output_,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if general_options['save_results']:

                    fig.savefig(os.path.join(dfsm_save_folder, f"{icase}_{output_}_comp.pdf"))

                plt.close(fig)

            # post processing
            case_name = case_names[icase]

            # compile results from DFSM
            OutData = compile_dfsm_results(T_dfsm,states_dfsm,controls_dfsm,outputs_dfsm,reqd_states,
                                            reqd_controls,reqd_outputs,GB_ratio,TStart[icase])

            ct.append(OutData)

            # get output
            output = AeroelasticOutput(OutData, dlc = FAST_namingOut, maginitude_channels = magnitude_channels, fatigue_channels = fatigue_channels )

            for i_blade in range(fst_vt['ElastoDyn']['NumBl']):
                output.add_gradient_channel(f'BldPitch{i_blade+1}', f'dBldPitch{i_blade+1}')

            output_list.append(output)
        

    cruncher = Crunch(outputs = [],lean = True)

    for output in output_list:
        cruncher.add_output(output)

    cruncher.outputs = output_list
    return cruncher,case_list, case_names, ct, dlc_generator, TMax, TStart






