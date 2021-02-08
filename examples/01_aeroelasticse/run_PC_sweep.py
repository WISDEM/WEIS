"""

Example script to run the DLCs with different pitch control gains in OpenFAST

"""

from weis.aeroelasticse.runFAST_pywrapper   import runFAST_pywrapper, runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_IEC         import CaseGen_IEC
from wisdem.commonse.mpi_tools              import MPI
import sys, os, platform, yaml
import numpy as np
from pCrunch.CaseGen_Control import append_case_matrix_yaml

# ROSCO 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_Utilities

def run_PC_sweep(omega,zeta=1.0):
    # Process inputs, single (omega, zeta) or multiple?
    
    
    # Turbine inputs
    iec = CaseGen_IEC()
    iec.Turbine_Class       = 'I'   # Wind class I, II, III, IV
    iec.Turbulence_Class    = 'B'   # Turbulence class 'A', 'B', or 'C'
    iec.D                   = 240.  # Rotor diameter to size the wind grid
    iec.z_hub               = 150.  # Hub height to size the wind grid
    cut_in                  = 3.    # Cut in wind speed
    cut_out                 = 25.   # Cut out wind speed
    n_ws                    = 3    # Number of wind speed bins
    TMax                    = 800.    # Length of wind grids and OpenFAST simulations, suggested 720 s
    Vrated                  = 10.59 # Rated wind speed
    Ttrans                  = max([0., TMax - 60.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
    TStart                  = max([0., TMax - 600.]) # Start of the recording of the channels of OpenFAST
    iec.cores               = 1
    iec.overwrite           = False
    
    # Initial conditions to start the OpenFAST runs
    u_ref     = np.arange(3.,26.) # Wind speed
    pitch_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5058525323662666, 5.253759185225932, 7.50413344606208, 9.310153958810268, 10.8972969450052, 12.412247669440042, 13.883219268525659, 15.252012626933068, 16.53735488246438, 17.76456777500061, 18.953261878035104, 20.11055307762722, 21.238680277668898, 22.30705111326602, 23.455462501156205] # Pitch values in deg
    omega_ref = [2.019140272160114, 2.8047214918577925, 3.594541645994511, 4.359025795823625, 5.1123509774611025, 5.855691196288371, 6.589281196735111, 7.312788026081227, 7.514186181824161, 7.54665511646938, 7.573823812448151, 7.600476033113538, 7.630243938880304, 7.638301051122195, 7.622050377183605, 7.612285710588359, 7.60743945212863, 7.605865650155881, 7.605792924227456, 7.6062185247519825, 7.607153933765292, 7.613179734210654, 7.606737845170748] # Rotor speeds in rpm
    iec.init_cond = {}
    iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':u_ref}
    iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = omega_ref
    iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':u_ref}
    iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_ref
    iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("HydroDyn","WaveHs")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
    iec.init_cond[("HydroDyn","WaveHs")]['val']    = [1.101917033, 1.101917033, 1.179052649, 1.315715154, 1.536867124, 1.835816514, 2.187994638, 2.598127096, 3.061304068, 3.617035443, 4.027470219, 4.51580671, 4.51580671, 6.98, 10.7]
    iec.init_cond[("HydroDyn","WaveTp")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
    iec.init_cond[("HydroDyn","WaveTp")]['val']    = [8.515382435, 8.515382435, 8.310063688, 8.006300889, 7.6514231, 7.440581338, 7.460834063, 7.643300307, 8.046899942, 8.521314105, 8.987021024, 9.451641026, 9.451641026, 11.7, 14.2]
    iec.init_cond[("HydroDyn","PtfmSurge")]        = {'U':[3., 15., 25.]}
    iec.init_cond[("HydroDyn","PtfmSurge")]['val'] = [4., 15., 10.]
    iec.init_cond[("HydroDyn","PtfmPitch")]        = {'U':[3., 15., 25.]}
    iec.init_cond[("HydroDyn","PtfmPitch")]['val'] = [-1., 3., 1.3]
    iec.init_cond[("HydroDyn","PtfmHeave")]        = {'U':[3., 25.]}
    iec.init_cond[("HydroDyn","PtfmHeave")]['val'] = [0.5,0.5]

    # DLC inputs
    if False:
        wind_speeds = np.linspace(int(cut_in), int(cut_out), int(n_ws))
        iec.dlc_inputs = {}
        iec.dlc_inputs['DLC']   = [1.1, 1.3, 1.4, 1.5, 5.1, 6.1, 6.3]
        iec.dlc_inputs['U']     = [wind_speeds, wind_speeds,[Vrated - 2., Vrated, Vrated + 2.],wind_speeds, [Vrated - 2., Vrated, Vrated + 2., cut_out], [], []]
        iec.dlc_inputs['Seeds'] = [[1],[1],[],[],[1],[1],[1]]
        # iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
        iec.dlc_inputs['Yaw']   = [[], [], [], [], [], [], []]
    else:
        wind_speeds = [18]
        iec.dlc_inputs = {}
        iec.dlc_inputs['DLC']   = [1.1]
        iec.dlc_inputs['U']     = [wind_speeds]
        iec.dlc_inputs['Seeds'] = [[1]]
        # iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
        iec.dlc_inputs['Yaw']   = [[]]

    iec.PC_MaxRat           = 2.
    iec.TStart              = Ttrans
    iec.TMax                = TMax    # wind file length
    iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
    iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS
    iec.wind_dir            = 'outputs/wind'
    # Management of parallelization
    # not using for now, copy from run_DLC.py if needed later
    iec.parallel_windfile_gen = True
    iec.cores = 4

    iec.run_dir = 'outputs/iea15mw/PC_sweep_play'

    # Run case generator / wind file writing
    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
    case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
    # case_inputs[("Fst","DT_Out")]            = {'vals':[0.01], 'group':0}  #0.005  
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}
    case_inputs[("Fst","CompHydro")]         = {'vals':[1], 'group':0}
    case_inputs[("Fst","CompSub")]           = {'vals':[0], 'group':0}
    case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmSgDOF")]   = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","PtfmSwDOF")]   = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","PtfmHvDOF")]   = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","PtfmRDOF")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","PtfmPDOF")]    = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","PtfmYDOF")]    = {'vals':["True"], 'group':0}
    case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
    run_dir1            = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
    if platform.system() == 'Windows':
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dll')
    elif platform.system() == 'Darwin':
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dylib')
    else:
        path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.so')

    case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':["True"], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':[1], 'group':0}
    case_inputs[("HydroDyn","WaveMod")]      = {'vals':[2], 'group':0}
    case_inputs[("HydroDyn","WvDiffQTF")]    = {'vals':["False"], 'group':0}
    channels = {}
    for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", "TipRDxr", "TipRDyr", "TipRDzr"]:
        channels[var] = True

    # mass sweep
    # case_inputs[("ElastoDyn","PtfmMass")] = {'vals': (1.7838E+07*np.array([.7,.8,.9,1,1.1,1.2,1.3])).tolist(), 'group':3}
    # case_inputs[("ElastoDyn","PtfmRIner")] = {'vals':(1.2507E+10*np.array([.7,.8,.9,1,1.1,1.2,1.3])).tolist(), 'group':3}
    # case_inputs[("ElastoDyn","PtfmPIner")] = {'vals':(1.2507E+10*np.array([.7,.8,.9,1,1.1,1.2,1.3])).tolist(), 'group':3}
    # case_inputs[("ElastoDyn","PtfmYIner")] = {'vals':(2.3667E+10*np.array([.7,.8,.9,1,1.1,1.2,1.3])).tolist(), 'group':3}


    # controller params

    # load default params
    print('here')
    rt_dir              = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'ROSCO_toolbox')
    rt_dir              = '/Users/dzalkind/Tools/ROSCO_toolbox'
    weis_dir            = os.path.join(rt_dir,'Tune_Cases')
    control_param_yaml  = os.path.join(rt_dir,'Tune_Cases','IEA15MW.yaml')
    inps                = yaml.safe_load(open(control_param_yaml))
    path_params         = inps['path_params']
    turbine_params      = inps['turbine_params']
    controller_params   = inps['controller_params']

    # make default controller, turbine objects for ROSCO_toolbox
    turbine             = ROSCO_turbine.Turbine(turbine_params)
    turbine.load_from_fast( path_params['FAST_InputFile'],
                            os.path.join(rt_dir, path_params['FAST_directory']), dev_branch=True)

    controller          = ROSCO_controller.Controller(controller_params)

    # tune default controller
    controller.tune_controller(turbine)

    # check if inputs are lists
    if not isinstance(omega,list):
        omega = [omega]
    if not isinstance(zeta,list):
        zeta = [zeta]

    # Loop through and make PI gains
    pc_kp = []
    pc_ki = []
    omega_zeta = []  # flattened (omega,zeta) pairs
    for o in omega:
        for z in zeta:
            controller.omega_pc = o
            controller.zeta_pc  = z
            controller.tune_controller(turbine)
            pc_kp.append(controller.pc_gain_schedule.Kp.tolist())
            pc_ki.append(controller.pc_gain_schedule.Ki.tolist())
            omega_zeta.append((o,z))

    # add control gains to case_list
    case_inputs[('DISCON_in', 'PC_GS_KP')] = {'vals': pc_kp, 'group': 3}
    case_inputs[('DISCON_in', 'PC_GS_KI')] = {'vals': pc_ki, 'group': 3}

    iec.case_name_base = 'pc_play'

    # generate cases
    case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)

    #for var in var_out+[var_x]:

    # Run FAST cases
    fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)

    # Monopile
    # fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
    # run_dir2                    = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + os.sep
    # fastBatch.FAST_directory    = os.path.join(run_dir2, 'OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-Monopile')   # Path to fst directory files
    fastBatch.channels          = channels
    # fastBatch.FAST_runDirectory = iec.run_dir
    fastBatch.case_list         = case_list
    fastBatch.case_name_list    = case_name_list
    fastBatch.debug_level       = 2

    # if MPI:
    #     fastBatch.run_mpi(comm_map_down)
    # else:
    #     fastBatch.run_serial()

    # U-Maine semi-sub
    fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    run_dir2                    = os.path.dirname( os.path.realpath(__file__) ) + os.sep
    fastBatch.FAST_directory    = os.path.join(run_dir2, 'OpenFAST_models','IEA-15-240-RWT','IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
    fastBatch.FAST_runDirectory = iec.run_dir
    if True:
        fastBatch.run_multi(cores=4)
    else:
        fastBatch.run_serial()

if __name__ == '__main__':
# Actually run example
    run_PC_sweep([.1,.15,.2,.25,.3,.35,.4,.45])      
    
    
    
    
