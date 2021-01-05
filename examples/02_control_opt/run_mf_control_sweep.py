'''
Functions for running linear and nonlinear control parameter optimizations

- Run full set of DLCs
- Process and find Worst Case
- Nonlinear
    - Tune ROSCO, update
    - Run single, worst case DLC
- Linear  (currently: only doing this!)
    - Generate linear model from nonlinear simulation
    - Tune linear ROSCO
    - Run linear simulation
- Process DEL, other measures for cost function

'''
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from weis.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml
from weis.aeroelasticse.LinearFAST import LinearFAST

# weis control modules
import weis.control.LinearModel as lin_mod


# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 

from ROSCO_toolbox import utilities as ROSCO_utilities
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox.ofTools.fast_io.output_processing import output_processing
fast_io = output_processing()

# WISDEM modules
from weis.aeroelasticse.Util import FileTools

# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis


import numpy as np
import sys, os, platform, glob, yaml

class MF_Turbine(object):

    def __init__(self):
        # Turbine Model
        # Select Turbine Model
        model_dir               = os.path.join(os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ), '01_aeroelasticse/OpenFAST_models')
        self.FAST_directory     = os.path.join(model_dir, 'IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
        self.FAST_InputFile     = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)

        # Parallel Processing
        self.n_cores            = 1

    def gen_level2_model(self,wind_speeds=[5,9,13,17,21],dofs=['GenDOF']):
        lin_fast = LinearFAST(FAST_ver='OpenFAST', dev_branch=True)

        # fast info
        weis_dir                 = os.path.dirname( os.path.dirname ( os.path.dirname( __file__ ) ) ) + os.sep
        
        lin_fast.FAST_InputFile           = self.FAST_InputFile   # FAST input file (ext=.fst)
        lin_fast.FAST_directory           = self.FAST_directory
        lin_fast.FAST_linearDirectory     = os.path.join(weis_dir,'outputs','iea_semi_lin')
        lin_fast.debug_level              = 2
        lin_fast.dev_branch               = True
        lin_fast.write_yaml               = True
        
        lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
        lin_fast.WindSpeeds                 = wind_speeds
        lin_fast.DOFs                       = dofs  # enable with 
        lin_fast.TMax                       = 1600   # should be 1000-2000 sec or more with hydrodynamic states
        lin_fast.NLinTimes                  = 12

        lin_fast.FAST_exe                   = '/Users/dzalkind/Tools/openfast/install/bin/openfast'

        # simulation setup
        lin_fast.cores                      = self.n_cores

        # overwrite steady & linearizations
        lin_fast.overwrite        = False

        lin_fast.gen_linear_model()

    def run_level2(self):
        pass


    def run_level3(self,wind_speeds=[16],omega=[],zeta=1,n_cores=1):
        # Run FAST cases
        fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)
        
        # Select Turbine Model
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.FAST_InputFile    = self.FAST_InputFile  # FAST input file (ext=.fst)

        fastBatch.debug_level       = 2
        fastBatch.overwrite_outfiles = False

        # Turbine inputs
        iec = CaseGen_IEC()
        iec.overwrite           = False
        iec.Turbine_Class       = 'I'   # Wind class I, II, III, IV
        iec.Turbulence_Class    = 'B'   # Turbulence class 'A', 'B', or 'C'
        iec.D                   = 240.  # Rotor diameter to size the wind grid
        iec.z_hub               = 150.  # Hub height to size the wind grid
        cut_in                  = 4.    # Cut in wind speed
        cut_out                 = 25.   # Cut out wind speed
        n_ws                    = 3    # Number of wind speed bins
        TMax                    = 800.    # Length of wind grids and OpenFAST simulations, suggested 720 s
        Vrated                  = 10.59 # Rated wind speed
        Ttrans                  = max([0., TMax - 400.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
        TStart                  = 0 # Start of the recording of the channels of OpenFAST

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
        iec.dlc_inputs = {}

        if not isinstance(wind_speeds,list):
            wind_speeds = [wind_speeds]

        iec.dlc_inputs['DLC']   = [1.1]
        iec.dlc_inputs['Seeds'] = [[25]]
        iec.dlc_inputs['U']     = [wind_speeds]
        
        iec.dlc_inputs['Yaw']   = [[]]
        iec.PC_MaxRat           = 2.
        iec.uniqueSeeds         = True
        iec.uniqueWaveSeeds     = True

        iec.TStart              = Ttrans
        iec.TMax                = TMax    # wind file length
        iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
        iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS


        # Naming, file management, etc
        weis_dir            = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        iec.wind_dir        = os.path.join(weis_dir,'wind/IEA-15MW')
        iec.case_name_base  = 'level3'

        if self.n_cores > 1:
            iec.parallel_windfile_gen = True
        else:
            iec.parallel_windfile_gen = False
        iec.run_dir = os.path.join(weis_dir,'outputs','iea_semi_level3')

        # Run case generator / wind file writing
        case_inputs = {}
        case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
        case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
        case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}

        case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
        case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}

        # Stop Generator from Turning Off
        case_inputs[('ServoDyn', 'GenTiStr')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ServoDyn', 'GenTiStp')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ServoDyn', 'SpdGenOn')] = {'vals': [0.], 'group': 0}
        case_inputs[('ServoDyn', 'TimGenOn')] = {'vals': [0.], 'group': 0}
        case_inputs[('ServoDyn', 'GenModel')] = {'vals': [1], 'group': 0}

        # Set control parameters
        if omega:
            weis_dir            = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            control_param_yaml  = os.path.join(weis_dir,'examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi/IEA15MW.yaml')
            inps                = yaml.safe_load(open(control_param_yaml))
            path_params         = inps['path_params']
            turbine_params      = inps['turbine_params']
            controller_params   = inps['controller_params']

            # make default controller, turbine objects for ROSCO_toolbox
            turbine             = ROSCO_turbine.Turbine(turbine_params)
            turbine.load_from_fast( fastBatch.FAST_InputFile,fastBatch.FAST_directory, dev_branch=True)

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
            m_omega = []  # flattened (omega,zeta) pairs
            m_zeta = []  # flattened (omega,zeta) pairs
            for o in omega:
                for z in zeta:
                    controller.omega_pc = o
                    controller.zeta_pc  = z
                    controller.tune_controller(turbine)
                    pc_kp.append(controller.pc_gain_schedule.Kp.tolist())
                    pc_ki.append(controller.pc_gain_schedule.Ki.tolist())
                    m_omega.append(o)
                    m_zeta.append(z)

            # add control gains to case_list
            case_inputs = {}
            case_inputs[('DISCON_in','omega')]          = {'vals': m_omega, 'group': 2}
            case_inputs[('DISCON_in','zeta')]          = {'vals': m_zeta, 'group': 2}
            case_inputs[('DISCON_in', 'PC_GS_KP')] = {'vals': pc_kp, 'group': 2}
            case_inputs[('DISCON_in', 'PC_GS_KI')] = {'vals': pc_ki, 'group': 2}


        run_dir1            = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
        if platform.system() == 'Windows':
            path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dll')
        elif platform.system() == 'Darwin':
            path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dylib')
        else:
            path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.so')
        case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}


        channels = {}
        for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", \
            "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", \
                "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", \
                    "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3",\
                        "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust",\
                            "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", \
                                "NcIMUTAxs", "NcIMUTAys", "NcIMUTAzs", "NcIMURAxs", "NcIMURAys", "NcIMURAzs", \
                                    "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya",\
                                    "LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", \
                                        "TipRDxr", "TipRDyr", "TipRDzr"]:
            channels[var] = True
        
        
        

        case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)

        fastBatch.case_list         = case_list
        fastBatch.case_name_list    = case_name_list
        fastBatch.channels          = channels
        fastBatch.FAST_runDirectory = iec.run_dir 

        if self.n_cores == 1:
            fastBatch.run_serial()
        else:
            fastBatch.run_multi(cores=n_cores)

        return fastBatch



if __name__ == '__main__':

    # 0. Set up Model, using default input files
    mf_turb = MF_Turbine()
    mf_turb.n_cores = 8

    # 1. Run linearization procedure
    wind_speeds = np.arange(12,20,2).tolist()
    mf_turb.gen_level2_model(wind_speeds,dofs=['GenDOF','TwFADOF1','PtfmPDOF'])


    # 2. post-process linearization results

    # Load system from OpenFAST .lin files
    lin_file_dir = os.path.join(weis_dir,'outputs/iea_semi_lin')
    linTurb = lin_mod.LinearTurbineModel(lin_file_dir,reduceStates=False)

    # 3. Run turbulent level 3 case
    fast_batch = mf_turb.run_level3()

    # load wind disturbance from output file
    outfiles    = [os.path.join(mf_turb.FAST_runDirectory,case + '.outb') for case in mf_turb.case_name_list]
    fast_out    = fast_io.load_fast_out(outfiles)

    Non_OutList = fast_out[0].keys()
    Non_OutData = fast_out[0]

    for out in fast_out:
        u_h         = out['RtVAvgxh']
        tt          = out['Time']
    
    if True:
        fastRead = InputReader_OpenFAST(FAST_ver='OpenFAST', dev_branch=True)
        fastRead.FAST_InputFile = fastBatch.FAST_InputFile   # FAST input file (ext=.fst)
        fastRead.FAST_directory = fastBatch.FAST_directory  # Path to fst directory files
        fastRead.execute()

        # Load Controller from DISCON.IN
        fp = ROSCO_utilities.FileProcessing()
        f = fp.read_DISCON(fastRead.fst_vt['ServoDyn']['DLL_InFile'])
        linCont = lin_mod.LinearControlModel([],fromDISCON_IN=True,DISCON_file=f)
    else:

        # Load controller from yaml file 
        parameter_filename  = os.path.join(weis_dir,'ROSCO_toolbox/Tune_Cases/IEA15MW.yaml')
        inps = yaml.safe_load(open(parameter_filename))
        path_params         = inps['path_params']
        turbine_params      = inps['turbine_params']
        controller_params   = inps['controller_params']

        controller_params['omega_pc'] = 0.15

        # Instantiate turbine, controller, and file processing classes
        turbine         = ROSCO_turbine.Turbine(turbine_params)
        controller      = ROSCO_controller.Controller(controller_params)

        # Load turbine data from OpenFAST and rotor performance text file
        turbine.load_from_fast(fastBatch.case_name_list[0]+'.fst', \
            fastBatch.FAST_runDirectory, \
                dev_branch=True)

        # Tune controller 
        controller.tune_controller(turbine)

        controller.turbine = turbine

        linCont = lin_mod.LinearControlModel(controller)
        print('here')



    Lin_OutList, Lin_OutData, P_cl = linTurb.solve(tt,u_h,Plot=False,open_loop=False,controller=linCont)


    # comparison plot
    if True:
        comp_channels = ['RtVAvgxh','GenSpeed','TwrBsMyt','PtfmPitch']
        ax = [None] * len(comp_channels)
        plt.figure(2)

        for iPlot in range(0,len(comp_channels)):
            ax[iPlot] = plt.subplot(len(comp_channels),1,iPlot+1)
            try:
                ax[iPlot].plot(tt,Non_OutData[comp_channels[iPlot]])
            except:
                print(comp_channels[iPlot] + ' is not in OpenFAST OutList')

            try:
                ax[iPlot].plot(tt,Lin_OutData[comp_channels[iPlot]])
            except:
                print(comp_channels[iPlot] + ' is not in Linearization OutList')
            ax[iPlot].set_ylabel(comp_channels[iPlot])
            ax[iPlot].grid(True)
            if not iPlot == (len(comp_channels) - 1):
                ax[iPlot].set_xticklabels([])

        plt.show()
