''' This example runs a linear turbine simulation
0. Run gen_linear_model() in WEIS/wies/aeroelasticse/LinearFAST.py  (separately)
1. Set up linear turbine model by running mbc transformation
2. Load controller from ROSCO_toolbox and set up linear control model
3. Use wind disturbance to determine operating point for turbine, pitch controller
4. Run linear simulation

'''

import numpy as np
import pandas as pd
import os
import weis.control.LinearModel as lin_mod
from weis.aeroelasticse.LinearFAST import LinearFAST
from ROSCO_toolbox import utilities as ROSCO_utilities
from ROSCO_toolbox.ofTools.fast_io import output_processing
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
# from pCrunch.Analysis import Loads_Analysis
import yaml
import matplotlib.pyplot as plt

PLOT = False


if __name__ == '__main__':

    lin_fast = LinearFAST(FAST_ver='OpenFAST', dev_branch=True);

    # fast info
    lin_fast.weis_dir                 = os.path.dirname( os.path.dirname ( os.path.dirname( os.path.abspath(__file__) ) ) ) + os.sep
    
    lin_fast.FAST_InputFile           = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    lin_fast.FAST_directory           = os.path.join(lin_fast.weis_dir, 'examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
    lin_fast.FAST_runDirectory        = os.path.join(lin_fast.weis_dir,'outputs','iea_semi_lin')
    lin_fast.FAST_exe                 = '/mnt/c/Users/jjasa/git/WEIS/local/bin/openfast'
    lin_fast.debug_level              = 2
    lin_fast.dev_branch               = True
    lin_fast.write_yaml               = True
    
    lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
    lin_fast.wind_speeds                 = [16.]     # np.arange(5,25,4,dtype=float).tolist()
    lin_fast.DOFs                       = ['GenDOF']  # enable with 
    lin_fast.TMax                       = 1600   # should be 1000-2000 sec or more with hydrodynamic states
    lin_fast.NLinTimes                  = 4
    lin_fast.TrimTol                    = 2e-2

    # lin_fast.FAST_exe                   = '/Users/dzalkind/Tools/openfast/install/bin/openfast'

    # simulation setup
    lin_fast.cores            = 1

    # overwrite steady & linearizations
    lin_fast.overwrite        = True

    lin_fast.gen_linear_model()

    # # 1. Set up linear turbine model by running mbc transformation
    # if True:  
    #     # Load system from OpenFAST .lin files
    #     lin_file_dir = os.path.join(weis_dir,'outputs/iea_semi_lin')
    #     linTurb = lin_mod.LinearTurbineModel(lin_file_dir,reduceStates=False)
    
    # else:  
    #     # Load system from .mat file
    #     linTurb = lin_mod.LinearTurbineModel('/Users/dzalkind/Tools/matlab-toolbox/Simulations/SaveData/LinearModels/PitTwr.mat', \
    #         fromMat=True)


    # # 1.5 Set up wind disturbance
    # if True:   
    #     # step wind input
    #     tt = np.arange(0,200,1/80)
    #     u_h = 16 * np.ones((tt.shape))
    #     u_h[tt > 100] = 17

    # else:
    #     # load wind disturbance from output file
    #     fast_io     = ROSCO_utilities.FAST_IO()
    #     fast_out    = output_processing.load_fast_out('/Users/dzalkind/Tools/matlab-toolbox/Simulations/SaveData/072720_183300.out')
    #     u_h         = fast_out[0]['RtVAvgxh']
    #     tt          = fast_out[0]['Time']

    
    # # 2. Load controller from ROSCO_toolbox and set up linear control model
    # if True:
    #     # Load controller from yaml file 
    #     parameter_filename  = os.path.join(weis_dir,'ROSCO_toolbox/Tune_Cases/IEA15MW.yaml')
    #     inps = yaml.safe_load(open(parameter_filename))
    #     path_params         = inps['path_params']
    #     turbine_params      = inps['turbine_params']
    #     controller_params   = inps['controller_params']

    #     controller_params['omega_pc'] = 0.2

    #     # Instantiate turbine, controller, and file processing classes
    #     turbine         = ROSCO_turbine.Turbine(turbine_params)
    #     controller      = ROSCO_controller.Controller(controller_params)

    #     # Load turbine data from OpenFAST and rotor performance text file
    #     turbine.load_from_fast(lin_turbine.FAST_InputFile,lin_turbine.FAST_directory,dev_branch=True)

    #     # Tune controller 
    #     controller.tune_controller(turbine)

    #     controller.turbine = turbine

    #     linCont = lin_mod.LinearControlModel(controller)
        
    # else:
    #     # Load Controller from DISCON.IN
    #     fp = ROSCO_utilities.FileProcessing()
    #     f = fp.read_DISCON('/Users/dzalkind/Tools/SaveData/Float_Test/UM_DLC0_100_DISCON.IN')

    #     linCont = lin_mod.LinearControlModel([],fromDISCON_IN=True,DISCON_file=f)


    # # 3. Use wind disturbance to determine operating point for turbine, pitch controller
    # # 4. Run linear simulation
    # Lin_OutList, Lin_OutData, P_cl = linTurb.solve(tt,u_h,Plot=False,open_loop=False,controller=linCont)
    # # linear plot
    # if True:
    #     channels = ['RtVAvgxh','GenSpeed','BldPitch','TwrBsMyt','PtfmPitch']
    #     ax = [None] * len(channels)
    #     plt.figure(2)

    #     for iPlot in range(0,len(channels)):
    #         ax[iPlot] = plt.subplot(len(channels),1,iPlot+1)
    #         try:
    #             ax[iPlot].plot(tt,Lin_OutData[channels[iPlot]])
    #         except:
    #             print(channels[iPlot] + ' is not in Linearization OutList')
    #         ax[iPlot].set_ylabel(channels[iPlot])
    #         ax[iPlot].grid(True)
    #         if not iPlot == (len(channels) - 1):
    #             ax[iPlot].set_xticklabels([])

    #     if PLOT:
    #         plt.show()


    # # sweep controller.pc_omega and run linearization
    # fast_data_lin = []

    # if True:
    #     ww = np.linspace(.05,.45,4)

    #     for om in ww:
    #         # Tune controller 
    #         controller.omega_pc = om
    #         controller.tune_controller(turbine)

    #         controller.turbine = turbine

    #         # update linear controller
    #         linCont = lin_mod.LinearControlModel(controller)

    #         # solve 
    #         Lin_OutList, Lin_OutData, P_cl = linTurb.solve(tt,u_h,Plot=False,open_loop=False,controller=linCont)

    #         # convert into pCrunch form
    #         fd_lin  = {}
    #         fd_lin['TwrBsMyt'] = Lin_OutData['TwrBsMyt']
    #         fd_lin['meta']    = {}
    #         fd_lin['meta']['name'] = 'omega: ' + str(om)
    #         fast_data_lin.append(fd_lin)

    #         comp_channels = ['RtVAvgxh','GenSpeed','TwrBsMyt','PtfmPitch']
    #         ax = [None] * len(comp_channels)
    #         plt.figure(3)

    #         for iPlot in range(0,len(comp_channels)):
    #             ax[iPlot] = plt.subplot(len(comp_channels),1,iPlot+1,label=comp_channels[iPlot])

    #             try:
    #                 ax[iPlot].plot(tt,Lin_OutData[comp_channels[iPlot]])
    #             except:
    #                 print(comp_channels[iPlot] + ' is not in Linearization OutList')
    #             ax[iPlot].set_ylabel(comp_channels[iPlot])
    #             ax[iPlot].grid(True)
    #             if not iPlot == (len(comp_channels) - 1):
    #                 ax[iPlot].set_xticklabels([])

    #     if PLOT:
    #         plt.show()
        

    # # Try some post processing using pCrunch
    # # chan_info = [('TwrBsMyt',4)]

    # # la = Loads_Analysis()
    # # fDEL = la.get_DEL(fast_data_lin,chan_info)
    # # print('here')




