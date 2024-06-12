import matplotlib.pyplot as plt 
import numpy as np
import os, platform
import sys

# ROSCO toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import control_interface as ROSCO_ci
from ROSCO_toolbox.utilities import write_DISCON
from ROSCO_toolbox.inputs.validation import load_rosco_yaml
from scipy.interpolate import CubicSpline,interp1d
from scipy.interpolate import CubicSpline
from ROSCO_toolbox.ofTools.util import spectral
import time as timer

# DFSM modules
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension,calculate_time, calculate_MSE
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.evaluate_dfsm import evaluate_dfsm
from scipy.integrate import solve_ivp
from mat4py import loadmat

from dfsm.dfsm_rosco_simulation import run_sim_ROSCO

if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    # datapath
    region = 'LPV'
    datapath =  '/home/athulsun/DFSM/data' + os.sep + 'FOWT_1p6_2' #this_dir + os.sep + 'outputs' + os.sep + 'FOWT_1p6' #+ os.sep + 'openfast_runs/rank_0'
    testpath = '/home/athulsun/DFSM/data' + os.sep + '1p6_test_2' # this_dir + os.sep + 'outputs' + os.sep + '1p6_test'
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    outfiles_test = [os.path.join(testpath,f) for f in os.listdir(testpath) if valid_extension(f)]
    outfiles_test = sorted(outfiles_test)
    
    # required states
    reqd_states = ['PtfmPitch','TTDspFA','GenSpeed']
    
    state_props = {'units' : ['[deg]','[m/s2]','[rpm]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['TwrBsFxt','TwrBsMyt','YawBrTAxp','NcIMURAys','GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,1,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [True,True,True],
                   'states_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                   'states_filter_tf': [[0.1],[0.1],[0.1]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [0,0,0],
                   'outputs_filter_flag': []
                   }
    
    # name of mat file that has the linearized models
    mat_file_name = this_dir + os.sep + 'FOWT_1p6_LPV2.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    sim_detail_test = SimulationDetails(outfiles_test, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = mat_file_name,region = region)
    
    
    # load and process data
    sim_detail.load_openfast_sim()
    sim_detail_test.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail_test.FAST_sim
    
    # plot data
    #plot_inputs(sim_detail,4,'separate')
    
    # split of training-testing data
    n_samples = 1
    test_inds = [1,3,5,7,9,11,13,15,17,19,21]
    test_inds = [0,2,4,6,8,10,12,14,16,18,20]
    
    # construct surrogate model
    dfsm_model = DFSM(sim_detail,n_samples = n_samples,L_type = 'LPV',N_type = None, train_split = 0.5)
    dfsm_model.construct_surrogate()
    dfsm_model.simulation_time = []

    GT_OF_all = []
    BP_OF_all = []
    BP_DFSM_all = []
    GT_DFSM_all = []

    # load results from matlab
    results_mat = loadmat('simulation_results_LPV2_13.mat')

    # test dfsm
    # flags related to testing
    simulation_flag = True 
    outputs_flag = (len(reqd_outputs) > 0)
    plot_flag = True
    dfsm,U_list,X_list,dx_list,Y_list = test_dfsm(dfsm_model,FAST_sim,test_inds,simulation_flag,plot_flag)
    
    
    for idx,ind in enumerate(test_inds):

        time = results_mat['time']
        time_py = X_list[idx]['time']

        x_DFSM_matlab = results_mat['state_cell_DFSM']
        x_DFSM_matlab = np.squeeze(np.array(x_DFSM_matlab[idx]))

        x_DFSM_python = X_list[idx]['DFSM']
        x_OF = X_list[idx]['OpenFAST']

        y_DFSM_matlab = results_mat['outputs_cell_DFSM']
        y_DFSM_matlab = np.squeeze(np.array(y_DFSM_matlab[idx]))
        y_DFSM_python = Y_list[idx]['DFSM']
        y_OF = Y_list[idx]['OpenFAST']
    
        ns = len(reqd_states)
        ny = len(reqd_outputs)


        fig1,ax1 = plt.subplots(ns,2)

        for i in range(ns):
            ax1[i,0].plot(time,x_DFSM_matlab[:,i],label='matlab')
            ax1[i,0].plot(time_py,x_DFSM_python[:,i],label = 'python')
            ax1[i,0].plot(time_py,x_OF[:,i],label = 'OpenFAST')

            ax1[i,0].set_title(reqd_states[i])
        

        for i in range(ns):
            ax1[i,1].plot(time,y_DFSM_matlab[:,i+2],label='matlab')
            ax1[i,1].plot(time_py,y_DFSM_python[:,i+2],label = 'python')
            ax1[i,1].plot(time_py,y_OF[:,i+2],label = 'OpenFAST')
            ax1[i,1].set_title(reqd_outputs[i+2])

        ax1[i,0].legend(ncol = 3)
    plt.show()




        