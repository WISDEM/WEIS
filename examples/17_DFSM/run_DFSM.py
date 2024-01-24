import os
import numpy as np
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension,calculate_time
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
import matplotlib.pyplot as plt
import warnings
from mat4py import loadmat

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    
    # datapath
    region = 'R'
    datapath = mydir + os.sep + 'outputs' + os.sep + 'simple_sim_BR' #'MHK_'+region+'_10' #+ os.sep + 'openfast_runs/rank_0'
 
    # get the entire path
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['GenSpeed']
    
    state_props = {'units' : ['[deg]','[rpm]','[m/s2]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = [] #['TwrBsFxt', 'GenPwr'] #['YawBrTAxp', 'NcIMURAys', 'GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1]),
                  'control_scaling_factor': np.array([1,1,1]),
                  'output_scaling_factor': np.array([1,1])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False,False,False],
                   'states_filter_type': [[],['filtfilt'],[]],
                   'states_filter_tf': [[],[0.5],[]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [0,0,0],
                   'outputs_filter_flag': []
                   }
    
    file_name = None #mydir + os.sep + 'linear-models.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = file_name,region = region)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim
    
    # plot data
    plot_inputs(sim_detail,4,'separate',save_flag = True)
    
    # split of training-testing data
    n_samples = [200]
    inputs_sampled = []
    dx_sampled = []
    
    for isample in n_samples:
        
        # instantiate DFSM model and construct surrogate
        dfsm_model = DFSM(sim_detail,n_samples = isample,L_type = 'LTI',N_type = None, train_split = 0.4)
        dfsm_model.construct_surrogate()
        
        inputs_sampled.append(dfsm_model.inputs_sampled)
        dx_sampled.append(dfsm_model.dx_sampled)
                    
        # extract test data
        test_data = dfsm_model.test_data
        
        # index of test data
        test_ind = [0]
        
        # flags related to testing
        simulation_flag = True 
        outputs_flag = (len(reqd_outputs) > 0)
        plot_flag = True
        
        # test dfsm
        dfsm,U_list,X_list,dx_list,Y_list = test_dfsm(dfsm_model,test_data,test_ind,simulation_flag,plot_flag)
        
        #Add state properties to list
        X_list_ = [dicti.update(state_props) for dicti in X_list]
        Y_list_ = [dicti.update(output_props) for dicti in Y_list]
        
        # plot dfsm
        plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag,save_flag = True)
        
        #calculate the time
        dfsm_time = calculate_time(dfsm_model)

        print(X_list[0].keys())

        plt.show()
            
            
                
       

    
    
    
    
    
    

