import os
import numpy as np
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.dfsm_sample_data import sample_data
import matplotlib.pyplot as plt
import time as timer

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    
    # datapath
    datapath = mydir + os.sep + 'outputs' + os.sep + 'MHK_TR'
    
    # get the entire path
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['PtfmPitch','GenSpeed','YawBrTAxp']
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    reqd_outputs = [] #'TwrBsFxt','TwrBsMyt'
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,100,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False],
                   'states_filter_tf': [1],
                   'controls_filter_flag': [False,False,False],
                   'outputs_filter_flag': [False,False]
                   }
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,add_dx2 = True)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim
    
    sampling = ['separate','together']
    
    
    sampling_type = 'KM'
    
    n_samples = [10,20,50]
    
    sampling_time =np.zeros((len(n_samples),len(sampling))) 
    
    fig,ax = plt.subplots(1,1)
    
    
    
    for is_,samp in enumerate(sampling):
        for n,n_samp in enumerate(n_samples):
            
            t1 = timer.time()
            inputs_sampled,dx_sampled,outputs_sampled,model_inputs,state_derivatives,outputs = sample_data(FAST_sim[:4],sampling_type,n_samp,samp)
            t2 = timer.time()
            
            sampling_time[n,is_] = t2-t1
            
            ax.plot(state_derivatives[:,0],state_derivatives[:,1],'.',color = 'b',markersize = 0.5,label = 'data')
            ax.plot(dx_sampled[:,0],dx_sampled[:,1],'.',color = 'r',markersize = 3,label = 'samples = ' +str(n_samp))
    ax.legend()
    ax.set_title('separate')
    ax.set_xlabel('dx1')
    ax.set_ylabel('dx2')
    
    fig,ax = plt.subplots(1)
    ax.plot(n_samples,sampling_time[:,0],label = 'separate')
    ax.plot(n_samples,sampling_time[:,1],label = 'together')
    ax.set_xlabel('n_samples')
    ax.set_ylabel('sampling time')
    ax.legend()
   
    
    # fig,ax = plt.subplots(1,1)
    
    # ax.plot(state_derivatives[:,1],state_derivatives[:,3],'.',color = 'b',markersize = 0.5,label = 'data')
    # ax.plot(dx_sampled[:,1],dx_sampled[:,3],'.',color = 'r',markersize = 3,label = 'samples')
    # ax.legend()
    # ax.set_xlabel('dx2')
    # ax.set_ylabel('dx4')