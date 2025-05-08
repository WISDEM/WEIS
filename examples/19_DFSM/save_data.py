import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle

from scipy.interpolate import CubicSpline,interp1d
from scipy.io import savemat

import time as timer

# DFSM modules
from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension
from weis.dfsm.test_dfsm import test_dfsm
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.evaluate_dfsm import evaluate_dfsm
from weis.dfsm.dfsm_sample_data import sample_data
from weis.dfsm.wrapper_LTI import wrapper_LTI

from numpy.linalg import lstsq

from weis.dfsm.dfsm_rosco_simulation import run_sim_ROSCO
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# get path to this directory
this_dir = os.path.dirname(os.path.realpath(__file__))
outputs_dir = this_dir + os.sep + 'outputs'


def ModelData():
    
    model_data = {}

    model_data['reqd_states'] = ['PtfmPitch','TTDspFA','GenSpeed'];ns = len(model_data['reqd_states'])
    model_data['reqd_controls'] =  ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev'];nc = len(model_data['reqd_controls'])
    model_data['reqd_outputs'] = ['TwrBsFxt','TwrBsMyt','GenPwr','RtFldCp','RtFldCt'] 

    model_data['datapath'] = outputs_dir+os.sep +'FOWT_1p6'

    scale_args = {'state_scaling_factor': np.array([1,1,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1,1,1,1])
                  }
    
    model_data['scale_args'] = scale_args

    n_var = ns*(nc+2*ns)

    model_data['n_var'] = n_var

    model_data['w_start'] = 14.2
    model_data['buff'] = 0.01
    model_data['train_inds'] = np.arange(0,7)
    model_data['tmin'] = 0

    model_data['test_inds'] = np.array([5,11,17,23,29])
    model_data['dfsm_file_name'] = 'dfsm_1p6_14_data.mat'
     
    return model_data


if __name__ == '__main__':
    
    
    # datapath
    region = 'LPV'

    model_data = ModelData()
    datapath =  model_data['datapath'] #this_dir + os.sep + 'outputs' + os.sep + 'FOWT_1p6' #+ os.sep + 'openfast_runs/rank_0'
    
    # get the path to all .outb files in the directory
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)

    # required states
    reqd_states = model_data['reqd_states'] 
    reqd_controls = model_data['reqd_controls']
    reqd_outputs = model_data['reqd_outputs']

    ns = len(reqd_states)
    nc = len(reqd_controls)
    ny = len(reqd_outputs)
    
    
    # scaling parameters
    scale_args = model_data['scale_args']
    
    # filter parameters
    filter_args = {'state_filter_flag': [True]*ns,
                   'state_filter_type': [['filtfilt']]*ns,
                   'state_filter_tf': [[0.1]]*ns,
                   }
    
    # instantiate class
    tmin = model_data['tmin']
    t1 = timer.time()
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=tmin
                                   ,add_dx2 = True,linear_model_file = None,region = region)
    t2 = timer.time()
    # load and process data
    sim_detail.load_openfast_sim()
    
    
    # extract data
    FAST_sim = sim_detail.FAST_sim

    FAST_sim_array = np.array(FAST_sim)

    state_names = FAST_sim_array[0]['state_names']

    # get index of nacelle rotational and translational accelerations
    try:
        ncimu_ind = reqd_outputs.index('NcIMURAys')
    except:
        ncimu_ind = None

    try:
        yawbr_ind = reqd_outputs.index('YawBrTAxp')
    except:
        yawbr_ind = None

    try:
        dptfmpitch_ind = state_names.index('dPtfmPitch')
    except:
        dptfmpitch_ind = None

    n_var = model_data['n_var']
   
    w_array = []

    for sim_det in FAST_sim_array:
        w_array.append(sim_det['w_mean'])

    w_array = np.round(np.array(w_array),1)
    W = np.unique(w_array)
    
    nW = len(W)
    nseeds = int(len(w_array)/nW)

    
    FAST_sim_array = np.array(FAST_sim)
    FAST_sim = np.reshape(FAST_sim_array,[nseeds,nW],order = 'F')
    # sort_ind = np.argsort(w_array)
    # FAST_sim = FAST_sim[:,sort_ind]

    train_inds = model_data['train_inds']
    

    n_samples = 1

    w_start = model_data['w_start']
    
    ind_start = np.where(W == w_start)[0][0]
    ind_forward_pass = np.where(W >= w_start)[0]
    ind_backward_pass = np.flip(np.where(W < w_start))[0]



    _,_,_,inputs,state_dx,outputs = sample_data(FAST_sim[:,ind_start],'KM',n_samples = 1)

    controls = inputs[:,:nc]
    states = inputs[:,nc:]
    Time = FAST_sim[0,0]['time']

    data_dict = {'reqd_states':reqd_states,'reqd_controls':reqd_controls,'reqd_outputs':reqd_outputs,'Time':Time,'States':states,'Controls':controls,'Outputs':outputs}

    
    dfsm_pkl =  model_data['dfsm_file_name']
    
    savemat(dfsm_pkl,data_dict)





        