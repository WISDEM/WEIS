'''
File to construct DFSM for two simulations and compare derivative  (sanity check)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from DFSM import SimulationDetails,DFSM,reduce_samples

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'outputs'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '1p1_samples_transition' + os.sep + 'lin_0.outb'
    
        # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq',
                     'BldPitch1', 'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    # create class instance
    training_sample = SimulationDetails(simulation_name = 'training',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind)
    
    # extract values
    training_sample.extract_channels()
    
    reduce_sample_flag = True
    
    if reduce_sample_flag:
        
        nt = 50
        
        # reduce samples
        time_ed,controls_ed1,states_ed1,state_derivatives_ed1,state_derivatives2_ed = reduce_samples(nt,training_sample)
        
        # arg sort for wind speed
        ind1 = np.argsort(controls_ed1[:,0])
        
        # stack the controls and states
        inputs_ed1 = np.hstack([states_ed1,controls_ed1])
        
        # arrange them in ascending order of wind speed values
        inputs_ed1 = inputs_ed1[ind1]
        state_derivatives_ed1 = state_derivatives_ed1[ind1]
        
        
        
    # second simulation
    
    # load validation file
    sim_file = datapath + os.sep + '1p1_samples_transition'+ os.sep + 'lin_1.outb'
    
    # create class instance
    validation_sample = SimulationDetails(simulation_name = 'validation',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind)
    
    # extract values
    validation_sample.extract_channels()
    state_names = validation_sample.state_names
    control_names = validation_sample.control_names
    
    input_names = state_names+control_names
    
    reduce_sample_flag = True
    
    if reduce_sample_flag:
        
        nt = 50
        
        # reduce samples
        time_ed,controls_ed2,states_ed2,state_derivatives_ed2,state_derivatives2_ed = reduce_samples(nt,validation_sample)
        
        # arg sort for the wind indices
        ind2 = np.argsort(controls_ed2[:,0])
        
        # combine states and controls
        inputs_ed2 = np.hstack([states_ed2,controls_ed2])
        
        # arrange them in ascending order of wind speed
        inputs_ed2 = inputs_ed2[ind2]
        state_derivatives_ed2 = state_derivatives_ed2[ind2]
        
    diff = inputs_ed1 - inputs_ed2
    
    inputs_combined = np.vstack([inputs_ed1,inputs_ed2])
    
    mean_combined = np.mean(inputs_combined,axis=0)
    
    DFSM_ed1 = DFSM(inputs = inputs_ed1,outputs = state_derivatives_ed1,scale_flag = 1)
    DFSM_ed1.scale_values()
    DFSM_ed1.construct_surrogate()
    
    dx_ed1 = DFSM_ed1.evaluate_surrogate(mean_combined)
    
    print('')
    print('dx1: ' + str(dx_ed1))
    print('')
    
    
    DFSM_ed2 = DFSM(inputs = inputs_ed2,outputs = state_derivatives_ed2,scale_flag = 1)
    DFSM_ed2.scale_values()
    DFSM_ed2.construct_surrogate()
    
    dx_ed2 = DFSM_ed2.evaluate_surrogate(mean_combined)
    
    print('')
    print('dx2: ' + str(dx_ed2))
    print('')
    
    ninputs = len(mean_combined)
    
    for i in range(ninputs):
        fig,ax = plt.subplots(1)
        ax.hist(inputs_ed1[:,i],label = 'fit',alpha = 0.6)
        ax.hist(inputs_ed2[:,i],label = 'val',alpha = 0.6)
        ax.vlines(x = mean_combined[i],ymin = 0,ymax = 100,color = 'k')
        ax.set_title(input_names[i])
        ax.legend()
    
    
    
        
        
        
    
        
    
    
    
    

