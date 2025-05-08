"""
Script containing utility functions required in the DFSM modules

"""
import numpy as np
from numpy.linalg import norm
import fnmatch


def valid_extension(fp):
    return any([fnmatch.fnmatch(fp,ext) for ext in ['*.outb']])

def valid_extension_DISCON(fp):
    return any([fnmatch.fnmatch(fp,ext) for ext in ['*.IN']])

def valid_extension_pickle(fp):
    return any([fnmatch.fnmatch(fp,ext) for ext in ['*.p']])


def calculate_MSE(x1,x2,scaled = False):
    
    '''
    Function to calculate the mean square error between two signals
    '''
    
    # number of samples
    nt = len(x1)

    if scaled:
        scaler = np.max(x1)
        x1 = x1/scaler 
        x2 = x2/scaler
    
    # MSE
    MSE = np.sum((x1-x2)**2)/nt
    
    return MSE

def calculate_time(DFSM):
    
    sampling_time = DFSM.sampling_time/60
    construction_time = (DFSM.linear_construct_time + DFSM.nonlin_construct_time)/60
    simulation_time = DFSM.simulation_time/60
    
    return {'sampling_time':sampling_time,'construction_time':construction_time,'simulation_time':simulation_time}

def calculate_SNE(x1,x2):
    
    '''
    Function to calculate the sum of normed errors (SNE) as defined in https://doi.org/10.1115/1.4037407
    '''
    
    # number of samples
    nt = len(x1)
    
    # initialize
    SNE = np.zeros((nt,))
    
    #SNE
    for i in range(nt):
        SNE[i] = norm(x1[i,:]-x2[i,:])
    
    # sum
    SNE = np.sum(SNE)
    
    return SNE

def compile_dfsm_results(time,states_dfsm,controls_dfsm,outputs_dfsm,state_names,control_names,output_names, GB_ratio,tmin = 0):

    # initialize
    OutData = {}

    time_ind = (time >= tmin)

    # add time
    OutData['Time'] = time[time_ind]

    nt = np.sum(time_ind)
    gs_ind = state_names.index('GenSpeed')

    # add states
    for i,name_ in enumerate(state_names):

        if name_[0] == 'd':
            pass 
        else:
            OutData[name_] = states_dfsm[time_ind,i]

    # add controls
    for i,name_ in enumerate(control_names):

        OutData[name_] = controls_dfsm[time_ind,i]

    # add controls
    if len(output_names) >= 1:
        for i,name_ in enumerate(output_names):

            OutData[name_] = outputs_dfsm[time_ind,i]

    if not('PtfmPitch' in state_names):
        OutData['PtfmPitch'] = np.zeros((nt,))

    if not('PtfmSurge' in state_names):
        OutData['PtfmSurge'] = np.zeros((nt,))

    if not('PtfmSway' in state_names):
        OutData['PtfmSway'] = np.zeros((nt,))
        OutData['PtfmOffset'] = np.zeros((nt,))


    if not('RotSpeed' in output_names):
        OutData['RotSpeed'] = states_dfsm[time_ind,gs_ind]/GB_ratio
    
    for i_blade in range(2):
        OutData[f'BldPitch{i_blade+2}'] = OutData['BldPitch1']

    for i_blade in range(3):
        OutData[f'dBldPitch{i_blade+1}'] = np.zeros((nt,))

    for chan in ['NcIMUTAxs', 'NcIMUTAys', 'NcIMUTAzs','NcIMUTA']:
        if chan not in output_names:
            OutData[chan] = np.zeros((nt,))


    return OutData

    

def extrapolate_controls(t_out,u, t):

    # extract time
    t1 = t[0]

    # scale time
    t -= t1 
    t_out = t_out - t1

    if len(t) == 2:
        # evaluate slope
        u1 = u[0]; u2 = u[1]
        b0 = (u2 - u1)/t[1]

        # extrapolate
        u_out = u1 + b0*t_out

    elif len(t) == 3:
         
        u1 = u[0];u2 = u[1]; u3 = u[2]

        b0 = (t[2]**2*(u1 - u2) + t[1]**2*(-u1 + u3))/(t[1]*t[2]*(t[1] - t[2]))
        c0 = ( (t[1]-t[2])*u1+ t[2]*u2 - t[1]*u3 ) / (t[1]*t[2]*(t[1] - t[2]))

        u_out = u1 + b0*t_out + c0*t_out**2
         

    return u_out


def reorganize_data(FAST_sim):

    n_sim = len(FAST_sim)

    wind_mean = np.zeros((n_sim,))

    for i in range(n_sim):
        wind_mean[i] = FAST_sim[i]['w_mean']
    
    w_unique = np.unique(wind_mean)

    n_unique = len(w_unique)

    FAST_sim = np.array(FAST_sim,dtype = object)

    FAST_sim = np.reshape(FAST_sim,[int(n_sim/n_unique),n_unique],order = 'F')

    return FAST_sim,w_unique,n_unique
