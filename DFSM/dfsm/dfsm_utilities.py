"""
Script containing utility functions required in the DFSM modules

"""
import numpy as np
from numpy.linalg import norm
import fnmatch


def valid_extension(fp):
    return any([fnmatch.fnmatch(fp,ext) for ext in ['*.outb']])

def calculate_MSE(x1,x2):
    
    '''
    Function to calculate the mean square error between two signals
    '''
    
    # number of samples
    nt = len(x1)
    
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

