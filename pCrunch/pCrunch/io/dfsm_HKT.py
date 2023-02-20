import os
import numpy as np
from pCrunch.io import load_FAST_out
from scipy.interpolate import PchipInterpolator,Rbf,CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.integrate import solve_ivp
from numpy.linalg import lstsq,qr,inv,norm
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk
from sklearn.gaussian_process.kernels import ExpSineSquared
import time as time1
from sklearn.cluster import KMeans


if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'OldDetailDesign'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '26kW_UNH_Mega.out'
    
        # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq',
                     'BldPitch1', 'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    outputs = load_FAST_out(sim_file)[0]

