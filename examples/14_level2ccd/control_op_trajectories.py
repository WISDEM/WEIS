import numpy as np
from scipy.io import loadmat
from pCrunch.io import load_FAST_out
from scipy.interpolate import PchipInterpolator
import os
from pCrunch.io import OpenFASTOutput
import matplotlib.pyplot as plt

op_point_mat = 'control_op_points.mat'
u_op = loadmat(op_point_mat)
uw = u_op['uw']

u_h = uw[0,:]


# get path to current directory
mydir = os.path.dirname(os.path.realpath(__file__))


# .outb file and pickle file
sim_file = mydir + os.sep + 'outputs/sim_16' + os.sep + 'lin_0.outb'

reqd_channels = ['Time', 'RtVAvgxh']

# load simulations
outputs = load_FAST_out(sim_file)[0]

Time = outputs['Time']
Time = Time -np.min(Time)
Wind = outputs['RtVAvgxh']

# control operating points interpolating function
Uo_pp = PchipInterpolator(u_h,uw,axis = 1,extrapolate = True)
Uo_fun = lambda w: Uo_pp(w)

W_pp = PchipInterpolator(Time,Wind)
W_fun = lambda t: W_pp(t)

UoLPV = Uo_fun(W_fun(Time))

UoLPV = UoLPV.T

outdata = {'Time':Time,
           'Wind1VelX':UoLPV[:,0],
           'GenTq':UoLPV[:,1],
           'BldPitch1':np.rad2deg(UoLPV[:,2])}

casename = 'oloc_0'

magnitude_channels = {'RootMc1': ['RootMxc1', 'RootMyc1', 'RootMzc1'], 
                      'RootMc2': ['RootMxc2', 'RootMyc2', 'RootMzc2'],
                      'RootMc3': ['RootMxc3', 'RootMyc3', 'RootMzc3']}

output = OpenFASTOutput.from_dict(outdata, casename,magnitude_channels=magnitude_channels)

output = [output]

plt.plot(Time,(UoLPV[:,0]))

for i_ts, timeseries in enumerate(output):
        timeseries.df.to_pickle(os.path.join(mydir,'oloc' + '_' + str(i_ts) + '.p'))