import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

dfsm_file = 'dfsm_iea15_test.pkl'

with open(dfsm_file,'rb') as handle:
    dfsm = pickle.load(handle)

model_construct_time = dfsm.model_construct_time
W = dfsm.W

dfsm.buff = np.array(dfsm.buff)

# plot properties
markersize = 10
linewidth = 1.5
fontsize_legend = 16
fontsize_axlabel = 18
fontsize_tick = 15

print(dfsm.buff)
print(dfsm.w_start)
print(dfsm.train_inds)

fig,ax = plt.subplots(1)

ax.plot(W,model_construct_time,'k.-',markersize = markersize)
ax.set_xlabel('Wind Speed [m/s]',fontsize = fontsize_axlabel)
ax.set_ylabel('Model Construction Time [s]',fontsize = fontsize_axlabel)
ax.tick_params(labelsize=fontsize_tick)
ax.grid()

plt.show()

