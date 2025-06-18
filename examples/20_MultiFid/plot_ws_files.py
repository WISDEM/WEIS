import dill
import matplotlib.pyplot as plt

hf_results_file = 'hf_ws_file.dill'
lf_results_file = 'lf_ws_file.dill'


with open(hf_results_file,'rb') as handle:
    hf_results = dill.load(handle)

with open(lf_results_file,'rb') as handle:
    lf_results = dill.load(handle)

fig,ax = plt.subplots(1)

ax.set_xlim([1,3])
ax.set_ylim([0.1,3])
ax.grid()

dv_lf = lf_results['desvars']
dv_hf = hf_results['desvars']

for lf in dv_lf:
    ax.plot(lf[0],lf[1],'.',markersize = 10,color = 'green')

for hf in dv_hf:
    ax.plot(hf[0],hf[1],'.',markersize = 10,color = 'red')    

plt.show()