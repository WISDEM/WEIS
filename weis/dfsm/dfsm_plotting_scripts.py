import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pCrunch.io import load_FAST_out
from rosco.toolbox.ofTools.fast_io import output_processing
from rosco.toolbox.ofTools.util import spectral


def plot_signal(signal_dict,save_flag,save_path):
    
    time = signal_dict['time']
    signals_act = signal_dict['OpenFAST']
    signals_dfsm = signal_dict['DFSM']
    n_signals = signal_dict['n']
    signal_names = signal_dict['names']
    units = signal_dict['units']
    
    if 'key_freq_name' in signal_dict:
        key_freq_name = signal_dict['key_freq_name']
        key_freq_val = signal_dict['key_freq_val']
    
    dx_flag = [not(name[0] == 'd') for name in signal_names]
    
    t0 = time[0];tf = time[-1]
    
    wd = os. getcwd()
        
    plot_path = wd + os.sep + save_path

    
    for idx,qty in enumerate(signal_names):
        
        if not(qty[0] == 'd'):
            
             # plot controls
             fig,ax = plt.subplots(2,1)
             if qty == 'RtVAvgxh' or qty == 'Wind1VelX':
                 qty = 'Current Speed'
             fig.suptitle(qty + ' '+units[idx])
             fig.subplots_adjust(hspace = 0.4)
             
             ax[0].set_xlabel('Time [s]')
             ax[0].plot(time,signals_act[:,idx],label = 'OpenFAST')
             ax[0].plot(time,signals_dfsm[:,idx],label = 'DFSM')
             ax[0].legend(ncol = 2,bbox_to_anchor = (0.7,1.26))
             ax[0].set_xlim([t0,tf])
             
             xf,FFT_act,_ = spectral.fft_wrap(time,signals_act[:,idx],averaging = 'Welch',averaging_window= 'hamming')
             xf,FFT_dfsm,_ = spectral.fft_wrap(time,signals_dfsm[:,idx],averaging = 'Welch',averaging_window= 'hamming')
             
             col_list = ['k','tab:red']
             
             if 'key_freq_name' in signal_dict:
                 for i,val in enumerate(key_freq_val[idx]):
                     ax[1].axvline(val,color = col_list[i],label = key_freq_name[idx][i],linestyle = ':')
                     ax[1].legend()
                     
             ax[1].loglog(xf,np.sqrt(FFT_act))
             ax[1].loglog(xf,np.sqrt(FFT_dfsm))
             
             ax[1].set_xlabel('Freq [Hz]')
             ax[1].set_xlim([np.min(xf),np.max(xf)])
             
             if save_flag:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                
                fig.savefig(plot_path +os.sep+ qty + '_comp.svg')
             
          

def plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag,control_flag = False,save_flag = False,save_path = 'plots'):
    
    n_results = len(U_list)
    
    for ix in range(n_results):
        
        # plot controls
        if control_flag:
            u_dict = U_list[ix]
            plot_signal(u_dict,save_flag,save_path)
        
                
        if simulation_flag:
            x_dict = X_list[ix]
            plot_signal(x_dict,save_flag,save_path)
        
        if outputs_flag:
            y_dict = Y_list[ix]
            plot_signal(y_dict,save_flag,save_path)
            
            
        

def plot_inputs(SimulationDetails,index,plot_type,save_flag = False,save_path = 'plots'):
    
    # extract 
    sim_details = SimulationDetails.FAST_sim[index]
    wd = os. getcwd()
    
    time = sim_details['time']
    controls = sim_details['controls']
    states = sim_details['states']
    outputs = sim_details['outputs']
    
    control_names = sim_details['control_names']
    
    for iu,qty in enumerate(control_names):
        if (qty == 'RtVAvgxh') or (qty == 'Wind1VelX'):
            control_names[iu] = 'Current Speed'
    
    state_names = sim_details['state_names']
    output_names = sim_details['output_names']
    
    n_controls = sim_details['n_controls']
    n_outputs = sim_details['n_outputs']
    n_states = sim_details['n_states']
    
    
    t0 = time[0]; tf = time[-1]
    print(state_names)
    state_flag = [not(name[0] == 'd') for name in state_names]
    n_states_ = sum(state_flag)
    states_ = states[:,state_flag]
    state_names_ = []
    
    for idx,flag in enumerate(state_flag):
        if flag:
            state_names_.append(state_names[idx])
    
    # depending on plot type, plot the time series quantities
    if plot_type == 'vertical':
         
        if not(len(outputs) > 0):
            
            # combine all signals into a single array
            quantities = np.hstack([controls,states_])
            
            # get the names
            quantity_names = control_names + state_names_
     
            
        else:
            
            # combine all signals into a single array
            quantities = np.hstack([controls,states_,outputs])
            
            # names of the quantities
            quantity_names = control_names + state_names_ + output_names
            
        n_qty = len(quantity_names)
        
        # intialize plot
        fig,ax = plt.subplots(n_qty,1)
        
        ax[-1].set_xlabel('Time [s]')
        
        fig.subplots_adjust(hspace = 1)
        
        for idx,qty in enumerate(quantity_names):
            
            if not(qty[0] == 'd'):
                ax[idx].plot(time,quantities[:,idx])
                ax[idx].set_title(qty)
                ax[idx].set_xlim([t0,tf])
                
            if not(idx == n_qty-1):
                ax[idx].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        
       
        
        plot_path = wd + os.sep + save_path
        
        if save_flag:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ 'inputs.svg')
            
    elif plot_type == 'separate':
        
        # plot controls
        fig,axc = plt.subplots(n_controls,1)
        if n_controls == 1:
            axc = [axc]
        axc[-1].set_xlabel('Time [s]')
        fig.subplots_adjust(hspace = 0.65)
        
        for idx,qty in enumerate(control_names):
            
            axc[idx].plot(time,controls[:,idx])
            axc[idx].set_title(qty)
            axc[idx].set_xlim([t0,tf])
            
        plot_path = wd + os.sep + save_path
        
        if save_flag:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ 'inputs.svg')
            
        # plot states    
        fig,axs = plt.subplots(n_states,1)
        if n_states == 1:
            axs = [axs]
        fig.subplots_adjust(hspace = 1)
        axc[-1].set_xlabel('Time [s]')
        
        for idx,qty in enumerate(state_names):
            
            axs[idx].plot(time,states[:,idx])
            axs[idx].set_title(qty)
            axs[idx].set_xlim([t0,tf])
        
        # plot outputs
        if n_outputs > 0:
            
            fig,axo = plt.subplots(n_outputs,1)
            axc[-1].set_xlabel('Time [s]')
            fig.subplots_adjust(hspace = 0.65)
        
            for idx,qty in enumerate(output_names):
                
                axo[idx].plot(time,outputs[:,idx])
                axo[idx].set_title(qty)
                axo[idx].set_xlim([t0,tf])
                
        
            
        
        
        
        
        
            
            
                
        
    
    