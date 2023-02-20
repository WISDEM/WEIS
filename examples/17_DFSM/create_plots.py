import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_inputs(SimulationDetails,save_flag):
    
    '''
    Function to plot the inputs given a simulation
    '''
    
    
    #subplot_ind = [(5,1,4),(5,1,5),(5,1,1),(5,1,2),(5,1,3)]
    
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.6,wspace = 0.2)
    fig.suptitle(SimulationDetails.simulation_name + '- Inputs')
    
    controls,states,state_derivatives,time = SimulationDetails.provide_inputs_outputs()
    t0 = time[0];tf = time[-1]
    
    control_names = SimulationDetails.control_names
    state_names = SimulationDetails.state_names
    dx_names = SimulationDetails.state_dx_names
    
    input_names = state_names+control_names
    inputs = np.hstack([states,controls])
    
    if len(input_names) == 5:
        subplot_ind = [(3,2,2),(3,2,4),(3,2,1),(3,2,3),(3,2,5)]
        
    elif len(input_names) == 3:
        subplot_ind = [(3,1,2),(3,1,3),(3,1,1)]
    
    for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time,inputs[:,idx],label = 'OF')
        ax_idx.set_xlim([t0,tf])
        ax_idx.set_title(input_names[idx])
        
        if ind[2] == 4 or ind[2] == 5 or ind[2] == 3:
            ax_idx.set_xlabel('Time [s]')
    
    #ax_idx.set_xlabel('Time [s]')
    if save_flag:
        fig.savefig(SimulationDetails.simulation_name+'-inputs.svg',format = 'svg',dpi = 1200)
    
    #---------------------------------------------------------------------------------------
    
    if len(state_names) == 2:
        subplot_ind = [(2,1,1),(2,1,2)]
    else:
        subplot_ind = [(2,2,1),(2,2,2),(2,2,3),(2,2,4)]
             
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.5,wspace = 0.4)
    fig.suptitle(SimulationDetails.simulation_name+ '- Outputs')
    #fig.subplots_adjust(hspace = 0.45)
   
    for idx,ind in enumerate(subplot_ind):
       ax_idx = plt.subplot(ind[0],ind[1],ind[2])
       ax_idx.plot(time,state_derivatives[:,idx],label = 'OF')
       ax_idx.set_xlim([t0,tf])
       ax_idx.set_title(dx_names[idx])
       
    ax_idx.set_xlabel('Time [s]')
    
    if save_flag:
        fig.savefig(SimulationDetails.simulation_name+'-outputs.svg',format = 'svg',dpi = 1200)
    
    
def plot_simulation_results(SimulationDetails,T,x_of,x_lm,x_dfsm,U,state_names,save_flag):

    t0 = T[0];tf = T[-1]

    sind = 0
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    fig.suptitle(SimulationDetails.simulation_name + ' - '+state_names[sind])
    
    ax1.plot(T,x_of[:,sind],label = 'OF')
    ax1.plot(T,x_lm[:,sind],color ='tab:orange' ,label = 'LM')
    ax1.set_ylabel(state_names[sind])
    ax1.set_title('OF vs LM')
    ax1.set_xlim([t0,tf])
    #ax1.legend(ncol = 2,loc = 'upper center') 
    
       
    ax2.plot(T,x_of[:,sind],label = 'OF')
    ax2.plot(T,x_dfsm[:,sind],color ='tab:green' ,label = 'LM + Corr')
    ax2.set_ylabel(state_names[sind])
    ax2.set_title('OF vs LM+Corr')
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim([t0,tf])
    #ax2.legend(ncol = 2,loc = 'upper center') 
    
    if save_flag:
        fig.savefig(SimulationDetails.simulation_name+' - ' + state_names[sind] +'.svg',format = 'svg',dpi = 1200)
    
    sind+=1
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    fig.suptitle(SimulationDetails.simulation_name + ' - '+state_names[sind])
    
    ax1.plot(T,x_of[:,sind],label = 'OF')
    ax1.plot(T,x_lm[:,sind],color ='tab:orange' ,label = 'LM')
    ax1.set_ylabel(state_names[sind])
    ax1.set_title('OF vs LM')
    ax1.set_xlim([t0,tf])
    #ax1.legend(ncol = 2,loc = 'upper center') 
    
       
    ax2.plot(T,x_of[:,sind],label = 'OF')
    ax2.plot(T,x_dfsm[:,sind],color ='tab:green' ,label = 'LM + Corr')
    ax2.set_ylabel(state_names[sind])
    ax2.set_title('OF vs LM+Corr')
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim([t0,tf])
    #ax2.legend(ncol = 2,loc = 'upper center')  
    
    if save_flag:
        fig.savefig(SimulationDetails.simulation_name+' - ' + state_names[sind] +'.svg',format = 'svg',dpi = 1200)
    
    
    
    x_error_linfit = x_of-x_lm
    x_error_dfsm = x_of-x_dfsm
    
    fig2,(ax1,ax2) = plt.subplots(1,2)
    fig2.suptitle(SimulationDetails.simulation_name +' - Error Histograms')
     
    sind = 0
    alpha = 0.5
    lab_linfit = 'OF-LM'
    lab_corr = 'OF-(LM+Corr)'
    

    ax1.hist(x_error_linfit[:,sind],alpha = alpha,label = lab_linfit ,color ='tab:orange')
    ax1.hist(x_error_dfsm[:,sind],alpha = alpha,label = lab_corr,color ='tab:green' )
    ax1.set_title(state_names[sind])
     
    sind+=1
     
    ax2.hist(x_error_linfit[:,sind],alpha = alpha,label = lab_linfit ,color ='tab:orange')
    ax2.hist(x_error_dfsm[:,sind],alpha = alpha,label = lab_corr,color ='tab:green' )
    ax2.set_title(state_names[sind])
    ax1.legend(ncol = 2,loc = 'upper center',bbox_to_anchor=(1, -0.05)) 
    
    if save_flag:
        fig2.savefig(SimulationDetails.simulation_name+' - ' + 'error_hist_state' +'.svg',format = 'svg',dpi = 1200)
    
    
    # fig,ax = plt.subplots(1)
    # fig.suptitle(SimulationDetails.simulation_name +' - Generator Power')
    
    # ax.plot(T,x_of[:,1]*U[:,1]/10,label = 'OF')
    # ax.plot(T,x_dfsm[:,1]*U[:,1]/10,label = 'LM + Corr',color ='tab:green' )
    # ax.set_ylabel('Generator Power [MW]')
    # ax.set_xlabel('Time [s]')
    # ax.legend(ncol = 2)
    # ax.set_xlim([t0,tf])
    
    # if save_flag:
    #     fig.savefig(SimulationDetails.simulation_name+' - ' + 'GenPwr' +'.svg',format = 'svg',dpi = 1200)
    
    
    
    

