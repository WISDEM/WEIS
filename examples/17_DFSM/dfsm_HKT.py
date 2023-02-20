import os
import numpy as np
import matplotlib.pyplot as plt
from DFSM import SimulationDetails,DFSM,reduce_samples
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq,eig,norm
from scipy.interpolate import PchipInterpolator,Rbf
from scipy.integrate import solve_ivp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk

from create_plots import plot_inputs,plot_simulation_results
from DFSM_linfit_rated import perform_Kmeans,train_corrective_function,run_comparisons,run_simulation

def odefun_linear_corr(t,x,u_fun,A_fun,B_fun,wmin,wmax,NN,scaler_input,scaler_output):
    
    '''
    Function to evaluate the derivative function for simulation
    '''
    
    # control inputs
    u = u_fun(t)

    # arrange states and inputs for evaluating DFSM
    input_ = np.hstack([x,u])
    
    input_ = input_.reshape([1,len(input_)])   

    input_scaled = scaler_input.transform(input_)
        
    dx =  scaler_output.inverse_transform(NN.predict(input_scaled)) # + NN.predict(input_scaled)

    # reshape    
    dx = np.squeeze(dx)

    return dx


if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'OldDetailDesign'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '26kW_UNH_Mega.out'
    
    # required channels
    reqd_channels = ['Wave1Elev','B1Surge','B1Pitch'] # ,'B1Heave'
    chan_unit = ['[m]','[m]','[m]','[m]','[rad]'] 
        
    # control index
    control_ind = [0]
    state_ind = [1,2]
    
    save_flag = True
    
    # extract training samples
    training_samples = SimulationDetails(simulation_name = 'Simulation 1',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,
                                       add_dx2 = 0,tmin = 0,tmax = 200)
    
    # extract validation samples
    validation_samples = SimulationDetails(simulation_name = 'Simulation 1',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,
                                       add_dx2 = 0,tmin = 200,tmax = 400)
    
    # extract details
    training_samples.extract_channels()
    validation_samples.extract_channels()
    
    # plot inputs
    #plot_inputs(training_samples,save_flag)
    
    nstates = len(training_samples.state_names)
    ncontrols = len(training_samples.control_names)
    dx_names = training_samples.state_dx_names
    
    
    # input for training the DFSM
    controls,states,state_derivatives,time = training_samples.provide_inputs_outputs()
    u_fit_pp = PchipInterpolator(time,controls)
    u_fun_fit = lambda t:u_fit_pp(t)
    
    controls_val,states_val,state_derivatives_val,time_val = validation_samples.provide_inputs_outputs()
    u_val_pp = PchipInterpolator(time_val,controls_val)
    u_val_fit = lambda t:u_val_pp(t)
     
    inputs = np.hstack([states,controls]) 
    
    # generate ED samples
    nt = 2000
    solve_options = {'method':'Radau','rtol':1e-9,'atol':1e-9}
    mu = 1
    
    # reduce samples
    time_,controls_,states_,state_derivatives_,state_derivatives2_ed = reduce_samples(nt,training_samples)
    
    inputs_fit = np.hstack([states_,controls_]); 
    X0_fit = states_[0,:]
    tspan_fit = [time_[0],time_[-1]]
    
    subplot_ind = [(2,2,1),(2,2,2)] #,(2,2,4),
    ax_ = []
             
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.5,wspace = 0.4)
    fig.suptitle(training_samples.simulation_name+ '- Outputs')
    #fig.subplots_adjust(hspace = 0.45)
   
    for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time_,state_derivatives_[:,idx],label = 'OF')
        ax_idx.set_xlim(tspan_fit)
        ax_idx.set_title(dx_names[idx])
        ax_.append(ax_idx)
       
    ax_idx.set_xlabel('Time [s]')
        
    
    # generate ED samples
    nclusters = [100] # 70 for below rated
    A = []; X = []
    
    for icluster in nclusters:
        print('')
        print(icluster)
        # perform kmeans clustering and obtain clusters
        inputs_km,outputs_km,ninputs,noutputs = perform_Kmeans(inputs,state_derivatives,icluster)
        
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        
        scaler_input.fit(inputs_fit)
        scaler_output.fit(state_derivatives_)
        
        inputs_scaled = scaler_input.transform(inputs_km)
        outputs_scaled = scaler_output.transform(outputs_km)
        
        # perform a linear fit between inputs and outputs
        LM = lstsq(inputs_scaled,outputs_scaled,rcond = None);LM = LM[0]; X.append(LM)
        
        # extract the states matrix and evaluate eigen values
        A_km = LM[0:nstates,0:nstates]; A.append(A_km)
        
        eig_km = eig(A_km)
        eig_km = eig_km[0]
        print('')
        print(eig_km)
        
        # evaluate derivativev alues predicted by DFSM
        dx_fit = np.dot(scaler_input.transform(inputs_fit),LM)
        dx_fit = scaler_output.inverse_transform(dx_fit)
        
        #dx_val = 
        
        # plot DFSM predicted values
        for idx,ind in enumerate(subplot_ind):
            ax_[idx].plot(time_,dx_fit[:,idx],label = 'LM')
            
    dx_error = state_derivatives_ - dx_fit
    inputs_error = inputs_fit
    
    fun_type = 'GPR'
    ncluster_ = [500] #700 for below rated
    
    for ncluster in ncluster_:
        print('')
        print(ncluster)
        corr_fun,fun_type = train_corrective_function(ncluster,inputs_error,dx_error,scaler_input,scaler_output,fun_type)
        
        
        nt_ = 500
        run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag)
        
        iteration = ncluster
        
        #if all(eig_km.real<0):
            
        time_ = None;dx_ind = []
        dx_error_fit2,inputs_fit2 = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input,scaler_output,time_,iteration,save_flag,corr_fun)
                                                    
    
    

