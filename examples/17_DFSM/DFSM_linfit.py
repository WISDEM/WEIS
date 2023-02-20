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

def odefun_linear(t,x,u_fun,surrogate_model):
    
    '''
    Function to evaluate the derivative function for simulation
    '''
    
    # control inputs
    u = u_fun(t)
    
    # arrange states and inputs for evaluating DFSM
    input_ = np.hstack([x,u])
    input_ = input_.reshape([1,len(input_)])

    # evaluate DFSM function
    dx = np.dot(input_,surrogate_model)
    
    # reshape    
    dx = np.squeeze(dx)

    return dx

def calculate_MSE(dx1,dx2):
    
    # number of samples
    nt = len(dx1)
    
    # MSE
    MSE = np.sum((dx1-dx2)**2)/nt
    
    return MSE

def odefun_linear_corr(t,x,u_fun,surrogate_model,NN,scaler_input,scaler_output):
    
    '''
    Function to evaluate the derivative function for simulation
    '''
    
    # control inputs
    u = u_fun(t)
    
    # arrange states and inputs for evaluating DFSM
    input_ = np.hstack([x,u])
    input_ = input_.reshape([1,len(input_)])
    
    # scale inputs
    input_scaled = scaler_input.transform(input_)

    # evaluate DFSM function
    dx = np.dot(input_,surrogate_model) + scaler_output.inverse_transform(NN.predict(input_scaled))

    # reshape    
    dx = np.squeeze(dx)

    return dx
#------------------------------------------------------------------------------
def perform_Kmeans(inputs,outputs,nclusters):
    
    # inititalize output array
    ninputs = np.shape(inputs)[1]
    noutputs = np.shape(outputs)[1]
        
    outputs_km = np.zeros([nclusters,noutputs])
    
    if len(inputs) == nclusters:
        inputs_km = inputs 
        outputs_km = outputs 
        
    else:
        # perform kmeans clustering algorithm
        kmeans = KMeans(n_clusters = nclusters,n_init = 10).fit(X = inputs)
    
        # extract centroid centers and centroid index
        inputs_km = kmeans.cluster_centers_
        labels_ = kmeans.labels_
        
        
        # loop through and find the state derrivative values at the cluster centroids
        for icluster in range(nclusters):
            
            # index of elements of the original array present in the current cluster
            cluster_ind = (labels_ == icluster)
            
            # extract the sate derivative values at these indices
            outputs_ind = outputs[cluster_ind,:]
            
            # evaluate the state derivative values at the centroid of this cluster
            outputs_km[icluster,:] = np.mean(outputs_ind,axis=0)
        
    return inputs_km,outputs_km,ninputs,noutputs
#------------------------------------------------------------------------------    
def run_comparisons(SimulationDetails,nt,lin_model,SM,scaler_input,scaler_output,SM_type,save_flag):
    
     # reduce samples
     time,controls,states,state_derivatives,state_derivative2_ = reduce_samples(nt,SimulationDetails)
     
     # time span
     t0 = time[0]; tf = time[-1]
     
     # group states and controls
     inputs = np.hstack([states,controls])
     inputs_scaled = scaler_input.transform(inputs)
     
     # evaluate linear model
     dx_linfit = np.dot(inputs,lin_model)
     
     # add corrective function
     dx_combined = dx_linfit + scaler_output.inverse_transform(SM.predict(inputs_scaled))

     nstates = len(SimulationDetails.state_names)
     dx_names = SimulationDetails.state_dx_names
     
     if nstates == 4:
         
         subplot_ind = [(2,2,1),(2,2,2),(2,2,3),(2,2,4)]
         
     elif nstates == 2:
         
         subplot_ind = [(2,1,1),(2,1,2)]
             
     
     fig = plt.figure()
     fig.subplots_adjust(hspace = 0.45,wspace = 0.35)
     fig.suptitle(SimulationDetails.simulation_name + ' - State Derivatives')
     #fig.subplots_adjust(hspace = 0.45)
    
     for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time,state_derivatives[:,idx],label = 'OF')
        ax_idx.plot(time,dx_linfit[:,idx],label = 'LM')
        ax_idx.plot(time,dx_combined[:,idx],label = 'LM + Corr')
        ax_idx.set_xlim([t0,tf])
        ax_idx.set_ylabel(dx_names[idx])
        
     ax_idx.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center') 
     ax_idx.set_xlabel('Time [s]')
     
     dx_error_linfit = state_derivatives-dx_linfit 
     dx_error_dfsm = state_derivatives-dx_combined
     
     fig2,(ax1,ax2) = plt.subplots(1,2)
     fig2.suptitle(SimulationDetails.simulation_name +' - Error Histograms')
     
     sind = 0
     alpha = 0.5

     ax1.hist(dx_error_linfit[:,sind],alpha = alpha,label = 'OF - LM',color ='tab:orange')
     ax1.hist(dx_error_dfsm[:,sind],alpha = alpha,label = 'OF - (LM + Corr)',color ='tab:green')
     ax1.set_title(dx_names[sind])
     
     sind+=1
     
     ax2.hist(dx_error_linfit[:,sind],alpha = alpha,label = 'OF - LM',color ='tab:orange')
     ax2.hist(dx_error_dfsm[:,sind],alpha = alpha,label = 'OF - (LM + Corr)',color ='tab:green')
     ax2.set_title(dx_names[sind])
     ax1.legend(ncol = 2,loc = 'upper center',bbox_to_anchor=(1, -0.05)) 
     
     
     if save_flag:
         fig.savefig(SimulationDetails.simulation_name+'-Corr.svg',format = 'svg',dpi = 1200)
         fig2.savefig(SimulationDetails.simulation_name+'-Hist.svg',format = 'svg',dpi = 1200)

#------------------------------------------------------------------------------     
def construct_interpolating_function(time,quantity):
    
    qty_pp = PchipInterpolator(time,quantity)
    
    # create callable function
    qty_fun = lambda t:qty_pp(t)
    
    # return
    return qty_fun
     
#------------------------------------------------------------------------------     
def run_simulation(tspan,x0,u_fun,lin_model,solve_options,SimulationDetails,scaler_input,scaler_output,t_eval,iteration,save_flag,corrective_function = None):
     
    '''
    Function to run simulation using the DFSM and compare it to the original function
    '''
    
    # run simulation
    if corrective_function == None:
        
        corr_function_flag = False
        
    else:
        
        corr_function_flag = True
        

    # run simulation with linear model with corrective function
    if corrective_function:
        
        sol =  solve_ivp(odefun_linear_corr,tspan,x0,method=solve_options['method'],args = (u_fun,lin_model,corrective_function,scaler_input,scaler_output),rtol = solve_options['rtol'],atol = solve_options['atol'])
        
        # extract solution
        T = sol['t']
        x_dfsm = sol['y']; x_dfsm = x_dfsm.T
        U = u_fun(T)
        
        # create inputs to evaluate the derivative function predicted
        inputs = np.hstack([x_dfsm,U])
        
        # evaluate linear model
        dx_linfit = np.dot(inputs,lin_model)
        
        # evaluate error
        dx_error = corrective_function.predict(inputs)
        
        # add error
        dx_dfsm = dx_linfit + dx_error
        
        sim_label = 'LM + Corr'
   
    
    # run simulation with just the linear function
    sol_lm = solve_ivp(odefun_linear,tspan,x0,method = solve_options['method'],args = (u_fun,lin_model),rtol = solve_options['rtol'],atol = solve_options['atol'])
    
    x_linfit = sol_lm['y'];x_linfit = x_linfit.T
    
    # plot results
    noutputs = len(x0)
    
    # get original inputs states and outputs
    controls,states,state_derivatives,time = SimulationDetails.provide_inputs_outputs()
    
    # construct interpolating functions for these quantities
    x_fun_of = construct_interpolating_function(time,states)
    dx_fun_of = construct_interpolating_function(time,state_derivatives)
    
    if corr_function_flag:
        x_of = x_fun_of(T)
        dx_of = dx_fun_of(T)
    
    # get names
    state_names = SimulationDetails.state_names
    state_dx_names = SimulationDetails.state_dx_names
    

    if corr_function_flag:
        
        
        #plot_simulation_results(SimulationDetails,T,x_of,x_linfit,x_dfsm,U,state_names,save_flag)
    
    
        for n in range(noutputs):
            
            fig,(ax1,ax2) = plt.subplots(2,1)
            fig.subplots_adjust(hspace = 0.45)
            fig.suptitle('DFSM iteration - '+str(iteration))
            
            ax1.plot(T,dx_of[:,n])
            ax1.plot(T,dx_dfsm[:,n])
            ax1.set_xlim(tspan)
            ax1.set_ylabel(state_dx_names[n])
            ax1.set_title(SimulationDetails.simulation_name + ' - ' + state_names[n])
            
            ax2.plot(T,x_of[:,n],label = 'OF')
            ax2.plot(T,x_dfsm[:,n],label = sim_label )
            ax2.set_xlim(tspan)
            ax2.set_ylabel(state_names[n])
            ax2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
            ax2.set_xlabel('Time [s]')
            
            ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
            
        
        # evaluate dx error
        dx_error = dx_of - dx_dfsm
        
        dx_norm = norm(dx_error,axis = 1)
        
        dx_norm_mean = np.mean(dx_norm)
        
        error_ind = dx_norm > dx_norm_mean
        
        return dx_error[error_ind,:],inputs[error_ind,:]
    
    else:
        
        return dx_dfsm
    
#------------------------------------------------------------------------------
def train_corrective_function(ncluster,inputs_error,dx_error,fun_type = 'GPR'):
    
    # for the given set of errors determine suitable clusters
    inputs_err,outputs_err,nI,nO = perform_Kmeans(inputs_error,dx_error,ncluster)
    
    # split training and testing data
    train_ip = inputs_err
    train_dx_err = outputs_err
    
    # scaler
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    
    # scale input data
    scaler_input = sc1.fit(train_ip)
    train_ip_scaled = scaler_input.transform(train_ip)

    
    scaler_output = sc2.fit(train_dx_err)
    train_dx_err_scaled = scaler_output.transform(train_dx_err)
    
    if fun_type == 'NN':
        
        # train a neural network to predict the errors
        corr_fun = MLPRegressor(hidden_layer_sizes = (50,10),max_iter = 300,activation = 'logistic',solver = 'adam',verbose = True,tol = 1e-5)
        corr_fun.fit(train_ip_scaled,train_dx_err_scaled)
        
    elif fun_type == 'GPR':
        
        # train a GPR
        kernel = 1*RBFsk(length_scale = [1]*ninputs,length_scale_bounds=(1e-2, 1e2))
        corr_fun = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 5,random_state = 34534)
        corr_fun.fit(train_ip_scaled,train_dx_err_scaled)
        
    #elif fun_type == 'RBF':
        
    return corr_fun,scaler_input,scaler_output,fun_type

#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'outputs'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '1p1_samples_rated' + os.sep + 'lin_0.outb'
    
    # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq','BldPitch1',
                      'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    save_flag = False
    
    # create class instance
    training_samples = SimulationDetails(simulation_name = 'Simulation 1',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,add_dx2 = 0)
    
    # extract values
    training_samples.extract_channels()
    plot_inputs(training_samples,save_flag)
    

    # load validation file
    sim_file = datapath + os.sep + '1p1_samples_rated'+ os.sep + 'lin_0.outb'
    
    # create class instance
    validation_samples = SimulationDetails(simulation_name = 'Simulation 2',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,add_dx2 = 0)
    
    # extract values
    validation_samples.extract_channels()
    plot_inputs(validation_samples,save_flag)
    

    nstates = len(validation_samples.state_names)
    ncontrols = len(validation_samples.control_names)
    
    
    # input for training the DFSM
    controls,states,state_derivatives,time = training_samples.provide_inputs_outputs()
    inputs = np.hstack([states,controls]) 
    
    # generate ED samples
    nt = 2000
    solve_options = {'method':'DOP853','rtol':1e-9,'atol':1e-9}
    mu = 1
    
    # reduce samples
    time_,controls_,states_,state_derivatives_,state_derivatives2_ed = reduce_samples(nt,training_samples)
    
    inputs_fit = np.hstack([states_,controls_]); 
    X0_fit = states_[0,:]
    tspan_fit = [time_[0],time_[-1]]
    
    # create interpolating function for inputs and states
    X_pp_fit = PchipInterpolator(time_,states_)
    x_fun_fit = lambda t:X_pp_fit(t)
    U_pp_fit = PchipInterpolator(time_,controls_)
    u_fun_fit = lambda t:U_pp_fit(t)
    dx_pp_fit = PchipInterpolator(time_,state_derivatives_)
    dx_fun_fit = lambda t:dx_pp_fit(t)
    
    # reduce samples
    time_val,controls_val,states_val,state_derivatives_val,state_derivatives2_ed = reduce_samples(nt,validation_samples)
    
    # stack states and controls to get inputs
    inputs_val = np.hstack([states_val,controls_val]); 
    X0_val = states_val[0,:]
    tspan_val = [time_val[0],time_val[-1]]
    
    # create interpolating function for controls
    X_pp_val = PchipInterpolator(time_val,states_val)
    x_fun_val = lambda t:X_pp_val(t)
    U_pp_val = PchipInterpolator(time_val,controls_val)
    u_fun_val = lambda t: U_pp_val(t)
    dx_pp_val = PchipInterpolator(time_val,state_derivatives_val)
    dx_fun_val = lambda t:dx_pp_val(t)
    
    # plot the original value of the derivative function surrogate models for both the fit and validation trajectories
    fig1,(ax1,ax2) = plt.subplots(2,1)
    fig1.subplots_adjust(hspace = 0.45)
    fig1.suptitle(training_samples.simulation_name + ' - State Derivatives')
        
    ax1.plot(time_,state_derivatives_[:,0],label = 'OF')
    ax1.set_ylabel('dPtfmPitch')
    ax1.set_xlim([time_[0],time_[-1]])
    
    
    
    ax2.plot(time_,state_derivatives_[:,1],label = 'OF')
    ax2.set_ylabel('dGenSpeed')
    ax2.set_xlim([time_[0],time_[-1]])
    
    fig2,(ax3,ax4) = plt.subplots(2,1)
    fig2.subplots_adjust(hspace = 0.45)
    fig2.suptitle(validation_samples.simulation_name + ' - State Derivatives')
        
    ax3.plot(time_val,state_derivatives_val[:,0],label = 'OF')
    ax3.set_ylabel('dPtfmPitch')
    ax3.set_xlim([time_[0],time_[-1]])
    #ax3.set_title('Validation Trajectory')
    
    ax4.plot(time_val,state_derivatives_val[:,1],label = 'OF')
    ax4.set_ylabel('dGenSpeed')
    ax4.set_xlim([time_[0],time_[-1]])
    
    
    # generate ED samples
    nclusters = [500] # 70 for below rated
    A = []; X = []
    
    for icluster in nclusters:
        
        # perform kmeans clustering and obtain clusters
        inputs_km,outputs_km,ninputs,noutputs = perform_Kmeans(inputs,state_derivatives,icluster)
        
        # perform a linear fit between inputs and outputs
        LM = lstsq(inputs_km,outputs_km);LM = LM[0]; X.append(LM)
        
        # extract the states matrix and evaluate eigen values
        A_km = LM[0:nstates,0:nstates]; A.append(A_km)
        
        eig_km = eig(A_km)
        eig_km = eig_km[0]
        print('')
        print(eig_km)
        
        # run simulation
        # dx_sim_fit = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input = [],scaler_output = [],t_eval = time_,iteration = 0)
        # dx_sim_val = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input = [],scaler_output = [],t_eval = time_,iteration = 0)
        
        # evaluate derivativev alues predicted by DFSM
        dx_fit = np.dot(inputs_fit,LM)
        dx_val = np.dot(inputs_val,LM)
        
        # plot DFSM predicted values
        # fitting trajectory
        ax1.plot(time_,dx_fit[:,0],label = 'LM')
        ax2.plot(time_,dx_fit[:,1],label = 'LM')
        
        # validation trajectory
        ax3.plot(time_val,dx_val[:,0],label = 'LM')
        ax4.plot(time_val,dx_val[:,1],label = 'LM')
        
    # legend    
    ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax2.set_xlabel('Time [s]')
    ax4.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax4.set_xlabel('Time [s]')

    breakpoint()
    if save_flag:
        fig1.savefig(training_samples.simulation_name+'-LM.svg',format = 'svg',dpi = 1200)
        fig2.savefig(validation_samples.simulation_name+'LM.svg',format = 'svg',dpi = 1200)
    
    # evaluate the error between lin fit and original
    dx_error_fit = state_derivatives_ - dx_fit
    dx_error_val = state_derivatives_val - dx_val
    
    # combine inputs and outputs for the fit and validation tajectories
    inputs_combined = np.vstack([inputs_fit,inputs_val])
    dx_error_combined = np.vstack([dx_error_fit,dx_error_val])
    time_combined = np.vstack([time_,time_val])
    #dx_error_combined[:,1] = 0
    
    fig,ax = plt.subplots(1)
    ax.hist(dx_error_combined[:,0],label = 'dx1',alpha = 0.7)
    ax.hist(dx_error_combined[:,1],label = 'dx2',alpha = 0.7)
    ax.set_xlabel('error')
    ax.legend()
    
    # evaluate the norm of the errors and find means
    dx_norm = norm(dx_error_combined,axis = 1)
    dx_norm_mean = np.mean(dx_norm)
    
    fig,ax = plt.subplots(1)
    ax.hist(dx_norm,alpha = 0.7)
    ax.vlines(mu*dx_norm_mean,ymin = 0, ymax = 1000,color = 'k')
    ax.set_xlabel('norm(error)')
    ax.legend()

    # find the index of points with norm(error) grater than the mean
    error_ind = dx_norm > mu*dx_norm_mean
    ind1 = error_ind[0:nt]
    ind2 = error_ind[nt:]
    
    mk_size = 1
    
    # plot the points with large error
    ax1.plot(time_[ind1],dx_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    ax2.plot(time_[ind1],dx_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    
    ax3.plot(time_val[ind2],dx_val[ind2,0],'.',color = 'k',markersize = mk_size)
    ax4.plot(time_val[ind2],dx_val[ind2,1],'.',color = 'k',markersize = mk_size)
    
    # plot corresponding inputs
    subplot_ind = [(3,2,2),(3,2,4),(3,2,1),(3,2,3),(3,2,5)]
    #subplot_ind = [(5,1,4),(5,1,5),(5,1,1),(5,1,2),(5,1,3)]
    
    control_names = training_samples.control_names
    state_names = training_samples.state_names
    
    input_names = state_names+control_names
    
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.55,wspace = 0.35)
    fig.suptitle(training_samples.simulation_name + '- Inputs')
    
    for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time_,inputs_fit[:,idx],label = 'OF')
        ax_idx.plot(time_[ind1],inputs_fit[ind1,idx],'.',color = 'k',markersize = mk_size)
        ax_idx.set_xlim(tspan_fit)
        ax_idx.set_ylabel(input_names[idx])
        
        if ind[2] == 4 or ind[2] == 5:
            ax_idx.set_xlabel('Time [s]')
            
            
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.55,wspace = 0.35)
    fig.suptitle(validation_samples.simulation_name + '- Inputs')
    
    for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time_val,inputs_val[:,idx],label = 'OF')
        ax_idx.plot(time_val[ind2],inputs_val[ind2,idx],'.',color = 'k',markersize = mk_size)
        ax_idx.set_xlim(tspan_fit)
        ax_idx.set_ylabel(input_names[idx])
        
        if ind[2] == 4 or ind[2] == 5:
            ax_idx.set_xlabel('Time [s]')
    

    # extract the points with large error
    inputs_error = np.copy(inputs_combined)
    dx_error = np.copy(dx_error_combined) 


    input_names = validation_samples.state_names + validation_samples.control_names
    fun_type = 'GPR'
    ncluster_ = [1000] #700 for below rated
    
    for ncluster in ncluster_:
        print('')
        print(ncluster)
        corr_fun,scaler_input,scaler_output,fun_type = train_corrective_function(ncluster,inputs_error,dx_error,fun_type)
        
        
        nt_ = 500
        run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag )
        
        # nt_ = 50000
        # run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag)
        
        # # nt_ = 50
        # # run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type)
        
        
        # # nt_ = 500
        # # run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type)
        
        # nt_ = 50000
        # run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag)
        
        # nt_ = 50
        # run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type)
        
        
        # run simulation
        iteration = ncluster
        
        if all(eig_km.real<0):
            
            time_ = np.linspace(0,600,200)
            dx_error_fit2,inputs_fit2 = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input,scaler_output,time_,iteration,save_flag,corr_fun)
            #dx_error_val2,inputs_val2 = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input,scaler_output,time_,iteration,save_flag,corr_fun)    
