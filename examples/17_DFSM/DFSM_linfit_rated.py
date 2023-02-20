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

def odefun_linear(t,x,u_fun,surrogate_model,scaler_input,scaler_output):
    
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
    dx = scaler_output.inverse_transform(np.dot(input_scaled,surrogate_model))
    
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
    dx = scaler_output.inverse_transform((np.dot(input_scaled,surrogate_model)+NN.predict(input_scaled)))

    # reshape    
    dx = np.squeeze(dx)

    return dx
#------------------------------------------------------------------------------
def sample_IO(SimulationDetails,nclusters,sampling_method):
    
    # obtain values
    
    if sampling_method == 'kmeans':
        
        controls,states,state_derivatives,time = SimulationDetails.provide_inputs_outputs()

        inputs = np.hstack([states,controls])
        
        outputs = state_derivatives
        
        inputs,outputs,ninputs,noutputs = perform_Kmeans(inputs,outputs,nclusters)
        
    elif sampling_method == 'ED':
        
        time,controls,states,state_derivatives,state_derivative2_ = reduce_samples(nclusters,SimulationDetails)
        
        inputs = np.hstack([states,controls])
        
        outputs = state_derivatives
        
        ninputs = len(inputs.T);noutputs = len(outputs.T)
        
        
    return inputs,outputs,ninputs,noutputs
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
     dx_linfit = np.dot(inputs_scaled,lin_model)
     
     # add corrective function
     dx_error = (SM.predict(inputs_scaled))

     dx_combined = scaler_output.inverse_transform(dx_linfit + dx_error)
     dx_linfit = scaler_output.inverse_transform(dx_linfit)

     nstates = len(SimulationDetails.state_names)
     dx_names = SimulationDetails.state_dx_names
     
     if nstates >2:
         
         subplot_ind = [(2,2,1),(2,2,2),(2,2,3)] # ,(2,2,4)
         sind = 2
         
     elif nstates == 2:
         
         subplot_ind = [(2,1,1),(2,1,2)]
         sind = 0
             
     
     fig = plt.figure()
     fig.subplots_adjust(hspace = 0.5,wspace = 0.35)
     fig.suptitle(SimulationDetails.simulation_name + ' - State Derivatives')
     #fig.subplots_adjust(hspace = 0.45)
    
     for idx,ind in enumerate(subplot_ind):
        ax_idx = plt.subplot(ind[0],ind[1],ind[2])
        ax_idx.plot(time,state_derivatives[:,idx],label = 'OF')
        ax_idx.plot(time,(dx_linfit[:,idx]),label = 'LM')
        ax_idx.plot(time,dx_combined[:,idx],label = 'LM + Corr')
        ax_idx.set_xlim([t0,tf])
        ax_idx.set_title(dx_names[idx])
        
     ax_idx.legend(ncol = 3,bbox_to_anchor=(-0.5, -0.05),loc = 'upper center') 
     ax_idx.set_xlabel('Time [s]')
     
     dx_error_linfit = state_derivatives-dx_linfit 
     dx_error_dfsm = state_derivatives-dx_combined
     
     fig2,(ax1,ax2) = plt.subplots(1,2)
     fig2.suptitle(SimulationDetails.simulation_name +' - Error Histograms')
     

     
     alpha = 0.5

     ax1.hist(dx_error_linfit[:,sind],alpha = alpha,label = 'OF - LM',color ='tab:orange')
     ax1.hist(dx_error_dfsm[:,sind],alpha = alpha,label = 'OF - (LM + Corr)',color ='tab:green')
     ax1.set_title(dx_names[sind])
     
     sind+=1
     
     ax2.hist(dx_error_linfit[:,sind],alpha = alpha,label = 'OF - LM',color ='tab:orange')
     ax2.hist(dx_error_dfsm[:,sind],alpha = alpha,label = 'OF - (LM + Corr)',color ='tab:green')
     ax2.set_title(dx_names[sind])
     ax1.legend(ncol = 2,loc = 'upper center',bbox_to_anchor=(1, -0.1)) 
     
     
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
        inputs_scaled = scaler_input.transform(inputs)
        
        # evaluate linear model
        dx_linfit = np.dot(inputs_scaled,lin_model)
        
        # evaluate error
        dx_error = corrective_function.predict(inputs_scaled)
       
        
        # add error
        dx_dfsm = dx_linfit + dx_error
        
        # scale
        dx_dfsm = scaler_output.inverse_transform(dx_dfsm)
        
        sim_label = 'LM + Corr'
   
    
    # run simulation with just the linear function
    sol_lm = solve_ivp(odefun_linear,tspan,x0,method = solve_options['method'],args = (u_fun,lin_model,scaler_input,scaler_output),rtol = solve_options['rtol'],atol = solve_options['atol'])
    
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
def train_corrective_function(ncluster,inputs_error,dx_error,scaler_input,scaler_output,fun_type = 'GPR'):
    
    
    # for the given set of errors determine suitable clusters
    inputs_err,outputs_err,nI,nO = perform_Kmeans(inputs_error,dx_error,ncluster)
    ninputs = len(inputs_err.T)
    
    # split training and testing data
    train_ip = inputs_err
    train_dx_err = outputs_err

    # scale input data    
    train_ip_scaled = scaler_input.transform(train_ip)

    train_dx_err_scaled = scaler_output.transform(train_dx_err)
    

    if fun_type == 'NN':
        
        # train a neural network to predict the errors
        corr_fun = MLPRegressor(hidden_layer_sizes = (50,10),max_iter = 300,activation = 'logistic',solver = 'adam',verbose = True,tol = 1e-5)
        corr_fun.fit(train_ip_scaled,train_dx_err_scaled)
        
    elif fun_type == 'GPR':
        
        # train a GPR
        kernel = 1*RBFsk(length_scale = [1]*ninputs,length_scale_bounds=(1e-2, 5*1e2))
        corr_fun = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 5,random_state = 34534)
        corr_fun.fit(train_ip_scaled,train_dx_err_scaled)
        
    #elif fun_type == 'RBF':
        
    return corr_fun,fun_type

#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'outputs'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '1p1_samples_rated' + os.sep + 'lin_1.outb'
    
    # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq','BldPitch1',
                      'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    save_flag = False
    
    # create class instance
    training_samples = SimulationDetails(simulation_name = 'Simulation 1',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,add_dx2 = 1)
    
    # extract values
    training_samples.extract_channels()
    plot_inputs(training_samples,save_flag)
    

    # load validation file
    sim_file = datapath + os.sep + '1p1_samples_rated'+ os.sep + 'lin_1.outb'
    
    # create class instance
    validation_samples = SimulationDetails(simulation_name = 'Simulation 2',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1,add_dx2 = 1)
    
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
    solve_options = {'method':'LSODA','rtol':1e-6,'atol':1e-6}
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
    nclusters = [100] # 70 for below rated
    
    sampling_method = 'kmeans'
    A = []; X = []
    
    for icluster in nclusters:
        
        print(icluster)
        
        # perform kmeans clustering and obtain clusters
        inputs,outputs,ninputs,noutputs = sample_IO(training_samples,icluster,sampling_method)
  
        sc1 = StandardScaler()
        sc2 = StandardScaler()
        
        scaler_input = sc1.fit(inputs)
        scaler_output = sc2.fit(outputs)
        
        inputs_scaled = scaler_input.transform(inputs)
        outputs_scaled = scaler_output.transform(outputs)
        
        # perform a linear fit between inputs and outputs
        LM = lstsq(inputs_scaled,outputs_scaled,rcond = None);LM = LM[0]; X.append(LM)
        
        # extract the states matrix and evaluate eigen values
        A_km = LM[0:nstates,0:nstates]; A.append(A_km)
        
        eig_km = eig(A_km.T)
        eig_km = eig_km[0]
        print('')
        print(eig_km)

        # run simulation
        # dx_sim_fit = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input = [],scaler_output = [],t_eval = time_,iteration = 0)
        # dx_sim_val = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input = [],scaler_output = [],t_eval = time_,iteration = 0)
        
        # evaluate derivativev alues predicted by DFSM
        dx_fit = np.dot(sc1.transform(inputs_fit),LM)
        dx_fit = sc2.inverse_transform(dx_fit)
        dx_val = np.dot(sc1.transform(inputs_val),LM)
        dx_val = sc2.inverse_transform(dx_val)
        
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
    

    # # find the index of points with norm(error) grater than the mean
    # error_ind = dx_norm > mu*dx_norm_mean
    # ind1 = error_ind[0:nt]
    # ind2 = error_ind[nt:]
    
    # mk_size = 1
    
    # # plot the points with large error
    # ax1.plot(time_[ind1],dx_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    # ax2.plot(time_[ind1],dx_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    
    # ax3.plot(time_val[ind2],dx_val[ind2,0],'.',color = 'k',markersize = mk_size)
    # ax4.plot(time_val[ind2],dx_val[ind2,1],'.',color = 'k',markersize = mk_size)
    
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
    dx_ind = [False,False,False,True]
    #dx_error[:,0:2] = 0

    
    input_names = validation_samples.state_names + validation_samples.control_names
    fun_type = 'GPR'
    ncluster_ = [500] #700 for below rated
    
    for ncluster in ncluster_:
        print('')
        print(ncluster)
        corr_fun,fun_type = train_corrective_function(ncluster,inputs_error,dx_error,scaler_input,scaler_output,dx_ind,fun_type)
        
        
        nt_ = 500
        run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,dx_ind,save_flag )
        
        # nt_ = 50000
        # run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag)
        
        # nt_ = 50
        # run_comparisons(training_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type)
        
        
        # nt_ = 500
        # run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type)
        
        # nt_ = 50000
        # run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,save_flag)
        
        nt_ = 500
        run_comparisons(validation_samples,nt_,LM,corr_fun,scaler_input,scaler_output,fun_type,dx_ind,save_flag)
        
        
        #run simulation
        iteration = ncluster
        
        if all(eig_km.real<0):
            
            time_ = None
            dx_error_fit2,inputs_fit2 = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input,scaler_output,time_,iteration,dx_ind,save_flag,corr_fun)
            #dx_error_val2,inputs_val2 = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input,scaler_output,time_,iteration,dx_ind,save_flag,corr_fun)    


    
    # inputs_error2 = np.vstack([inputs_fit2,inputs_val2])
    # dx_error2 = np.vstack([dx_error_fit2,dx_error_val2])
    
    # inputs_error = np.vstack([inputs_error,inputs_error2])
    # dx_error = np.vstack([dx_error,dx_error2])
    
    # ncluster = 500
    # corr_fun,scaler_input,scaler_output,fun_type = train_corrective_function(ncluster,inputs_error,dx_error,fun_type)
    
    # iteration+=1
    
    # dx_error_fit3,inputs_fit3 = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input,scaler_output,time_,iteration,corr_fun)
    # dx_error_val3,inputs_val3 = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input,scaler_output,time_,iteration,corr_fun)    

    # inputs_error3 = np.vstack([inputs_fit3,inputs_val3])
    # dx_error3 = np.vstack([dx_error_fit3,dx_error_val3])

    # inputs_error = np.vstack([inputs_error,inputs_error3])
    # dx_error = np.vstack([dx_error,dx_error3])
    
    # ncluster = 500
    # corr_fun,scaler_input,scaler_output,fun_type = train_corrective_function(ncluster,inputs_error,dx_error,fun_type)
    
    # iteration+=1
    
    # dx_error_fit4,inputs_fit4 = run_simulation(tspan_fit,X0_fit,u_fun_fit,LM,solve_options,training_samples,scaler_input,scaler_output,time_,iteration,corr_fun)
    # dx_error_val4,inputs_val4 = run_simulation(tspan_val,X0_val,u_fun_val,LM,solve_options,validation_samples,scaler_input,scaler_output,time_,iteration,corr_fun)    
    
    # breakpoint()
    # sol_fit = solve_ivp(odefun_linear_corr,tspan_fit,X0_fit,method=solve_method,args = (u_fun_fit,LM,gpr,scaler_input,scaler_output),rtol = solve_options['rtol'],atol = solve_options['atol'])
    
    # # extract solution
    # T_fit = sol_fit['t']
    # X_fit = sol_fit['y']; X_fit = X_fit.T
    # U_fit = u_fun_fit(T_fit)
    # X_fit_of = x_fun_fit(T_fit)
    # dx_fit_of = dx_fun_fit(T_fit)
    
    # mse_fit_x1 = calculate_MSE(X_fit_of[:,0],X_fit[:,0])
    # mse_fit_x2 = calculate_MSE(X_fit_of[:,1],X_fit[:,1])
    
    # # create inputs to evaluate the derivative function predicted
    # inputs_fit_ = np.hstack([X_fit,U_fit])
    # input_scaled_ = scaler_input.transform(inputs_fit_)
    # dx_fit_ = np.dot(inputs_fit_,LM) 
    # dx_fit_corr,std_fit_corr = gpr.predict(input_scaled_,return_std = True) 
    # dx_fit_corr = scaler_output.inverse_transform(dx_fit_corr)
    # dx_fit_dfsm = dx_fit_ + dx_fit_corr
    
    # dx_fit_error2 = dx_fit_of - dx_fit_dfsm
    
    # # run simulation
    # sol_val = solve_ivp(odefun_linear_corr,tspan_val,X0_val,t_eval = T_fit,method=solve_method,args = (u_fun_val,LM,gpr,scaler_input,scaler_output),rtol = solve_options['rtol'],atol = solve_options['atol'])
    
    # # extract solution
    # T_val = sol_val['t']
    # X_val = sol_val['y']; X_val = X_val.T
    # U_val = u_fun_val(T_val)
    # X_val_of = x_fun_val(T_val)
    # dx_val_of = dx_fun_val(T_val)
    
    
    # mse_val_x1 = calculate_MSE(X_val_of[:,0],X_val[:,0])
    # mse_val_x2 = calculate_MSE(X_val_of[:,1],X_val[:,1])
    
    
    # # create inputs to evaluate the derivative function predicted
    # inputs_val_ = np.hstack([X_val,U_val])
    # input_scaled_ = scaler_input.transform(inputs_val_)
    # dx_val_ = np.dot(inputs_val_,LM) 
    # dx_val_corr,std_val_corr = gpr.predict(input_scaled_,return_std = True) 
    # dx_val_corr = scaler_output.inverse_transform(dx_val_corr)
    # dx_val_dfsm = dx_val_ + dx_val_corr
    
    # dx_val_error2 = dx_val_of - dx_val_dfsm
    
    # dx_error_combined2 = np.vstack([dx_fit_error2,dx_val_error2])
    
    # # combine inputs and outputs for the fit and validation tajectories
    # inputs_combined2 = np.vstack([inputs_fit_,inputs_val_])
    # time_combined2 = np.vstack([T_fit,T_val])
    
    # dx_norm2 = norm(dx_error_combined2,axis = 1)
    # dx_norm_mean2 = np.mean(dx_norm2)
    
    
    
    # error_ind = (dx_norm2>dx_norm_mean2)
    # nt_ = len(T_fit)
    
    # ind1 = error_ind[0:nt_]
    # ind2 = error_ind[nt_:]
    
    
    
    # #-------------------------------------------------------------------
    # # plot PtfmPitch
    # fig_fit1,(ax_dx_f1,ax_x_f1) = plt.subplots(2,1)
    # fig_fit1.subplots_adjust(hspace = 0.45)
    
    # ax_dx_f1.plot(T_fit,dx_fit_of[:,0])
    # ax_dx_f1.plot(T_fit,dx_fit_dfsm[:,0])
    # ax_dx_f1.plot(T_fit[ind1],dx_fit_dfsm[ind1,0],'.',color = 'k',markersize = mk_size)
    # #ax_dx_f1.fill_between(T_fit.ravel(),dx_fit_[:,0]-1.96*std_fit_corr[:,0],dx_fit_[:,0]+1.96*std_fit_corr[:,0],alpha = 0.3)
    # ax_dx_f1.set_xlim(tspan_fit)
    # ax_dx_f1.set_ylabel('dPtfmPitch')
    # ax_dx_f1.set_title('Fitting Trajectory - PtfmPitch-' + 'MSE='+str(mse_fit_x1))
    
    # ax_x_f1.plot(time_,states_[:,0],label = 'OF')
    # ax_x_f1.plot(T_fit,X_fit[:,0],label = 'DFSM+GPR')
    # ax_x_f1.plot(T_fit[ind1],X_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    # ax_x_f1.set_xlim(tspan_fit)
    # ax_x_f1.set_ylabel('PtfmPitch')
    # ax_x_f1.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    # ax_x_f1.set_xlabel('Time [s]')
    
    # # plot genspeed
    # fig_fit2,(ax_dx_f2,ax_x_f2) = plt.subplots(2,1)
    # fig_fit2.subplots_adjust(hspace = 0.45)
    
    # ax_dx_f2.plot(T_fit,dx_fit_of[:,1])
    # ax_dx_f2.plot(T_fit,dx_fit_dfsm[:,1])
    # ax_dx_f2.plot(T_fit[ind1],dx_fit_dfsm[ind1,1],'.',color = 'k',markersize = mk_size)
    # #ax_dx_f2.fill_between(T_fit.ravel(),dx_fit_[:,1]-1.96*std_fit_corr[:,1],dx_fit_[:,1]+1.96*std_fit_corr[:,1],alpha = 0.3)
    # ax_dx_f2.set_xlim(tspan_fit)
    # ax_dx_f2.set_ylabel('dGenSpeed')
    # ax_dx_f2.set_title('Fitting Trajectory - GenSpeed-' + 'MSE='+str(mse_fit_x2))
    
    # ax_x_f2.plot(time_,states_[:,1],label = 'OF')
    # ax_x_f2.plot(T_fit,X_fit[:,1],label = 'DFSM+GPR')
    # ax_x_f2.plot(T_fit[ind1],X_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    # ax_x_f2.set_xlim(tspan_fit)
    # ax_x_f2.set_ylabel('GenSpeed')
    # ax_x_f2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    # ax_x_f2.set_xlabel('Time [s]')

    # # plot 
    # # plot PtfmPitch
    # fig_val1,(ax_dx_v1,ax_x_v1) = plt.subplots(2,1)
    # fig_val1.subplots_adjust(hspace = 0.45)
    
    # ax_dx_v1.plot(T_val,dx_val_of[:,0])
    # ax_dx_v1.plot(T_val,dx_val_dfsm[:,0])
    # ax_dx_v1.plot(T_val[ind2],dx_val_dfsm[ind2,0],'.',color = 'k',markersize = mk_size)
    # #ax_dx_v1.fill_between(T_val.ravel(),dx_val_[:,0]-1.96*std_val_corr[:,0],dx_val_[:,0]+1.96*std_val_corr[:,0],alpha = 0.3)
    # ax_dx_v1.set_xlim(tspan_val)
    # ax_dx_v1.set_ylabel('dPtfmPitch')
    # ax_dx_v1.set_title('Validation Trajectory - PtfmPitch-' + 'MSE='+str(mse_val_x1))
    
    # ax_x_v1.plot(time_val,states_val[:,0],label = 'OF')
    # ax_x_v1.plot(T_val,X_val[:,0],label = 'DFSM+GPR')
    # ax_x_v1.plot(T_val[ind2],X_val[ind2,0],'.',color = 'k',markersize = mk_size)
    # ax_x_v1.set_xlim(tspan_val)
    # ax_x_v1.set_ylabel('PtfmPitch')
    # ax_x_v1.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    # ax_x_v1.set_xlabel('Time [s]')
    
    # # plot genspeed
    # fig_val2,(ax_dx_v2,ax_x_v2) = plt.subplots(2,1)
    # fig_val2.subplots_adjust(hspace = 0.45)
    
    # ax_dx_v2.plot(T_val,dx_val_of[:,1])
    # ax_dx_v2.plot(T_val,dx_val_dfsm[:,1])
    # ax_dx_v2.plot(T_val[ind2],dx_val_dfsm[ind2,1],'.',color = 'k',markersize = mk_size)
    # #ax_dx_v2.fill_between(T_val.ravel(),dx_val_[:,1]-1.96*std_val_corr[:,1],dx_val_[:,1]+1.96*std_val_corr[:,1],alpha = 0.3)
    # ax_dx_v2.set_xlim(tspan_val)
    # ax_dx_v2.set_ylabel('dGenSpeed')
    # ax_dx_v2.set_title('Validation Trajectory - GenSpeed-' + 'MSE='+str(mse_val_x2))
    
    # ax_x_v2.plot(time_val,states_val[:,1],label = 'OF')
    # ax_x_v2.plot(T_val,X_val[:,1],label = 'DFSM+GPR')
    # ax_x_v2.plot(T_val[ind2],X_val[ind2,1],'.',color = 'k',markersize = mk_size)
    # ax_x_v2.set_xlim(tspan_val)
    # ax_x_v2.set_ylabel('GenSpeed')
    # ax_x_v2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    # ax_x_v2.set_xlabel('Time [s]')
    
    
    
    
    
    
    
    # initialize storage arrays
    # dx_fit_corrected_mlp = np.zeros((nt,noutputs))
    # dx_val_corrected_mlp = np.zeros((nt,noutputs))
    
    # dx_fit_corrected_gpr = np.zeros((nt,noutputs))
    # dx_val_corrected_gpr = np.zeros((nt,noutputs))
    
    # # loop through and evaluate 
    # for i in range(nt):
        
    #     ip_fit = inputs_fit[i,:]
    #     ip_val = inputs_val[i,:]
        
    #     ip_fit = ip_fit.reshape([1,len(ip_fit)])
    #     ip_fit_scaled = scaler_input.transform(ip_fit)
    #     ip_val = ip_val.reshape([1,len(ip_val)])
    #     ip_val_scaled = scaler_input.transform(ip_val)
        
    #     #dx_eval_ed[i,:] = np.squeeze(np.dot(ip,X_ed))
    #     # dx_fit_corrected_mlp[i,:] = np.squeeze(np.dot(ip_fit,X_km)+scaler_output.inverse_transform(mlp_reg.predict(ip_fit_scaled)))
    #     # dx_val_corrected_mlp[i,:] = np.squeeze(np.dot(ip_val,X_km)+scaler_output.inverse_transform(mlp_reg.predict(ip_val_scaled)))
        
    #     dx_fit_corrected_gpr[i,:] = np.squeeze(np.dot(ip_fit,LM)+scaler_output.inverse_transform(gpr.predict(ip_fit_scaled))) 
    #     dx_val_corrected_gpr[i,:] = np.squeeze(np.dot(ip_val,LM)+scaler_output.inverse_transform(gpr.predict(ip_val_scaled)))
        
    # # plot
    # fig,(ax1,ax2) = plt.subplots(2,1)
    # fig.subplots_adjust(hspace = 0.45)
        
    # ax1.plot(time_,state_derivatives_[:,0],label = 'OF')
    # ax1.plot(time_,dx_fit[:,0],label = 'DFSM_linfit')
    # #ax1.plot(time_,dx_fit_corrected_mlp[:,0],label = 'DFSM + NN')
    # ax1.plot(time_,dx_fit_corrected_gpr[:,0],label = 'DFSM + GPR')
    # ax1.set_ylabel('dx1')
    # ax1.set_xlim([time_[0],time_[-1]])
    # ax1.set_title('Fitting Trajectory')
    
    # ax2.plot(time_,state_derivatives_[:,1],label = 'OF')
    # ax2.plot(time_,dx_fit[:,1],label = 'DFSM_linfit')
    # #ax2.plot(time_,dx_fit_corrected_mlp[:,1],label = 'DFSM + NN')
    # ax2.plot(time_,dx_fit_corrected_gpr[:,1],label = 'DFSM + GPR')
    # ax2.set_ylabel('dx2')
    # ax2.set_xlim([time_[0],time_[-1]])
    
    # fig,(ax3,ax4) = plt.subplots(2,1)
    # fig.subplots_adjust(hspace = 0.45)
        
    # ax3.plot(time_val,state_derivatives_val[:,0],label = 'OF')
    # ax3.plot(time_val,dx_val[:,0],label = 'DFSM_linfit')
    # #ax3.plot(time_val,dx_val_corrected_mlp[:,0],label = 'DFSM + NN')
    # ax3.plot(time_val,dx_val_corrected_gpr[:,0],label = 'DFSM + GPR')
    # ax3.set_ylabel('dx1')
    # ax3.set_xlim([time_[0],time_[-1]])
    # ax3.set_title('Validation Trajectory')
    
    # ax4.plot(time_val,state_derivatives_val[:,1],label = 'OF')
    # ax4.plot(time_val,dx_val[:,1],label = 'DFSM_linfit')
    # #ax4.plot(time_val,dx_val_corrected_mlp[:,1],label = 'DFSM + NN')
    # ax4.plot(time_val,dx_val_corrected_gpr[:,1],label = 'DFSM + GPR')
    # ax4.set_ylabel('dx2')
    # ax4.set_xlim([time_[0],time_[-1]])
    
    # # plot the points with large error
    # ax1.plot(time_[ind1],dx_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    # ax2.plot(time_[ind1],dx_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    
    # ax3.plot(time_val[ind2],dx_val[ind2,0],'.',color = 'k',markersize = mk_size)
    # ax4.plot(time_val[ind2],dx_val[ind2,1],'.',color = 'k',markersize = mk_size)
    
    # #ax1.plot(time_,dx_fit_corrected[:,0],label = 'corrected')

    # ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax2.set_xlabel('Time [s]')
    # ax4.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax4.set_xlabel('Time [s]')
    
    # # dx_error_fit_mlp = state_derivatives_ - dx_fit_corrected_mlp
    # # dx_error_val_mlp = state_derivatives_val - dx_fit_corrected_mlp
    
    # # dx_error_combined_mlp = np.vstack([dx_error_fit_mlp,dx_error_val_mlp])
    
    # dx_error_fit_gpr = state_derivatives_ - dx_fit_corrected_gpr
    # dx_error_val_gpr = state_derivatives_val - dx_fit_corrected_gpr
    
    # dx_error_combined_gpr = np.vstack([dx_error_fit_gpr,dx_error_val_gpr])
    
    # # fig,ax = plt.subplots(1)
    
    # # ax.hist(dx_error_combined_mlp[:,0],label = 'Correction',alpha = 0.6)
    # # ax.hist(dx_error_combined[:,0],label = 'linfit',alpha = 0.6)
    # # ax.legend()
    # # ax.set_title('error histogram for NN dx1')
    
    # fig,ax = plt.subplots(1)
    
    # ax.hist(dx_error_combined_gpr[:,0],label = 'Correction',alpha = 0.6)
    # ax.hist(dx_error_combined[:,0],label = 'linfit',alpha = 0.6)
    # ax.legend()
    # ax.set_title('error histogram for GPR dx1')
    
    
    # for i_ip in range(ninputs):
    #     fig,ax = plt.subplots(1)
    #     ax.hist(inputs_km[:,i_ip],label = 'Fit',alpha = 0.7)
    #     ax.hist(inputs_err[:,i_ip],label = 'Error',alpha = 0.7)
    #     ax.set_xlabel(input_names[i_ip])
    #     ax.legend()
        
        
    
    
    
    
    
    
    
        