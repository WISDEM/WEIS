import os
import numpy as np
import matplotlib.pyplot as plt
from DFSM import SimulationDetails,DFSM,reduce_samples
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq,eig,norm

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
    
    

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'outputs'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '1p1_samples_rated' + os.sep + 'lin_0.outb'
    
        # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq',
                     'BldPitch1', 'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    # create class instance
    training_samples = SimulationDetails(simulation_name = 'training',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1)
    
    # extract values
    training_samples.extract_channels()
    
    # load validation file
    sim_file = datapath + os.sep + '1p1_samples_rated'+ os.sep + 'lin_1.outb'
    
    # create class instance
    validation_samples = SimulationDetails(simulation_name = 'validation',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind,filter_flag = 1)
    
    # extract values
    validation_samples.extract_channels()
    
    
    # input for training the DFSM
    controls,states,state_derivatives = training_samples.provide_inputs_outputs()
    inputs = np.hstack([states,controls]) 
    
    # generate ED samples
    nt = 2000
    
    # reduce samples
    time_,controls_,states_,state_derivatives_,state_derivatives2_ed = reduce_samples(nt,training_samples)
    
    inputs_fit = np.hstack([states_,controls_]); 
    
    # reduce samples
    time_val,controls_val,states_val,state_derivatives_val,state_derivatives2_ed = reduce_samples(nt,validation_samples)
    
    # stack states and controls to get inputs
    inputs_val = np.hstack([states_val,controls_val]); 
    
    # plot the original value of the derivative function surrogate models for both the fit and validation trajectories
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
        
    ax1.plot(time_,state_derivatives_[:,0],label = 'OF')
    ax1.set_ylabel('dx1')
    ax1.set_xlim([time_[0],time_[-1]])
    ax1.set_title('Fitting Trajectory')
    
    ax2.plot(time_,state_derivatives_[:,1],label = 'OF')
    ax2.set_ylabel('dx2')
    ax2.set_xlim([time_[0],time_[-1]])
    
    fig,(ax3,ax4) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
        
    ax3.plot(time_val,state_derivatives_val[:,0],label = 'OF')
    ax3.set_ylabel('dx1')
    ax3.set_xlim([time_[0],time_[-1]])
    ax3.set_title('Validation Trajectory')
    
    ax4.plot(time_val,state_derivatives_val[:,1],label = 'OF')
    ax4.set_ylabel('dx2')
    ax4.set_xlim([time_[0],time_[-1]])
    
    
    # generate ED samples
    nclusters = [500]
    A = []; X = []
    
    for icluster in nclusters:
        
        # perform kmeans clustering and obtain clusters
        inputs_km,outputs_km,ninputs,noutputs = perform_Kmeans(inputs,state_derivatives,icluster)
        
        # perform a linear fit between inputs and outputs
        X_km = lstsq(inputs_km,outputs_km);X_km = X_km[0]; X.append(X_km)
        
        # extract the states matrix and evaluate eigen values
        A_km = X_km[0:2,0:2]; A.append(A_km)
        
        eig_km = eig(A_km)
        eig_km = eig_km[0]
            
        # initialize storage array
        dx_fit = np.zeros((nt,noutputs))
        dx_val = np.zeros((nt,noutputs))
        
        # evaluate the DFSM and check its predictive capabilities
        for i in range(nt):
            
            # extract input values
            ip_fit = inputs_fit[i,:]
            ip_val = inputs_val[i,:]
            
            # reshape inputs
            ip_fit = ip_fit.reshape([1,len(ip_fit)])
            ip_val = ip_val.reshape([1,len(ip_val)])
            
            # evaluate surrogate model and find dx
            dx_fit[i,:] = np.squeeze(np.dot(ip_fit,X_km))
            dx_val[i,:] = np.squeeze(np.dot(ip_val,X_km))
            
        # plot DFSM predicted values
        
        # fitting trajectory
        ax1.plot(time_,dx_fit[:,0],label = 'DFSM_linfit')
        ax2.plot(time_,dx_fit[:,1],label = 'DFSM_linfit')
        
        # validation trajectory
        ax3.plot(time_val,dx_val[:,0],label = 'DFSM_linfit')
        ax4.plot(time_val,dx_val[:,1],label = 'DFSM_linfit')
    
    # legend    
    ax2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax2.set_xlabel('Time [s]')
    ax4.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax4.set_xlabel('Time [s]')
    
    # evaluate the error between lin fit and original
    dx_error_fit = state_derivatives_ - dx_fit
    dx_error_val = state_derivatives_val - dx_val
    
    # combine inputs and outputs for the fit and validation tajectories
    inputs_combined = np.vstack([inputs_fit,inputs_val])
    dx_error_combined = np.vstack([dx_error_fit,dx_error_val])
    time_combined = np.vstack([time_,time_val])
    
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
    ax.vlines(dx_norm_mean,ymin = 0, ymax = 1000,color = 'k')
    ax.set_xlabel('norm(error)')
    ax.legend()
    
    # find the index of points with norm(error) grater than the mean
    error_ind = dx_norm > 0.8*dx_norm_mean
    ind1 = error_ind[0:nt]
    ind2 = error_ind[nt:]
    
    mk_size = 1
    
    # plot the points with large error
    ax1.plot(time_[ind1],dx_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    ax2.plot(time_[ind1],dx_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    
    ax3.plot(time_val[ind2],dx_val[ind2,0],'.',color = 'k',markersize = mk_size)
    ax4.plot(time_val[ind2],dx_val[ind2,1],'.',color = 'k',markersize = mk_size)
    
    # extract the points with large error
    inputs_error = np.copy(inputs_combined)
    dx_error = np.copy(dx_error_combined) 
    dx_error[~error_ind,:] = 0
    dx_error[:,1] = 0

    input_names = validation_samples.state_names + validation_samples.control_names
    
    # for the given set of errors determine suitable clusters
    ncluster = np.sum(error_ind)
    inputs_err,outputs_err,nI,nO = perform_Kmeans(inputs_error,dx_error,ncluster)
    
    # split training and testing data
    train_ip,test_ip,train_dx_err,test_dx_err = train_test_split(inputs_err,outputs_err,test_size = 0.1)
    
    # scaler
    sc = StandardScaler()
    
    # scale input data
    scaler_input = sc.fit(train_ip)
    train_ip_scaled = scaler_input.transform(train_ip)
    test_ip_scaled = scaler_input.transform(test_ip)
    
    # train a neural network to predict the errors
    mlp_reg = MLPRegressor(hidden_layer_sizes = (200,100,50),max_iter = 300,activation = 'relu',solver = 'adam',verbose = True,tol = 1e-5)
    mlp_reg.fit(train_ip_scaled,train_dx_err)
    
    # predict the test values, and check the errors
    dx_err_predicted = mlp_reg.predict(test_ip_scaled)
    print(dx_err_predicted-test_dx_err)
    
    
    # initialize storage arrays
    dx_fit_corrected = np.zeros((nt,noutputs))
    dx_val_corrected = np.zeros((nt,noutputs))
    
    # loop through and evaluate 
    for i in range(nt):
        
        ip_fit = inputs_fit[i,:]
        ip_val = inputs_val[i,:]
        
        ip_fit = ip_fit.reshape([1,len(ip_fit)])
        ip_fit_scaled = scaler_input.transform(ip_fit)
        ip_val = ip_val.reshape([1,len(ip_val)])
        ip_val_scaled = scaler_input.transform(ip_val)
        
        #dx_eval_ed[i,:] = np.squeeze(np.dot(ip,X_ed))
        dx_fit_corrected[i,:] = np.squeeze(np.dot(ip_fit,X_km)+mlp_reg.predict(ip_fit_scaled))
        dx_val_corrected[i,:] = np.squeeze(np.dot(ip_val,X_km)+mlp_reg.predict(ip_val_scaled))
    
    # plot
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
        
    ax1.plot(time_,state_derivatives_[:,0],label = 'OF')
    ax1.plot(time_,dx_fit[:,0],label = 'DFSM_linfit')
    ax1.plot(time_,dx_fit_corrected[:,0],label = 'DFSM + NN')
    ax1.set_ylabel('dx1')
    ax1.set_xlim([time_[0],time_[-1]])
    ax1.set_title('Fitting Trajectory')
    
    ax2.plot(time_,state_derivatives_[:,1],label = 'OF')
    ax2.plot(time_,dx_fit[:,1],label = 'DFSM_linfit')
    ax2.plot(time_,dx_fit_corrected[:,1],label = 'DFSM + NN')
    ax2.set_ylabel('dx2')
    ax2.set_xlim([time_[0],time_[-1]])
    
    fig,(ax3,ax4) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
        
    ax3.plot(time_val,state_derivatives_val[:,0],label = 'OF')
    ax3.plot(time_val,dx_val[:,0],label = 'DFSM_linfit')
    ax3.plot(time_val,dx_val_corrected[:,0],label = 'DFSM + NN')
    ax3.set_ylabel('dx1')
    ax3.set_xlim([time_[0],time_[-1]])
    ax3.set_title('Validation Trajectory')
    
    ax4.plot(time_val,state_derivatives_val[:,1],label = 'OF')
    ax4.plot(time_val,dx_val[:,1],label = 'DFSM_linfit')
    ax4.plot(time_val,dx_val_corrected[:,1],label = 'DFSM + NN')
    ax4.set_ylabel('dx2')
    ax4.set_xlim([time_[0],time_[-1]])
    
    # plot the points with large error
    ax1.plot(time_[ind1],dx_fit[ind1,0],'.',color = 'k',markersize = mk_size)
    ax2.plot(time_[ind1],dx_fit[ind1,1],'.',color = 'k',markersize = mk_size)
    
    ax3.plot(time_val[ind2],dx_val[ind2,0],'.',color = 'k',markersize = mk_size)
    ax4.plot(time_val[ind2],dx_val[ind2,1],'.',color = 'k',markersize = mk_size)
    
    #ax1.plot(time_,dx_fit_corrected[:,0],label = 'corrected')

    ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax2.set_xlabel('Time [s]')
    ax4.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center');ax4.set_xlabel('Time [s]')
    
    # for i_ip in range(ninputs):
    #     fig,ax = plt.subplots(1)
    #     ax.hist(inputs_km[:,i_ip],label = 'Fit',alpha = 0.7)
    #     ax.hist(inputs_err[:,i_ip],label = 'Error',alpha = 0.7)
    #     ax.set_xlabel(input_names[i_ip])
    #     ax.legend()
        
        
    
    
    
    
    
    
    
        