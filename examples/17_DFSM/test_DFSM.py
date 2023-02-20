import os
import numpy as np
import matplotlib.pyplot as plt
from DFSM import SimulationDetails,DFSM,reduce_samples
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq,eig,norm
from scipy.interpolate import PchipInterpolator,CubicSpline
from scipy.integrate import solve_ivp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk

from create_plots import plot_inputs,plot_simulation_results


def vanderpol(t,x,u_fun,a):
    
    # control function
    u = u_fun(t)
    
    # extract states
    x0 = x[0];x1 = x[1]
    
    # ecaluate surrogate model
    #dx = np.array([x1+np.sin(x1)*np.cos(x0),((1-(x0)**2))*x0*x1+u])
    
    dx = np.array([-x0+x1**2,x0-x1-x0**2+u])
    
    dx = np.array([-0.84*x0-x1-x0*x1,x0+x1+x0*x1+u])

    return dx

def vanderpol_dfsm(t,x,u_fun,LM,corr_fun):
    
    # control function
    u = u_fun(t)
    
    inputs = np.hstack([x,u])
    inputs = inputs.reshape([1,len(inputs)])
    
    dx = np.dot(inputs,LM) + corr_fun.predict(inputs)
    
    dx = np.squeeze(dx)
    
    return dx
    

#def vanderpol_dfsm()
def evaluate_dx(time,states):
    
    states_pp = CubicSpline(time,states)
    
    #
    dx_pp = states_pp.derivative
        
    # evaluate first time derivative
    dx_pp1 = dx_pp(nu = 1)
    
    state_derivatives = dx_pp1(time)
    
    return state_derivatives
    
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
def train_corrective_function(ncluster,inputs_error,dx_error,fun_type = 'GPR'):
    
    # for the given set of errors determine suitable clusters
    inputs_err,outputs_err,nI,nO = perform_Kmeans(inputs_error,dx_error,ncluster)
    
    # split training and testing data
    train_ip = inputs_err
    train_dx_err = outputs_err

    
    if fun_type == 'NN':
        
        # train a neural network to predict the errors
        corr_fun = MLPRegressor(hidden_layer_sizes = (50,10),max_iter = 300,activation = 'logistic',solver = 'adam',verbose = True,tol = 1e-5)
        corr_fun.fit(train_ip,train_dx_err)
        
    elif fun_type == 'GPR':
        
        # train a GPR
        kernel = 1*RBFsk(length_scale = [1]*ninputs,length_scale_bounds=(1e-2, 1e2))
        corr_fun = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 5,random_state = 34534)
        corr_fun.fit(train_ip,train_dx_err)
        
    #elif fun_type == 'RBF':
        
    return corr_fun,fun_type
#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # time span
    t0 = 0;tf = 5
    solve_options = {'method':'DOP853','rtol':1e-10,'atol':1e-10}
    
    
    nt = 5000
    
    t = np.linspace(t0,tf,nt)
    
    u_fit = 2+np.sin((2*np.pi)*np.arange(1,nt+1)/200) + 0.5*np.random.random(nt)
    u_fit_pp = PchipInterpolator(t,u_fit);u_fit_fun = lambda t:u_fit_pp(t)
    
    u_val = 2+np.cos((np.pi/15)*np.arange(1,nt+1)/100)+ 0.1*np.random.random(nt)
    u_val_pp = PchipInterpolator(t,u_val);u_val_fun = lambda t:u_val_pp(t)
    
    u_test = 2+np.cos((5*np.pi)*np.arange(1,nt+1)/500) #u_fit + u_val
    u_test_pp = PchipInterpolator(t,u_test);u_test_fun = lambda t:u_test_pp(t)
    
    fig,ax = plt.subplots(1)
    fig.suptitle('Controls')
    
    ax.plot(t,u_fit_fun(t),label = 'fit')
    ax.plot(t,u_val_fun(t),label = 'val')
    ax.plot(t,u_test_fun(t),label = 'test')
    ax.set_xlim([t0,tf])
    ax.set_ylabel('Inputs')
    ax.set_xlabel('Time [s]')
    ax.legend()

    t_eval = np.linspace(t0,tf,1000)
    Y0array = np.array([[2.4080,-0.7371],[1.6120,4.1154]])

    
    sol_fit = solve_ivp(vanderpol,[t0,tf],y0 = Y0array[0],method = solve_options['method'],args = (u_fit_fun,1),rtol = solve_options['rtol'],atol = solve_options['atol'])
    T_fit = sol_fit['t']
    X_fit = sol_fit['y'];X_fit = X_fit.T
    U_fit = u_fit_fun(T_fit);U_fit = np.reshape(U_fit,[len(T_fit),1])
    dx_fit = evaluate_dx(T_fit,X_fit)
    
    
    
    sol_val = solve_ivp(vanderpol,[t0,tf],y0 = Y0array[1],method = solve_options['method'],args = (u_val_fun,1),rtol = solve_options['rtol'],atol = solve_options['atol'])
    T_val = sol_val['t']
    X_val = sol_val['y'];X_val = X_val.T
    U_val = u_val_fun(T_val);U_val = np.reshape(U_val,[len(T_val),1])
    dx_val = evaluate_dx(T_val,X_val)
    
    # construct 
    
    inputs_fit = np.hstack([X_fit,U_fit])
    outputs_fit = dx_fit
    
    inputs_val = np.hstack([X_val,U_val])
    outputs_val = dx_val
    
    nclusters = 10
    
    inputs_km,outputs_km,ninputs,noutputs = perform_Kmeans(inputs_fit,outputs_fit,nclusters)
    
    LM = lstsq(inputs_km,outputs_km,rcond = None);LM = LM[0]
    A = LM[0:2,0:2]; eig_val = eig(A.T);eig_val = eig_val[0]
    
    print(eig_val)
    
    dx_fit_lm = np.dot(inputs_fit,LM)
    dx_val_lm = np.dot(inputs_val,LM)
    
    #--------------------------------------------------------------------------
    
    fig1,(ax1,ax2) = plt.subplots(2,1)
    fig1.subplots_adjust(hspace = 0.45)
    fig1.suptitle('State Derivatives')
        
    ax1.plot(T_fit,dx_fit[:,0],label = 'Actual')
    ax1.plot(T_fit,dx_fit_lm[:,0],label = 'LM')
    ax1.set_ylabel('dx1')
    ax1.set_xlim([t0,tf])
    
    ax2.plot(T_fit,dx_fit[:,1],label = 'Actual')
    ax2.plot(T_fit,dx_fit_lm[:,1],label = 'LM')
    ax2.set_ylabel('dx2')
    ax2.set_xlim([t0,tf])
    
    #--------------------------------------------------------------------------
    fig,(ax3,ax4) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    fig.suptitle('State Derivatives')
        
    ax3.plot(T_val,dx_val[:,0],label = 'Actual')
    ax3.plot(T_val,dx_val_lm[:,0],label = 'LM')
    ax3.set_ylabel('dx1')
    ax3.set_xlim([t0,tf])
    
    ax4.plot(T_val,dx_val[:,1],label = 'Actual')
    ax4.plot(T_val,dx_val_lm[:,1],label = 'LM')
    ax4.set_ylabel('dx2')
    ax4.set_xlim([t0,tf])
    
    dx_fit_error = dx_fit-dx_fit_lm
    dx_val_error = dx_val-dx_val_lm
    
    dx_error = np.vstack([dx_fit_error,dx_val_error])
    inputs_error = np.vstack([inputs_fit,inputs_val])
    
    
    #-------------------------------------------------------------------------
    nclusters = 50
    corr_fun,fun_type = train_corrective_function(nclusters,inputs_error,dx_error,fun_type = 'GPR')
    
    dx_dfsm_fit = dx_fit_lm + corr_fun.predict(inputs_fit)
    dx_dfsm_val = dx_val_lm + corr_fun.predict(inputs_val)
    
    ax1.plot(T_fit,dx_dfsm_fit[:,0],label = 'LM+Corr')
    ax2.plot(T_fit,dx_dfsm_fit[:,1],label = 'LM+Corr')
    
    ax3.plot(T_val,dx_dfsm_val[:,0],label = 'LM+Corr')
    ax4.plot(T_val,dx_dfsm_val[:,1],label = 'LM+Corr')
    
    #--------------------------------------------------------------------------
    
    sol_fit = solve_ivp(vanderpol_dfsm,[t0,tf],y0 = Y0array[0],method = solve_options['method'],args = (u_fit_fun,LM,corr_fun),rtol = solve_options['rtol'],atol = solve_options['atol'])
    T_fit2 = sol_fit['t']
    X_fit2 = sol_fit['y'];X_fit2 = X_fit2.T
    U_fit2 = u_fit_fun(T_fit2);U_fit2 = np.reshape(U_fit2,[len(T_fit2),1])

    
    
    
    sol_val = solve_ivp(vanderpol_dfsm,[t0,tf],y0 = Y0array[1],method = solve_options['method'],args = (u_val_fun,LM,corr_fun),rtol = solve_options['rtol'],atol = solve_options['atol'])
    T_val2 = sol_val['t']
    X_val2 = sol_val['y'];X_val2 = X_val2.T
    U_val2 = u_val_fun(T_val2);U_val2 = np.reshape(U_val2,[len(T_val2),1])

    
    #--------------------------------------------------------------------------
    
    fig1,(ax1,ax2) = plt.subplots(2,1)
    fig1.subplots_adjust(hspace = 0.45)
    fig1.suptitle('States')
        
    ax1.plot(T_fit,X_fit[:,0],label = 'Actual')
    ax1.plot(T_fit2,X_fit2[:,0],label = 'DFSM')
    ax1.set_ylabel('x1')
    ax1.set_xlim([t0,tf])
    
    ax2.plot(T_fit,X_fit[:,1],label = 'Actual')
    ax2.plot(T_fit2,X_fit2[:,1],label = 'DFSM')
    ax2.set_ylabel('x2')
    ax2.set_xlim([t0,tf])
    ax2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    
    #--------------------------------------------------------------------------
    fig,(ax3,ax4) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    fig.suptitle('States')
        
    ax3.plot(T_val,X_val[:,0],label = 'Actual')
    ax3.plot(T_val2,X_val2[:,0],label = 'DFSM')
    ax3.set_ylabel('x1')
    ax3.set_xlim([t0,tf])
    
    ax4.plot(T_val,X_val[:,1],label = 'Actual')
    ax4.plot(T_val2,X_val2[:,1],label = 'DFSM')
    ax4.set_ylabel('x2')
    ax4.set_xlim([t0,tf])
    ax4.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    
    #--------------------------------------------------------------------------
    sol_test_act = solve_ivp(vanderpol,[t0,tf],y0 = Y0array[0],method = solve_options['method'],args = (u_test_fun,0),rtol = solve_options['rtol'],atol = solve_options['atol'])
    
    T_test_act = sol_test_act['t']
    X_test_act = sol_test_act['y'];X_test_act = X_test_act.T
    
    sol_test_dfsm = solve_ivp(vanderpol_dfsm,[t0,tf],y0 = Y0array[0],method = solve_options['method'],args = (u_test_fun,LM,corr_fun),rtol = solve_options['rtol'],atol = solve_options['atol'])    
    
    T_test_dfsm = sol_test_dfsm['t']
    X_test_dfsm = sol_test_dfsm['y'];X_test_dfsm = X_test_dfsm.T
    
    fig1,(ax1,ax2) = plt.subplots(2,1)
    fig1.subplots_adjust(hspace = 0.45)
    fig1.suptitle('States')
        
    ax1.plot(T_test_act,X_test_act[:,0],label = 'Actual')
    ax1.plot(T_test_dfsm,X_test_dfsm[:,0],label = 'DFSM')
    ax1.set_ylabel('x1')
    ax1.set_xlim([t0,tf])
    
    ax2.plot(T_test_act,X_test_act[:,1],label = 'Actual')
    ax2.plot(T_test_dfsm,X_test_dfsm[:,1],label = 'DFSM')
    ax2.set_ylabel('x2')
    ax2.set_xlim([t0,tf])
    ax2.legend(ncol = 2,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
    