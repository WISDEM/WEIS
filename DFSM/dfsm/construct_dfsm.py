import os
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from numpy.linalg import lstsq,qr,inv,norm
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk
from sklearn.gaussian_process.kernels import ExpSineSquared
import time as timer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dfsm.dfsm_sample_data import sample_data
from sklearn.neural_network import MLPRegressor
from mat4py import loadmat
from dfsm.wrapper_LTI import wrapper_LTI
from pyoptsparse import IPOPT, Optimization,NSGA2,SLSQP
import argparse


class DFSM:
    
    def __init__(self,SimulationDetails,L_type = 'LTI',N_type = 'GPR',n_samples = 300,sampling_method = 'KM',train_split = 0.8):
        
        self.L_type = L_type
        self.N_type = N_type 
        self.n_samples = n_samples
        self.sampling_method = sampling_method 
        self.train_split = train_split
        
        FAST_sim = SimulationDetails.FAST_sim
        
        self.n_model_inputs = SimulationDetails.n_model_inputs
        self.n_deriv = SimulationDetails.n_deriv
        self.n_outputs = SimulationDetails.n_outputs
        self.gen_speed_ind = SimulationDetails.gen_speed_ind
        self.FA_Acc_ind = SimulationDetails.FA_Acc_ind
        self.NacIMU_FA_Acc_ind = SimulationDetails.NacIMU_FA_Acc_ind
        self.linear_model_file = SimulationDetails.linear_model_file
        self.region = SimulationDetails.region
        
        n_sim = SimulationDetails.n_sim
        
        train_index = int(np.floor(train_split*n_sim))
        self.train_data = FAST_sim[0:train_index]
        
        self.test_data = FAST_sim[train_index:]
        
    def construct_nonlinear(self,inputs,outputs,N_type,error_ind,n_inputs,n_outputs,ftype = 'deriv',scaling = True):
        
        sm = []
        
        if N_type == 'GPR':
            
            
            if scaling:
                # scale inputs
                scaler = StandardScaler().fit(inputs)
                inputs_scaled = scaler.transform(inputs)

                self.scaler = scaler
                
                
            else:
                
                inputs_scaled = inputs
                self.scaler = None
                
            # store inputs    
            self.inputs_scaled = inputs_scaled
            
            
            for ifunc in range(n_outputs):
                
                if error_ind[ifunc]:
                    
                    # train a gaussian process model
                    kernel = 1*RBFsk(length_scale = [1]*n_inputs,length_scale_bounds=(1e-5, 1e5))
                    smi = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 5,n_targets = 1, optimizer = 'fmin_l_bfgs_b')
                    smi.fit(inputs_scaled,outputs[:,ifunc])
                    
                    sm.append(smi)
                    
                else:
                    sm.append(None)
            
        elif N_type == 'NN':
            
            
            # train a neural network to predict the errors
            sm = MLPRegressor(hidden_layer_sizes = (100,50,10),max_iter = 1000,activation = 'tanh',solver = 'adam',verbose = True,tol = 1e-4)
            sm.fit(inputs,outputs[:,error_ind])
            
        if ftype == 'deriv':
            
            self.nonlin_deriv = sm
            
        elif ftype == 'outputs':
            
            self.nonlin_outputs = sm
            
       
    def construct_surrogate(self):
        
        # extract samples
        t1 = timer.time()
        inputs_sampled,dx_sampled,outputs_sampled,model_inputs,state_derivatives,outputs = sample_data(self.train_data,self.sampling_method,self.n_samples,grouping = 'together')
        t2 = timer.time()
        
        self.sampling_time = t2 - t1
        
        # depending on the type of L and N construct the surrogate model
        if self.L_type == None:
            
            # set the AB and CD matrices as empty
            self.AB = []
            self.CD = []
            self.lin_construct = 0
            
            self.error_ind_deriv = np.full(self.n_deriv,True)
            
            if self.n_outputs > 0:
                self.error_ind_outputs = np.full(self.n_outputs,True)
            
            t1 = timer.time()
            self.construct_nonlinear(inputs_sampled,dx_sampled,self.N_type,self.error_ind_deriv,self.n_model_inputs,self.n_deriv,'deriv')
            
            if self.n_outputs > 0:
                self.construct_nonlinear(inputs_sampled,outputs_sampled,self.N_type,self.error_ind_outputs,self.n_model_inputs,self.n_outputs,'outputs')
            
            t2 = timer.time()
            
            self.nonlin_construct_time = t2-t1
            self.linear_construct_time = 0
            self.inputs_sampled = inputs_sampled
            self.dx_sampled = dx_sampled
            
        else:
            
            if self.L_type == 'LTI':
                
                if self.linear_model_file == None:
                    
                    n_states = self.n_deriv 
                    n_controls = self.n_model_inputs - n_states
                    
                    # start timer
                    t1 = timer.time()
                    
                    # construct ta least squares approximation
                    AB = lstsq(model_inputs,state_derivatives,rcond = -1)
                    AB = AB[0]
                    
                    # extract the A matrix
                    AB_ = AB.T
                    
                    A = AB_[:,n_controls:]
                    Aeig = np.linalg.eig(A)[0]
                    Aeig_real = Aeig.real
                    
                    if all(Aeig_real < 0):
                        print('')
                        print('The least squares estimate is stable, using this as the linear model')
                        print('')
                    else:
                        print('')
                        print('The linear model identified using least-squares estimation is unstable, using grey-box estimation to identify a stable model')
                        print('')
                        
                        # extract the identified parameters
                        par0 = AB_[int(n_states/2):,:]
                        
                        # reshape
                        par0 = np.squeeze(par0.reshape([-1,1],order = 'F'))
                    
                        # get the length and specify upper and lower bounds
                        nx = len(par0)
                        lb = [-100]*nx
                        ub = [100]*nx
                        
                        # instantiate wrapper class
                        LTI = wrapper_LTI(inputs = model_inputs,outputs = state_derivatives,nstates = int(n_states),ncontrols = int(n_controls))
                    
                        # # set default options
                        parser = argparse.ArgumentParser()
                        parser.add_argument("--opt",help = "optimizer",type = str)
                        args = parser.parse_args()
                        
                        # set specific solver options
                        optOptions_ipopt = {'max_iter':500,'tol':1e-8,'print_level':1,
                                    'file_print_level':5,'dual_inf_tol':float(1e-8),'linear_solver':'mumps'}
                        
                        # options for nsga
                        optOptions_nsga = {'maxGen':200,'seed':34534,'PopSize':200}
                        
                        # options for SLSQP
                        optOptions_slsqp = {'ACC':float(1e-6),'MAXIT':500,'IPRINT':1}
                        
                        # optimization problem
                        prob = wrapper_LTI(inputs = model_inputs,outputs = state_derivatives,
                                            nstates = n_states,ncontrols = n_controls)
                        
                        # add relevant fields
                        optProb = Optimization('LTI',prob.objective_function)
                        optProb.addObj('obj')
                        optProb.addConGroup('con',n_states,lower = [None]*n_states,upper = -0*np.ones((n_states,)))  
                        optProb.addVarGroup("xvars", nx, lower=lb, upper=ub, value=par0)
                        
                        hybrid_flag = True
                        
                        if hybrid_flag:
                            print('')
                            print('Using a hybrid-optimization approach to estimate the linear model parameters')
                            print('Using the NSGA2 algorithm to get a good starting point')
                            print('')
                            # solve the problem using NSGA2 algorithm
                            opt = NSGA2(options = optOptions_nsga)
                            sol = opt(optProb)
                            
                            # extract the solution
                            X_nsga_dict = sol.xStar
                            X_nsga = X_nsga_dict['xvars']
                            F_ga = sol.fStar
                            self.X_nsga = X_nsga
                            self.X_nsga_dict = X_nsga_dict
                        else:
                            X_nsga = par0
                            self.X_nsga = X_nsga
                            self.X_nsga_dict = {'xvars':X_nsga}
                              
                        
                        grad_flag = True
                        solver = 'SLSQP'
                        # Hack for pyoptsparse segfault
                        
                        if grad_flag:
                            
                            print('')
                            print('Using a gradient-based algorithm to estimate the model parameters')
                            print('')
                            
                            # add problem elements for IPOPT
                            optProb2 = Optimization('LTI',prob.objective_function)
                            optProb2.addObj('obj')
                            optProb2.addConGroup('con',n_states,lower = [None]*n_states,upper = -0*np.ones((n_states,)))  
                            optProb2.addVarGroup("xvars", nx, lower=lb, upper=ub, value=X_nsga)
                            
                            if solver == 'SLSQP':
                                opt_ipopt = SLSQP(options = optOptions_slsqp)
                                sol_ipopt = opt_ipopt(optProb2,sens = 'FD')
                                
                            elif solver == 'IPOPT':
                                # solve
                                opt_ipopt = IPOPT(args,options = optOptions_ipopt)
                                sol_ipopt = opt_ipopt(optProb2,sens = 'FD')
                        
                    
                            # extract ipopt solution
                            F_ipopt = sol_ipopt.fStar
                            
                            X_ipopt_dict = sol_ipopt.xStar
                            X_ipopt = X_ipopt_dict['xvars']
                            self.X_ipopt = X_ipopt
                            self.X_ipopt_dict = X_ipopt_dict
                            
                        else:
                            X_ipopt = X_nsga
                            self.X_ipopt = X_ipopt
                            self.X_ipopt_dict = {'xvars':X_ipopt}
                            
                            
                        # linear model
                        A,B = LTI.linear_model(X_ipopt)
                        func_ip,sens = LTI.objective_function(self.X_ipopt_dict)
                        
                        AB = np.hstack([B,A]).T
                        
                else:
                    print('')
                    print('Loading linear model from mat file')
                    print('')
                    
                    AB_dict = loadmat(self.linear_model_file)
                    
                    if self.region == 'BR':
                        AB = AB_dict['AB_br']
                        
                    elif self.region == 'TR':
                        AB = AB_dict['AB_tr']
                        
                    elif self.region == 'R':
                        AB = AB_dict['AB_r']
                    
                    AB = np.array(AB)
                
                self.AB = AB

                if self.n_outputs > 0:
                    CD = lstsq(model_inputs,outputs,rcond = -1)
                    
                    self.CD = CD[0]
                    
                else:
                    
                    self.CD = []
                
                # end timer
                t2 = timer.time()
                
                # training time
                self.linear_construct_time = t2-t1
                self.inputs_sampled = inputs_sampled
                self.dx_sampled = dx_sampled
                self.model_inputs = model_inputs
                self.state_derivatives = state_derivatives
                
                
                # evaluate the error between the linear model and actual derivatives
                dx_error = dx_sampled - np.dot(inputs_sampled,self.AB)
                
                # evaluate mean
                error_mean = np.mean(dx_error,0)
                
                # find indices with error > 1e-5
                error_ind_deriv = np.array((np.abs(error_mean) > 1e-5))
                
                # store indices and error
                self.error_ind_deriv = error_ind_deriv
                self.dx_error = dx_error
                
                
                if self.n_outputs > 0:
                    
                    # error between outputs
                    outputs_error = outputs_sampled - np.dot(inputs_sampled,self.CD)
                    
                    # store indices
                    self.error_ind_outputs = (np.abs(np.mean(outputs_error,0)) > 1e-5)
                                                    
            
            if self.N_type == None:
                
                # if no nonlinear corrective function is specified, do nothing
                self.nonlin_deriv = None
                self.nonlin_outputs = None
                self.nonlin_construct_time = 0
                
            else:
                print('')
                print('Constructing nonlinear corrective function')
                print('')
                
                t1 = timer.time()

                self.construct_nonlinear(inputs_sampled,dx_error,self.N_type,self.error_ind_deriv,self.n_model_inputs,self.n_deriv,'deriv')
                
                if self.n_outputs > 0:
                    self.construct_nonlinear(inputs_sampled,outputs_error,self.N_type,self.error_ind_outputs,self.n_model_inputs,self.n_outputs,'outputs')
                
                t2 = timer.time()
                
                self.nonlin_construct_time = t2-t1
                
        
