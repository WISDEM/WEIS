import os
import numpy as np
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension,calculate_time
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from numpy.linalg import lstsq as lstsq_np
from scipy.linalg import lstsq as lstsq_sp
from dfsm.dfsm_sample_data import sample_data
import matplotlib.pyplot as plt
import time as timer
from numpy.linalg import eigvals,matrix_rank,cond,det,norm
from dfsm.wrapper_LTI import wrapper_LTI

from pyoptsparse import IPOPT, Optimization,NSGA2,SLSQP
import argparse


       

if __name__ == '__main__':
    
     # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    
    # datapath
    datapath = mydir + os.sep + 'outputs' + os.sep + 'MHK_TR_10' #+ os.sep + 'openfast_runs/rank_0'
    
    # get the entire path
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['PtfmPitch','GenSpeed','YawBrTAxp']
    
    state_props = {'units' : ['[deg]','[rpm]','[m/s2]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = [] #['YawBrTAxp', 'NcIMURAys', 'GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,100,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False,False,False],
                   'states_filter_type': [['filtfilt'],['filtfilt'],['filtfilt']],
                   'states_filter_tf': [[0.5],[1],[0.5]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [0,0,0],
                   'outputs_filter_flag': []
                   }
    # instantiate class
    sim_detail_nf = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin = 10,add_dx2 = True)

    # load and process data
    sim_detail_nf.load_openfast_sim()
    
    # extract data
    FAST_sim_nf = sim_detail_nf.FAST_sim
    
    
    sampling = ['together']
    sampling_type = 'KM'
            
     
    #---------------------------------------
    inputs_sampled,dx_sampled,outputs_sampled,model_inputs_nf,state_derivatives_nf,outputs = sample_data(FAST_sim_nf[:4],sampling_type,2)
    
    # get the number of states and controls
    nstates = 2*len(reqd_states)
    ncontrols = len(reqd_controls)
    
    # solve the least squares problem
    # instantiate DFSM model and construct surrogate
    dfsm_model = DFSM(sim_detail_nf,n_samples = 5,L_type = 'LTI',N_type = None,train_split = 0.4)
    dfsm_model.construct_surrogate()
    
    # AB_ls = lstsq_np(model_inputs_nf,state_derivatives_nf,rcond = -1)
    # AB_ls = AB_ls[0]
    # AB_ls = AB_ls.T
    
    # # extract the identified parameters
    # par0 = AB_ls[int(nstates/2):,:]
    
    # # reshape
    # par0 = np.squeeze(par0.reshape([-1,1],order = 'F'))

    # # instantiate wrapper class
    LTI = wrapper_LTI(inputs = model_inputs_nf,outputs = state_derivatives_nf,nstates = int(nstates),ncontrols = int(ncontrols))
    
    # x = {'xvars':par0}
    func_ga,sens = LTI.objective_function(dfsm_model.X_nsga_dict)
    func_ip,sens = LTI.objective_function(dfsm_model.X_ipopt_dict)
    
    # # # set default options
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--opt",help = "optimizer",type = str)
    # args = parser.parse_args()
    
    # # set specific solver options
    # optOptions_ipopt = {'max_iter':500,'tol':1e-8,'print_level':1,
    #             'file_print_level':5,'dual_inf_tol':float(1e-8),'linear_solver':'ma27'}
    
    # # options for nsga
    # optOptions_nsga = {'maxGen':100,'seed':34534}
    
    # # options for SLSQP
    # optOptions_slsqp = {'ACC':float(1e-6),'MAXIT':500,'IPRINT':1}
    
    # # optimization problem
    # prob = wrapper_LTI(inputs = model_inputs_nf,outputs = state_derivatives_nf,
    #                     nstates = nstates,ncontrols = ncontrols)
    
    # # add relevant fields
    # optProb = Optimization('LTI',prob.objective_function)
    # optProb.addObj('obj')
    # optProb.addConGroup('con',nstates,lower = [None]*nstates,upper = -0*np.ones((nstates,)))  
    # optProb.addVarGroup("xvars", nx, lower=lb, upper=ub, value=par0)
    
    # hybrid_flag = False
    
    # if hybrid_flag:
    #     # solve the problem using NSGA2 algorithm
    #     opt = NSGA2(options = optOptions_nsga)
    #     sol = opt(optProb)
        
    #     # extract the solution
    #     X_nsga_dict = sol.xStar
    #     X_nsga = X_nsga_dict['xvars']
    #     F_ga = sol.fStar
        
    # else:
    #     X_nsga = par0
    
    
                  
    
    # flag = True ; solver = 'SLSQP'
    # # Hack for pyoptsparse segfault
    
    # if flag:
        
    #     # add problem elements for IPOPT
    #     optProb2 = Optimization('LTI',prob.objective_function)
    #     optProb2.addObj('obj')
    #     optProb2.addConGroup('con',nstates,lower = [None]*nstates,upper = -0*np.ones((nstates,)))  
    #     optProb2.addVarGroup("xvars", nx, lower=lb, upper=ub, value=X_nsga)
        
    #     if solver == 'SLSQP':
    #         opt_ipopt = SLSQP(options = optOptions_slsqp)
    #         sol_ipopt = opt_ipopt(optProb2,sens = 'FD')
            
    #     elif solver == 'IPOPT':
    #         # solve
    #         opt_ipopt = IPOPT(args,options = optOptions_ipopt)
    #         sol_ipopt = opt_ipopt(optProb2,sens = 'FD')
    

    #     # extract ipopt solution
    #     F_ipopt = sol_ipopt.fStar
        
    #     X_ipopt_dict = sol_ipopt.xStar
    #     X_ipopt = X_ipopt_dict['xvars']
        
    #     # linear model
    #     A,B = LTI.linear_model(X_ipopt)
    #     func_ip,sens = LTI.objective_function(X_ipopt_dict)
        
    test_flag = True

    if test_flag:

        # extract the linear model identified using NSGA2
       
        #dfsm_model.AB = np.hstack([B,A]).T
        
         # extract test data
        test_data = dfsm_model.test_data
        
        # index of test data
        test_ind = [0]
        
        # flags related to testing
        simulation_flag = True 
        outputs_flag = (len(reqd_outputs) > 0)
        plot_flag = True
    
        # test dfsm
        dfsm,U_list,X_list,dx_list,Y_list = test_dfsm(dfsm_model,test_data,test_ind,simulation_flag,plot_flag)
        
        # Add state properties to list
        X_list_ = [dicti.update(state_props) for dicti in X_list]
        Y_list_ = [dicti.update(output_props) for dicti in Y_list]
        
        # plot dfsm
        plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag,save_flag = True)
        
        for ns in range(nstates):
            
            dx_comp = dx_list[0]
            
            fig,ax = plt.subplots(1)
            
            ax.plot(dx_comp['time'],dx_comp['OpenFAST'][:,ns],label = 'OpenFAST',color = 'r',linestyle = '--')
            ax.plot(dx_comp['time'],dx_comp['DFSM'][:,ns],label = 'DFSM',color = 'k')
            
            ax.set_title(dx_comp['names'][ns])
            ax.legend(ncol = 2)
            ax.set_xlim([dx_comp['time'][0],dx_comp['time'][-1]])
        
            
            
            
            
            
   
    

