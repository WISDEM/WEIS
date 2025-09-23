import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline,interp1d
from weis.dfsm.evaluate_dfsm import evaluate_dfsm
import time as timer

def odefun(t,x,u_fun,DFSM):
    
    u = u_fun(t)
    
    # combine
    inputs = np.hstack([u,x])
    
    # evaluate dfsm
    dx = evaluate_dfsm(DFSM,inputs,'deriv')
    
    return dx
    


def test_dfsm(DFSM,test_cases,test_ind,simulation_flag = True,plot_flag = True, solver_options = None):
    
    n_cases = len(test_ind)
    
    U_list = []
    X_list = []
    dx_list = []
    Y_list = []
    simulation_time = np.zeros(n_cases)

    
    
    # loop through and evaluate test cases using the dfsm model
    for idx,ind in enumerate(test_ind):
        
        # extract test case
        case = test_cases[ind]
        
        # extract data
        time = case['time']
        controls = case['controls']
        states = case['states']
        state_derivatives = case['state_derivatives']
        outputs = case ['outputs']
        n_outputs = case['n_outputs']
        
        inputs = np.hstack([controls,states])
        
        t0 = time[0];tf = time[-1]
        tspan = [t0,tf]
        
        x0 = states[0,:]


        
        # create interpolating function for controls
        u_pp = CubicSpline(time,controls)
        u_fun = lambda t: u_pp(t)
        
        # solver method and options
        if solver_options == None:
            solver_options = {'method':'RK45','rtol':1e-6,'atol':1e-6}
        
        
        if simulation_flag:
            
            t1 = timer.time()
            sol =  solve_ivp(odefun,tspan,x0,method=solver_options['method'],args = (u_fun,DFSM),rtol = solver_options['rtol'],atol = solver_options['atol'])
            t2 = timer.time()
            
            # extract solution

            T = sol['t']
            states_dfsm = sol['y']
            states_dfsm = states_dfsm.T
            
            states_dfsm = interp1d(T,states_dfsm,axis = 0)(time)
            controls = u_fun(time)
            #state_derivatives = interp1d(time,state_derivatives,axis = 0)(T)
            
            # if n_outputs > 0:
            #     outputs = interp1d(time,outputs,axis = 0)(T)
                
            inputs_dfsm = np.hstack([controls,states_dfsm])
            inputs = np.hstack([controls,states])
            #time = T
        
            simulation_time[idx] = t2-t1
            
            X_dict = {'time':time,'names':case['state_names'],'n':case['n_states'],'OpenFAST':states,'DFSM':states_dfsm,'error':states - states_dfsm}
            U_dict = {'time': time, 'names': case['control_names'],'n': case['n_controls'],'OpenFAST':controls,'DFSM':controls}
            
            X_list.append(X_dict)
            U_list.append(U_dict)

        else:
            
            simulation_time = 0
            
        # evaluate state derivatives predicted by the DFSM
        fun_type = 'deriv'
        dx_dfsm = evaluate_dfsm(DFSM,inputs_dfsm,fun_type)
        
        dx_dict = {'time':time,'names':case['dx_names'],'n':case['n_deriv'],'OpenFAST':state_derivatives,'DFSM':dx_dfsm}
        
        dx_list.append(dx_dict)
        
        if n_outputs > 0 and simulation_flag:
            
            fun_type = 'outputs'
            outputs_dfsm = evaluate_dfsm(DFSM,inputs_dfsm,fun_type)
            
            Y_dict = {'time':time,'names':case['output_names'],'n':case['n_outputs'],'OpenFAST':outputs,'DFSM':outputs_dfsm,'error': outputs - outputs_dfsm}
            
            Y_list.append(Y_dict)
            
    DFSM.simulation_time = np.mean(simulation_time)
    
    return DFSM,U_list,X_list,dx_list,Y_list
            
            
            
            
            
        
            
        
        
    


