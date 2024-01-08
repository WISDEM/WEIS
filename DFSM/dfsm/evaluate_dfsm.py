import numpy as np


def evaluate_dfsm(DFSM,inputs,fun_type = 'deriv'):
    
    # get number of points
    n_points = len(np.shape(inputs))
    
    if n_points == 1:
        inputs = inputs.reshape(1,-1)
        nt = 1
    else:
        nt = np.shape(inputs)[0]
    
    # based on the function type extract the info
    if fun_type == 'deriv':
        
        # state derivative function
        lin = DFSM.AB
        nonlin = DFSM.nonlin_deriv
        no = DFSM.n_deriv
        error_ind = DFSM.error_ind_deriv
        
    elif fun_type == 'outputs':
        
        # output function
        lin = DFSM.CD
        nonlin = DFSM.nonlin_outputs
        no = DFSM.n_outputs
        error_ind = DFSM.error_ind_outputs
        
    if DFSM.L_type == None:
        
        dx_lin = np.zeros((nt,no))
        
    else:
        
        # if LTI model
        if DFSM.L_type == 'LTI':
            dx_lin = np.dot(inputs,lin)
    
    # initialize nonlinear part
    dx_nonlin = np.zeros((nt,no))
    
    # loop through and evaluate
    if not(DFSM.N_type == None):

        for io in range(no):
        
            if error_ind[io]:
                
                nonlin_func = nonlin[io]
                
                scaler = DFSM.scaler
                
                # scale
                if scaler == None:
                    inputs_ = inputs 
                else:
                    inputs_ = scaler.transform(inputs)
        
                # predict 
                nonlin_prediction = nonlin_func.predict(inputs_)
                
                # # reshape if the prediction is for a single point
                # if len(np.shape(nonlin_prediction)) == 1:
                #     nonlin_prediction = nonlin_prediction.reshape(-1,1)

                # add
                dx_nonlin[:,io] = dx_nonlin[:,io] + nonlin_prediction

        
    dx = dx_lin + dx_nonlin
    
    if n_points == 1:
        dx = np.squeeze(dx)
        
    return dx
            
        
        
    
            
    
    
    

