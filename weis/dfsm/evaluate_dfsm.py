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

        elif DFSM.L_type == 'LPV':

            A_fun = DFSM.A_fun
            B_fun = DFSM.B_fun
            C_fun = DFSM.C_fun
            D_fun = DFSM.D_fun

            nu = DFSM.n_model_inputs - DFSM.n_deriv

            if nt == 1:

                inputs = np.squeeze(inputs)

                # extract inputs and outputs
                u = inputs[:nu]
                x = inputs[nu:]

                # extract wind speed
                w = u[0]
                
                # evaluate the LPV function to get the system matrices
                if fun_type == 'deriv':

                    # A and B matrices for the derivative function
                    A_ = A_fun(w)
                    B_ = B_fun(w)
                    

                    dx_lin = np.dot(A_,x) + np.dot(B_,u)

                elif fun_type == 'outputs':

                    # C and D matrices for outputs
                    C_ = C_fun(w)
                    D_ = D_fun(w)
                    
                    dx_lin = np.dot(C_,x) + np.dot(D_,u)
                
            elif nt > 1:

                # initialize
                dx_lin = np.zeros((nt,no))

                # extract inputs and outputs
                u = inputs[:,:nu]
                x = inputs[:,nu:]

                for i in range(nt):

                    w = u[i,0]
                    x_ = x[i,:]
                    u_ = u[i,:]

                    # evaluate the LPV function to get the system matrices
                    if fun_type == 'deriv':

                        # A and B matrices for the derivative function
                        A_ = A_fun(w)
                        B_ = B_fun(w)
                        

                        dx_lin[i,:] = np.dot(A_,x_) + np.dot(B_,u_)

                    elif fun_type == 'outputs':

                        # C and D matrices for outputs
                        C_ = C_fun(w)
                        D_ = D_fun(w)

                        dx_lin[i,:] = np.dot(C_,x_) + np.dot(D_,u_)
 
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
            
        
        
    
            
    
    
    

