# -*- coding: utf-8 -*-
"""
DTQPy_multiphase
Solve the LQDO problem without mesh refinement

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from dtqpy.src.DTQPy_create import DTQPy_create
from dtqpy.src.solver.DTQPy_SOLVER import DTQPy_SOLVER
from dtqpy.src.DTQPy_scalingLinear import DTQPy_scalingLinear

def DTQPy_multiphase(setup,opts):
    
    H,f,c,A,b,Aeq,beq,lb,ub,setup,internal,opts = DTQPy_create(setup,opts)
    
    if len(setup.Scaling) != 0:
        Scaleflag = True
        H,f,c,A,b,Aeq,beq,lb,ub,internal,SM,SC = DTQPy_scalingLinear(H,f,c,A,b,Aeq,beq,lb,ub,internal,setup.Scaling)
    else:
        Scaleflag = False
    
        
    
    
    [X,F,internal,opts] = DTQPy_SOLVER(H,f,A,b,Aeq,beq,lb,ub,internal,opts)
    
    
    if Scaleflag :
        
        X = X.reshape(-1,1)
        X = X*(SM) + SC
    
    if F == None:
        raise Exception("The optimization problem was not solved. Check log for more details")
    else:
        F = F+c
    
    nt = internal.nt;nu = internal.nu; ny = internal.ny; npl = internal.npl
    
    T = internal.t
    U = np.reshape(X[np.arange(0,nu*nt)],(nt,nu),order = 'F')
    Y = np.reshape(X[np.arange(nu*nt,(ny+nu)*nt)],(nt,ny),order = 'F')
    P = np.reshape(X[np.arange((nu+ny)*nt,(nu+ny)*nt+npl)],(npl,1),order = 'F')
    
    return T,U,Y,P,F,internal,opts