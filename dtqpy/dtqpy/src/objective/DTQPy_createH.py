# -*- coding: utf-8 -*-
"""
DTQPy_createH
Create matrix for the quadratic objective term

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import inbuilt libraries
import numpy as np
from scipy.sparse import csc_matrix

# import DTQPy specific functions
from dtqpy.src.objective.DTQPy_L import DTQPy_L
from dtqpy.src.objective.DTQPy_M import DTQPy_M

def DTQPy_createH(L,M,internal,opts):
    
    # initialize
    HI = np.array([]);HJ = np.array([]);HV = np.array([])
    
    # Lagrange terms
    if len(L) !=0:
        I,J,V = DTQPy_L(L,internal,opts)
        HI = np.append(HI,I); HJ = np.append(HJ,J); HV = np.append(HV,V)
        
    # Mayer terms    
    if len(M) !=0:
        I,J,V = DTQPy_M(M,internal,opts)
        HI = np.append(HI,I); HJ = np.append(HJ,J); HV = np.append(HV,V)
    
    # create sparse matrix
    #breakpoint()
    if HV.any() == False:
        H = csc_matrix((internal.nx,internal.nx)) # no hessian
    else:
        H = csc_matrix((HV,(HI,HJ)),shape = (internal.nx,internal.nx))
        H = H+H.T # make symmetric, then times 2 for 1/2*x'*H*x
    #breakpoint() 
    return H