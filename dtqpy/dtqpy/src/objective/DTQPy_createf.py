# -*- coding: utf-8 -*-
"""
DTQPy_createf
Create matrix for the linear objective term

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import inbuilt libraries
import numpy as np
from scipy.sparse import csc_matrix

# import DTQPy specific functions
from dtqpy.src.objective.DTQPy_L import DTQPy_L
from dtqpy.src.objective.DTQPy_M import DTQPy_M

def DTQPy_createf(Llinear,Mlinear,internal,opts):
    
    # initialize
    fJ = np.array([]);fV = np.array([])
    
    # Lagrange terms
    if len(Llinear) !=0:
        null,J,V = DTQPy_L(Llinear,internal,opts)
        fJ = np.append(fJ,J); fV = np.append(fV,V)
        
    # Mayer terms    
    if len(Mlinear) !=0:
        null,J,V = DTQPy_M(Mlinear,internal,opts)
        fJ = np.append(fJ,J); fV = np.append(fV,V)
    #breakpoint()
    # create sparse matrix
    if fV.any() == False:
        f = csc_matrix((internal.nx,1)) # no hessian
    else:
        fY = np.zeros((len(fJ),))
        f = csc_matrix((fV,(fJ,fY)),shape = (internal.nx,1))
        
    #breakpoint()   
    return f