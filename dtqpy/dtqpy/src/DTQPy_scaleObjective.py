#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_SOLVER_pyoptsparse
Reorganize matrices and solve the problem using IPOPT from pyoptsparse
Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np 
from numpy.linalg import norm
from math import log10,floor

def DTQPy_scaleObjective(H,f,nx):
    
    # initialize vector 
    X = np.ones((nx,))
    
    # find values of x'Hx and f'x
    Hx = floor(log10(np.abs(np.dot(X,H.dot(X)))))
    fx = floor(log10(np.abs((f.T).dot(X))))
    
    # find scaling value
    C = 11 #np.min([Hx,fx])
    
    # scale
    H = H/10**C
    
    f = f/10**C
    
    
    return H,f,C

