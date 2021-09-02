# -*- coding: utf-8 -*-
"""
DTQPy_create_bnds
Create matrix for the quadratic objective term

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from dtqpy.src.add_constraints.DTQPy_bnds import DTQPy_bnds

def DTQPy_create_bnds(LB,UB,internal):
    nx = internal.nx
    
    lb = np.ones((nx,len(LB)))
    lb *= -np.inf
    ub = np.ones((nx,len(UB)))
    ub *= np.inf
    
    for i in range(len(LB)):
        LB[i].Check_shape()
        I,V = DTQPy_bnds(LB[i],internal)
        #breakpoint()
        lb[I,i] = V
        
    lb = np.max(lb,axis=1)
    #breakpoint()
    
    for i in range(len(UB)):
        UB[i].Check_shape()
        I,V = DTQPy_bnds(UB[i],internal)
        #breakpoint()
        ub[I,i] = V
    
    ub = np.min(ub,axis=1)
    
    return lb,ub