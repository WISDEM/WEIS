# -*- coding: utf-8 -*-
"""
DTQPy_createc
Create matrix for the constant objective term

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import inbuilt libraries
import numpy as np
from scipy.sparse import csc_matrix

# import DTQPy specific functions
from dtqpy.src.objective.DTQPy_L import DTQPy_L
from dtqpy.src.objective.DTQPy_M import DTQPy_M

def DTQPy_createc(Lconstant,Mconstant,internal,opts):
    
    # initialize
    c = 0
    
    # Lagrange terms
    if len(Lconstant) !=0:
        null,null,V = DTQPy_L(Lconstant,internal,opts)
        c = np.sum(V)
        
    # Mayer terms    
    if len(Mconstant) !=0:
        for idx in range(len(Mconstant)):
            c+= Mconstant[idx].matrix

    return c
