# -*- coding: utf-8 -*-
"""
DTQPy_M
Create sequences for Mayer terms

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import inbuilt libraries
import numpy as np
from numpy.matlib import repmat

# import DTQPy specific functions
from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod
from dtqpy.src.DTQPy_getQPIndex import DTQPy_getQPIndex

def DTQPy_M(Mfull,internal,opts):
    
    # extract variables
    nt = internal.nt; IN = internal.IN; I_stored = internal.I_stored
    
    # initialize storage arrays
    Isav = np.array([]); Jsav = np.array([]); Vsav = np.array([])
    
    # go through each Mayer term
    for k in range(len(Mfull)):
        
        # obtain current substructure
        Mleft = Mfull[k].left
        Mright = Mfull[k].right
        Mmatrix = Mfull[k].matrix
        
        # obtain matrix
        Mt = DTQPy_tmultiprod(Mmatrix,[],np.array([0]))
        
        if Mleft != 0:
            R = IN[Mleft-1]
        else:
            R = np.array([0])
            
        if Mright !=0:
            #breakpoint()
            C = IN[Mright-1]
        else:
            C = np.array([0])
            
        # determine locations and matrix values at these points
        for i in range(len(R)):
            for j in range(len(C)):
                
                # get current matrix value
                Mv = Mt[:,i,j]
                
                if Mv.any():
                    
                    # hessian index sequence
                    r = DTQPy_getQPIndex(R[i],Mleft,0,nt,I_stored)
                    c = DTQPy_getQPIndex(C[j],Mright,0,nt,I_stored)
                    
                    # assign
                    Isav = np.append(Isav,r)
                    Jsav = np.append(Jsav,c)
                    Vsav = np.append(Vsav,Mv)
                    
    return Isav,Jsav,Vsav    
    