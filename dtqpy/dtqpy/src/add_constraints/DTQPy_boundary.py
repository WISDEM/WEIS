# -*- coding: utf-8 -*-
"""
DTQPy_boundary
Create sequences for boundary constraint terms

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np

from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod
from dtqpy.src.DTQPy_getQPIndex import DTQPy_getQPIndex

def DTQPy_boundary(yz,internal):
    # extract some of the variables
    nt = internal.nt;IN = internal.IN;I_stored = internal.I_stored
    
    # initialize storage arrays
    Jsav = np.array([],dtype = int); Vsav = np.array([])
    
    # go through each sub structure
    for j in range(len(yz.linear)):
        
        # extract matrix
        yzt = yz.linear[j].matrix
        
        # variable locations for the variable type
        C = IN[yz.linear[j].right-1]
        #breakpoint()
        for i in range(len(C)):
            
            # single value assigned
            Vs = yzt[i,0]
            
            # check if entries are nonzero
            if Vs.any():
                
                Js = DTQPy_getQPIndex(C[i],yz.linear[j].right,0,nt,I_stored)
                
                # combine
                Jsav = np.append(Jsav,Js); Vsav = np.append(Vsav,Vs)
                
    # assign constant value
    b = yz.b
    #breakpoint()
    return Jsav,Vsav,b
        
        