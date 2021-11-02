# -*- coding: utf-8 -*-
"""
DTQPy_path
Create sequences for path constraint terms

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np

from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod
from dtqpy.src.DTQPy_getQPIndex import DTQPy_getQPIndex

def DTQPy_path(YZ,internal):
    
    # extract some of the variables
    nt = internal.nt; t = internal.t; IN = internal.IN; I_stored = internal.I_stored; auxdata = internal.auxdata
    
    # initiaalize storage arrays
    Isav = np.array([],dtype = int); Jsav = np.array([],dtype = int); Vsav = np.array([])
    
    # row loactions
    Ir = np.arange(0,nt)
    
    # go through each sub structure
    
    for j in range(len(YZ.linear)):
        
        # Find time dependnt matrix
        YZt = DTQPy_tmultiprod(YZ.linear[j].matrix,auxdata,t)
        
        if (len(np.shape(YZt)) == 3) and (np.shape(YZt)[-1]==1):
            YZt = np.squeeze(YZt,axis=-1)
        
        # variable locations for the varibale type
        C = IN[YZ.linear[j].right-1]
        
        for i in range(len(C)):
            
            # row loactions
            Is  = Ir
            
            # column locations
            Js = DTQPy_getQPIndex(C[i],YZ.linear[j].right,1,nt,I_stored)
            
            # nt values assigned
            if len(np.shape(YZt)) == 3:
                Vs = YZt[:,:,i]
            else:
                Vs = YZt[:,i]
                
            NZeroIndex = np.nonzero(Vs)
            Is = Is[NZeroIndex[0]]; Js = Js[NZeroIndex[0]]; Vs = Vs[NZeroIndex[0]]
            
            # combine
            Isav = np.append(Isav,Is);Jsav = np.append(Jsav,Js);Vsav = np.append(Vsav,Vs)
            
            
            
    b = DTQPy_tmultiprod(YZ.b,auxdata,t)
    #breakpoint()
    return Isav,Jsav,Vsav,b
            