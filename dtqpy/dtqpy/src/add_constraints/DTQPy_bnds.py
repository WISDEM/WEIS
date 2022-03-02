# -*- coding: utf-8 -*-
"""
DTQPy_bnds
Create simple bounds

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod

def DTQPy_getQPIndex(x,xtype,Flag,nt,I_stored):
    if (xtype ==1) or (xtype==2):
        I = I_stored[:,x-1]
        return I
    elif (xtype == 5) or (xtype == 7):
        I = I_stored[-1,x-1]
        return I
    elif (xtype == 0):
        I = 1
    else:
        I = I_stored[0,x-1]
        return I



def DTQPy_bnds(bnd,internal):
    nt = internal.nt; I_stored = internal.I_stored
    ny = internal.ny; IN = internal.IN
    #breakpoint()
    
    # will change with t_multiprod
    Bndt = DTQPy_tmultiprod(bnd.matrix,internal.auxdata,internal.t)
    #breakpoint()
    if (len(np.shape(Bndt))==3) and (np.shape(Bndt)[-1] == 1):
        Bndt = np.squeeze(Bndt,axis = -1)
        
    C = IN[bnd.right-1]
    right = bnd.right
    Isav = np.array([],dtype = 'int');Vsav = np.array([])
    #breakpoint()
    
    
    for k in range(len(C)):
        #print(k)
        if right in range(1,3):
            r = DTQPy_getQPIndex(C[k],right,1,nt,I_stored)
            
            if (len(np.shape(Bndt)) == 3):
                Vs = Bndt[:,:,k]
            else:
                Vs = Bndt[:,k]
                
            Isav = np.append(Isav,r)
            Vsav = np.append(Vsav,Vs)
            
        elif right in range(3,8):
            r = DTQPy_getQPIndex(C[k],right,0,nt,I_stored)
            #breakpoint()
            if (len(np.shape(Bndt)) == 3) :
                Vs = Bndt[0,:,k]
            else:
                Vs = Bndt[0,k]
                
            Isav = np.append(Isav,r)
            Vsav = np.append(Vsav,Vs)
            
            
    
    #breakpoint()
    return Isav,Vsav