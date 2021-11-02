# -*- coding: utf-8 -*-
"""
DTQPy_getQPIndex
Optimization variable index generating functions

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

from numpy.matlib import repmat

def DTQPy_getQPIndex(x,xtype,Flag,nt,I_stored):
    if (xtype ==1) or (xtype==2):
        I = I_stored[:,x-1]
        return I
    elif (xtype == 5) or (xtype == 7):
        I = I_stored[-1,x-1]
    elif (xtype == 0):
        I = 1
    else:
        I = I_stored[0,x-1]
        
    if Flag and (xtype!=1) and (xtype != 2):
        I = repmat(I,nt,1)
    #breakpoint()
    return I