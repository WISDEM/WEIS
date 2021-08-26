# -*- coding: utf-8 -*-
"""
DTQPy_L
Create sequences for lagrange term


Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import built in libraries
import numpy as np
from numpy.matlib import repmat

# import DTQPy specific functions
from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod
from dtqpy.src.DTQPy_getQPIndex import DTQPy_getQPIndex

def DTQPy_L(Lfull,internal,opts):
    
    # extract some variables
    nt = internal.nt; t = internal.t; tm = internal.tm; h = internal.h; w = internal.w;
    auxdata = internal.auxdata; IN = internal.IN; Istored = internal.I_stored;
    quadrature = opts.dt.quadrature; 
    
    # check for off-diagonal terms
    OffFlag = (quadrature == 'CQHS')
    
    # initialize storage arrays
    Isav = np.array([]); Jsav = np.array([]); HWsav = np.array([]); Qsav = np.array([])
    
    # augumented step size vector
    h0 = np.append(h,np.array([0]))
    
    # go through each Lagrange term
    for k in range(len(Lfull)):
        
        # obtain current substructure
        Lleft = Lfull[k].left
        Lright = Lfull[k].right
        Lmatrix = Lfull[k].matrix
        
        # find time dependednt 
        Lt = DTQPy_tmultiprod(Lmatrix,auxdata,t)
        
        # to do include offlag
        #breakpoint()
        
        # check if both left and right fields are equal to 0
        if Lleft != 0:
            R = IN[Lleft-1]
        else:
            R = np.array([0])
            
        if Lright !=0:
            C = IN[Lright-1]
        else:
            C = np.array([0])
            
        # determine loactions and matrix values at these points
        for i in range(len(R)):
            for j in range(len(C)):
                #breakpoint()
                # get current matrix values
                Lv = Lt[:,i,j]
                
                # check if entry has nonzero values
                if Lv.any():
                    
                    # Hessian index sequence
                    r = DTQPy_getQPIndex(R[i],Lleft,1,nt,Istored) # To do
                    c = DTQPy_getQPIndex(C[j],Lright,1,nt,Istored) # to do
                    
                    
                    # assign main diagonal
                    Isav = np.append(Isav,r)
                    Jsav = np.append(Jsav,c)
                    
                    # tp do
                    # integration weights
                    HWsav = np.append(HWsav,h0)
                    #breakpoint()
                    # main diagonal matrix values
                    Qsav = np.append(Qsav,Lv)
                    
                    # to do off diagonal
        
        # method specific calculation
    #breakpoint()
    V = (HWsav + np.roll(HWsav,1,axis=0))*Qsav/2 # composite trapezoidal rule
     
    return Isav,Jsav,V
        
        
    

                   
    