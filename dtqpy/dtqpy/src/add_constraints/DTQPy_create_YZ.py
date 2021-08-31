# -*- coding: utf-8 -*-
"""
DTQPy_create_YZ
Create matrices for general linear constraints

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from scipy.sparse import csc_matrix

from dtqpy.src.add_constraints.DTQPy_path import DTQPy_path
from dtqpy.src.add_constraints.DTQPy_boundary import DTQPy_boundary

def DTQPy_create_YZ(YZ,internal):
    
    # total number of constraints
    nYZ = len(YZ)
    
    # initialize storage arrays
    AeqIsav = np.array([],dtype = int); AeqJsav = np.array([],dtype = int)
    AeqVsav = np.array([])
    beqIsav = np.array([],dtype = int);beqJsav = np.array([]);beqVsav = np.array([])
    
    # initialize number of mixed linear constraints
    N = 0
    
    # loop through each constraint structure
    for k in range(nYZ):
        
        YZ[k].Check_shape()
        YZflag = False
        
        # go through each substructure
        for j in range(len(YZ[k].linear)):
            
            YZ[k].linear[j].Check_shape()
            
            # check if any variable is control or state
            if YZ[k].linear[j].right < 3:
                YZflag = True # needs to be a path constraint
                
            # check for time varying matrices
            if YZ[k].linear[j].matrix.dtype.kind == 'O':
                YZflag = True # needs to be a path constraint
        
        # check if RHS is time varying
        if YZ[k].b.dtype.kind == 'O':
            YZflag = True # needs to be a path constraint
            
        if YZflag:
            
            # path constraint
            AI,AJ,AV,b = DTQPy_path(YZ[k],internal)
            
            # combine
            AeqIsav = np.append(AeqIsav,AI+N)
            AeqJsav = np.append(AeqJsav,AJ)
            AeqVsav = np.append(AeqVsav,AV)
            beqIsav = np.append(beqIsav,N+np.arange(internal.nt))
            beqVsav = np.append(beqVsav,b)
            
            # update the current number of mixed linear constraints
            N = N + internal.nt
        else:
            
            # boundary constraints
            AJ,AV,b = DTQPy_boundary(YZ[k],internal)
            
            # combine
            AeqIsav = (N)*np.ones((len(AV),))
            AeqJsav = np.append(AeqJsav,AJ)
            AeqVsav = np.append(AeqVsav,AV)
            beqIsav = np.append(beqIsav,N)
            beqVsav = np.append(beqVsav,b)
            
            # update the current number of mixed linear constraints
            N = N+1
    if nYZ !=0:
        if YZflag:
            beqJsav = np.zeros((len(beqIsav),))
        else:
            beqJsav = np.array([0])
    
    # create sparse matrix
    Aeq = csc_matrix((AeqVsav,(AeqIsav,AeqJsav)),shape=(N,internal.nx))
    beq = csc_matrix((beqVsav,(beqIsav,beqJsav)),shape = (N,1))
    
    return Aeq,beq
            
        