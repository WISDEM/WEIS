# -*- coding: utf-8 -*-
"""
DTQPy_tmatrix
Evaluate a potentially time-varying matrix on a specific time mesh

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from numpy.matlib import repmat
import types


def DTQPy_tmatrix(A,p,*args):
    
    
    # check if another time mesh is imputted
    if len(args) !=0:
        t = args[0]
    else:
        t = p.t
        
    # convert lambda function to a np array of type object
    if type(A) is types.LambdaType:
            Ae = np.empty((1,1),dtype ='O')
            Ae[0,0] = A
            A = Ae
    
    # if A is empty 
    if len(A)==0:
        At = np.array([])
    elif A.dtype.kind == 'f' or A.dtype.kind == 'i': # constant matrix
        r,c = np.shape(A) 
        
        # add dummy dimension
        Al = np.empty((1,r,c))
        Al[0,:,:] = A
        
        # replicate the matrix nt times
        At = np.tile(Al,(len(t),1,1))
    else:
        
        # obtain shape
        r,c = np.shape(A) 
        
        # initialize
        At = np.zeros((len(t),r,c))
        
        # go through each row and column in A
        for i in range(r):
            for j in range(c):
                if len(A)==0: # A is empty
                    pass # do nothing
                elif type(A[i,j]) is types.LambdaType: # A is time varying
                    if A[i,j].__code__.co_argcount == 2:
                          At[:,i,j] = np.ravel(A[i,j](t,p))
                          
                    elif A[i,j].__code__.co_argcount == 1:
                        At[:,i,j] = np.ravel(A[i,j](t))
                        
                    else:
                        raise ValueError("not a properly defined function")
                        
                elif A[i,j] == 0:
                    pass
                else:
                    At[:,i,j] = A[i,j] # A is time invariant
                    
    return At    
                
        
        
            