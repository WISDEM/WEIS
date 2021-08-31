# -*- coding: utf-8 -*-
"""
DTQPy_tmultiprod
Evaluate the product of potentially time-varying matrices

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from dtqpy.src.utilities.DTQPy_tmatrix import DTQPy_tmatrix
#from DTQPy_tmatrix import DTQPy_tmatrix
import types

def DTQPy_tmultiprod(matrices,p,*args):
    
# convert lambda function to a np array of type object
    if type(matrices) is types.LambdaType:
            Ae = np.empty((1,1),dtype ='O')
            Ae[0,0] = matrices
            matrices = Ae
    
            
    # empty matrix
    if len(matrices)==0:
        A = np.array([])
    elif matrices[0,0] == 'prod':
        
        # remove 'prod' specifier
        matrices = matrices[:,1:]
        
        # initial matrix
        A = DTQPy_tmatrix(matrices[0,0],p,*args)
        
        # get the shape
        s = np.shape(A);
        
        # check for trailing singleton dimension
        sflag = (s[-1] == 1)
        
        # compute matrix products
        for k in range(1,matrices.shape[1]):
            B = DTQPy_tmatrix(matrices[0,k],p,*args)
            
            """
            The presence of a trailing singleton dimension will chenge how np.einsum 
            is carried out and the results
            """
            
            if sflag:
                A = np.einsum('ij,ikl->ikl',np.squeeze(A,axis=-1),B)
            else:
                A = np.einsum('ijk,ikl->ijl', A, B)
        
    
    
    else:
        # single matrix
        A = DTQPy_tmatrix(matrices,p,*args) 
        
        
    return A
        