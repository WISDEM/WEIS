#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_scalingLinear.m
Apply scaling to the problem based on a linear transformation

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

import numpy as np
import types
from scipy.sparse import csc_matrix
from scipy import sparse

def createScalingVector(y,ny,T,nt):
    
    # create the scaling vector for the particular input
    
    if type(y) is types.LambdaType:
        # evaluate time-varying function
        Y = y(T)
        
        # reshape time-based matrix to  column vector
        Y.reshape(-1,1)
        
    elif np.size(y) == ny:
        
        # expand scalar scaling
        Y = np.repeat(y.reshape(-1,1),[nt])
        Y = Y.reshape(-1,1)
    elif np.shape(y) == [nt,ny]:
        
        # reshape time-based matrix to column vector
        Y = y.reshape(-1,1)
    else:
        raise ValueError("Wrong Size")
        
    return Y
    

def DTQPy_scalingLinear(H,f,c,A,b,Aeq,beq,lb,ub,internal,scaling):
    
    # extract values
    T = internal.t; nt = internal.nt; nu = internal.nu; ny = internal.ny; npl = internal.npl
    
    # initialize matrices
    s1mat = np.ones((nu*nt,1)); s2mat = np.ones((ny*nt,1)); s3mat = np.ones((npl,1))
    s1con = np.zeros((nu*nt,1)); s2con = np.zeros((ny*nt,1)); s3con = np.zeros((npl,1))
    
    for k in range(len(scaling)):
        
        # extract
        mat = scaling[k].matrix
        sc = scaling[k].constant
        right = scaling[k].right
        
        if right == 1:
            # controls
            s1mat = createScalingVector(mat,nu,T,nt)
            s1con = createScalingVector(sc,nu,T,nt)
            
        elif right == 2:
            # states
            s2mat = createScalingVector(mat,ny,T,nt)
            s2con = createScalingVector(sc,ny,T,nt)
            
        elif right ==3:
            # parameters
            s3mat = createScalingVector(mat,nu,[],nt)
            s3con = createScalingVector(sc,npl,[],nt)
            
        else:
            raise ValueError("")
    
    # combine
    sm = np.vstack([s1mat,s2mat,s3mat])
    nR = len(sm)
    r = np.arange(nR)
    
    # scaling diagonal matrix
    sM = csc_matrix((np.squeeze(sm),(r,r)),shape = (nR,nR))
    
    # scaling constant vector
    sC = sparse.vstack([s1con,s2con,s3con])
    
    
    if any(f.nonzero()[0]):
        c = c+ ((f.T).dot(sC)).todense()
        
    if any(H.nonzero()[0]):
        c = c + 0.5*((sC.T).dot(H.dot(sC))).todense()
        
    if any(f.nonzero()[0]):
        f = sM*f
        
    if any(H.nonzero()[0]):
        if any(f.nonzero()[0]):
            f = f + sM.dot(H.dot(sC))
        else:
            f = sM.dot(H.dot(sC))
            
        H = sM.dot(H.dot(sM))
    
    if any(A.nonzero()[0]):
        b = b - A.dot(sC)
        A = A.dot(sM)
        
    if any(Aeq.nonzero()[0]):
        beq = beq - Aeq.dot(sC)
        Aeq = Aeq.dot(sM)
    
    if any(ub):
        ub = (ub - np.squeeze(sC.todense()))/np.squeeze(sm)
        
    if any(lb):
        lb = (lb - np.squeeze(sC.todense()))/np.squeeze(sm)
    
      
    
        
       
    return H,f,c,A,b,Aeq,beq,lb.T,ub.T,internal,sm,sC.todense()