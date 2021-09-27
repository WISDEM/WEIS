# -*- coding: utf-8 -*-
"""
DTQPy_CLASS_INTERNAL
Create Internal class and its associated elements

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from numpy.matlib import repmat
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *

def DTQPy_extact_order_subsets(Otemp):
    
    # initialize
    Oquadratic = []
    Olinear = []
    Oconstant = []
        
    for i in range(len(Otemp)):
            # check shape of objective array
            Otemp[i].Check_shape()
            
            # get logical for right index
            rLogical = (Otemp[i].right>0)
            
            # get logical for left index
            lLogical = (Otemp[i].left>0)
            
            # combine
            IndCombined = np.array([Otemp[i].left,Otemp[i].right])
            
            # check for nonzero
            NonZero = np.nonzero(IndCombined)
            
            # find logical sum
            sLogical = rLogical + lLogical
 
            if sLogical == 2:
                
                # Quadratic
                Oquadratic.append(Otemp[i])
                
            elif sLogical == 1:
                
                # linear objective term
                if rLogical:
                   Ol = LQ_objective(left = 0,right = sum(IndCombined[NonZero]),matrix = Otemp[i].matrix)
                elif lLogical:
                    Ol = LQ_objective(left = 0,right = sum(IndCombined[NonZero]),matrix = Otemp[i].matrix.T)
                Olinear.append(Ol)
                
            elif sLogical == 0:
                
                # Constant term
                Oc = LQ_objective(left = 0,right = 0,matrix = Otemp[i].matrix)
                Oconstant.append(Oc)
    
    return Oquadratic,Olinear,Oconstant

def DTQPy_initialize(setup,dt):
    
    # create empty class
    class internal:
       pass
    
    # instantiate class      
    i = internal()
    
    # initial and final time
    i.t0 = setup.t0
    i.tf = setup.tf
    
    #create mesh
    t = np.linspace(i.t0,i.tf,dt.nt)
    t = t[None].T
    w = []; D = []
    nt = len(t); i.nt = nt;
    i.t = t;i.w = w;i.D = D;
    h = np.diff(t,axis=0);i.h = h
    
    # mid points
    i.tm = np.array([])
    
    # get the number of controls, states and parameters
    ny = max([np.shape(setup.A)[0],np.shape(setup.B)[0],np.shape(setup.G)[0],np.shape(setup.d)[0]]);i.ny = ny
    nu = np.shape(setup.B)[1];i.nu = nu
    npl = np.shape(setup.G)[1];i.npl = npl
    nd = np.shape(setup.d)[1];i.nd = nd
    
    # total number of optimization variables
    nx = (nu+ny)*nt + npl; i.nx = nx
    
    # initialize dummy A matrix
    if np.size(setup.A) == 0:
        setup.A = np.zeros((ny,ny))
        
    # indices
    IN = []
    IN.append(np.arange(1,nu+1))
    IN.append(np.arange(nu+1,nu+ny+1))
    IN.append(np.arange(nu+ny+1,nu+ny+npl+1))
    IN.append(IN[1])
    IN.append(IN[1])
    IN.append(IN[0])
    IN.append(IN[0])
    
    I_stored = np.hstack([np.reshape(np.arange(0,(nu+ny)*nt),(nt,(ny+nu)),order = 'F'),repmat(((nu+ny)*nt+np.arange(0,npl)),nt,1)])
    
    i.IN = IN
    i.I_stored = I_stored
    
    # auxdata
    i.auxdata = setup.auxdata
    
    # lagrange and mayer terms
    
    # get Lagrange terms
    Ltemp = setup.Lagrange
    
    # get quadratic, linear and constant terms
    Lquadratic,Llinear,Lconstant = DTQPy_extact_order_subsets(Ltemp)
    
    # get Mayer term
    Mtemp = setup.Mayer
    
    # get quadratic, linear and constant terms
    Mquadratic,Mlinear,Mconstant = DTQPy_extact_order_subsets(Mtemp)
    
     # add to setup  
    setup.Lquadratic = Lquadratic; setup.Llinear = Llinear; setup.Lconstant = Lconstant
    setup.Mquadratic = Mquadratic; setup.Mlinear = Mlinear; setup.Mconstant = Mconstant
    
    # Scaling terms
    lenScale = len(setup.Scaling)
    if lenScale !=0:
        for k in range(lenScale):
            
            setup.Scaling[k].Check_Fields(i)
 
    return setup,i