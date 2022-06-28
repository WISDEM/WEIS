#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_SOLVER_pyoptsparse
Reorganize matrices and solve the problem using IPOPT from pyoptsparse

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
# import pyoptsparse
from pyoptsparse import IPOPT, Optimization
from scipy import sparse


import argparse

import numpy as np


def DTQPy_SOLVER_ipopt(H,f,A,b,Aeq,beq,lb,ub,internal,opts):
    
    
    # obtain solver preferences
    solver = opts.solver
    
    # set default options
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt",help = "optimizer",type = str, default = "IPOPT")
    args = parser.parse_args()
    
    # set specific solver options
    optOptions = {'max_iter':solver.maxiters,'tol':solver.tolerence,'print_level':1,
                'file_print_level':5,'dual_inf_tol':float(np.sqrt(solver.tolerence))}
   
    
    # obtain the number of linear equality/inequality constraints
    n = A.shape[0]; neq = Aeq.shape[0]
    
    # define a wrapper function to pass additional arguments to pyoptsparse
    class PyOptSp_wrapper():
        def __init__(self, H = H,f = f,A = A,b = b,Aeq = Aeq,beq = beq,lb = lb,ub = ub,internal = internal):
            
            # assign
            self.H = H; self.f = f
            self.A = A; self.b = b; self.Aeq = Aeq; self.beq = beq;
            self.lb = lb; self.ub = ub
            self.internal = internal
            
        # function to calculate the objective function value
        def ObjFun(self,xdict):
            
            # extract values
            H = self.H; f = self.f
            
            # extract vector of optimization variables
            x = xdict['xvars']
            
            # initialize
            funcs = {}
            
            # objective function = 1/2*x'*H*x + f'*x
            funcs['obj'] =0.5*np.dot(x,H.dot(x)) + (f.T).dot(x)

            fail = False
            
            # return
            return funcs,fail
        
        # function to evaluate the jacobian/gradient of the objective function
        def Sens(self,xdict,funcs):
            
            # extract variables
            H = self.H; f = self.f           
            x = xdict['xvars']
            
            # initialize
            funcsens = {}
            
            # gradient of the objective function
            funcsens = {
                'obj' : {'xvars': H.dot(x) + (f.T)   },
                }
            fail = False
           
            return funcsens, fail
        
    # initialize problem   
    prob = PyOptSp_wrapper(H =H, f = f, A = A,b = b,Aeq = Aeq,beq = beq,lb = lb,ub = ub,internal = internal)
    
    # solver preferences
    optProb = Optimization("DTQPy",prob.ObjFun)
    
    # add objective function
    optProb.addObj("obj")
    
    # number of optimization variables
    nx = internal.nx
    
    # LB and UB for the optimization variables
    x0 = np.zeros(nx)
    optProb.addVarGroup("xvars", nx, lower=lb, upper=ub, value=x0)
    
    # construct the upper and lower bounds for linear equality and inequality constraints
    # beq < Aeq < beq
    # -inf < A < b
    lower = np.vstack([-np.inf*np.ones((n,1)),beq.todense()])
    upper = np.vstack([b.todense(),beq.todense()])
    
    # add linear constraints
    optProb.addConGroup(
        "lincon", # type
        n + neq, # number of linear constraints
        lower = lower, # lower limit
        upper = upper, # upper limit
        linear = True, # linear
        jac = {"xvars": sparse.vstack([A,Aeq])} # jacobian/gradient of linear constraints
        )
    
    # setup IPOPT problem
    opt = IPOPT(args,options = optOptions)
    
    # solve the problem
    sol = opt(optProb,sens = prob.Sens)
    
    # extract results
    X = sol.xStar
    X = X['xvars']
    
    # get solution status
    inform = sol.optInform
    EXITFLAG = inform["value"]
    internal.output = inform["text"]
    
    # extract optimal objective function value
    if EXITFLAG < -1:
        F = None
    else:
        F = sol.fStar
    
    # return
    return X,F,internal,opts       
        
