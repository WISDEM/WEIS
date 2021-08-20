#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_TEST_scaling
Testing scaling functionality

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

import os
import sys

os.chdir('../')
import numpy as np
from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve
import matplotlib.pyplot as plt


# Bryson Denham problems
def problem():
    opts = options()
    
    # options
    opts.dt.nt = 1000
    opts.solver.function = 'pyoptsparse'
    
    s = setup()
    s.t0 = 0; s.tf = 1
    
    A = np.array([[0,1],[0,0]])
    B = np.array([[0],[1]])
    
    # Lagrange term
    L = LQ_objective(left = 1, right = 1,matrix = 0.5*np.ones((1,1)))
    
    # ub and lb
    UB = [Simple_Bounds() for n in range(3)]
    LB = [Simple_Bounds() for n in range(2)]
    
    UB[0].right = 4; UB[0].matrix = np.array([[0],[1]])
    LB[0].right = 4; LB[0].matrix = np.array([[0],[1]])
    
    UB[1].right = 5; UB[1].matrix = np.array([[0],[-1]])
    LB[1].right = 5; LB[1].matrix = np.array([[0],[-1]])
    
    UB[2].right = 2; UB[2].matrix = np.array([[1/9],[np.inf]])
    
    s.A = A; s.B = B; s.Lagrange = L; s.UB = UB; s.LB = LB
    
    return s,opts
    

test = [1]

[s,opts] = problem()
T1 = [];U1 = [];X1 = [];P1 = [];F1 = [];internal1 = [];opts1 = []
fig,ax = plt.subplots()
fig, a = plt.subplots()

for k in range(len(test)):
    
    s.Scaling = []
    
    if test[k] == 1:
        scaling = Scaling(right = 1, matrix = 6)
        s.Scaling = scaling
        
    elif test[k] == 2:
        scaling = Scaling(right = 2,matrix = np.array([1/9,1]))
        s.Scaling = scaling
        
    
    T,U,X,P,F,internal,opts = DTQPy_solve(s,opts)
    T1.append(T); U1.append(U); X1.append(X); P1.append(P); F1.append(F);internal1.append(internal);opts1.append(opts)
    
    # plot
    #plt.close('all')
    
    ax.plot(T, X[:,0], label="x1")
    ax.plot(T, X[:,1], label="x2")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('States');
    
    
    a.plot(T,U,label = "u")
    a.set_xlabel('t')
    a.set_ylabel('u')
    a.set_title('Controls');

    
    