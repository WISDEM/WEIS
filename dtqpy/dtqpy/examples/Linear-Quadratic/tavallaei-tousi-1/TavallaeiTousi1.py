#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 07:56:26 2021

@author: athulsun
"""

import os
import sys


os.chdir('../../../')
import numpy as np
from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve

import matplotlib.pyplot as plt

aux = auxdata()

opts = options()
opts.dt.nt = 1000
opts.solver.function = 'pyoptsparse'

opts.solver.tolerence = 1e-5
opts.solver.maxiters = 10000

# tunable mparameters
auxdata.tf = 5;
auxdata.y0 = np.array([[-4],[4]]);
auxdata.k = 3;
auxdata.r = 0.5;
auxdata.Q = np.array([[1,0],[0,2]]);
auxdata.A = np.array([[2,0],[1,-1]]);
auxdata.B = np.array([[1],[0]]); # modified from reference


# initial time
t0 = 0

L = [LQ_objective() for n in range(2)]

L[0].right = 1
L[0].left = 1
L[0].matrix = auxdata.r

L[1].right = 2
L[1].left = 2
L[1].matrix = np.array([['prod',lambda t: t**3,auxdata.Q]],dtype = 'O')

LB = [Simple_Bounds() for n in range(1)]
UB = [Simple_Bounds() for n in range(1)]

UB[0].right = 4
UB[0].matrix = auxdata.y0
LB[0].right = 4
LB[0].matrix = auxdata.y0

s = setup()

s.Lagrange = L
s.UB = UB
s.LB = LB
s.t0 = t0; s.tf = auxdata.tf
s.auxdata = auxdata
s.A = auxdata.A
s.B = auxdata.B

T,U,X,P,F,internal,opts = DTQPy_solve(s,opts) 

# plot
plt.close('all')
fig,ax = plt.subplots()
ax.plot(T, X[:,0], label="x1")
ax.plot(T, X[:,1], label="x2")
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('States');
ax.legend()

fig, a = plt.subplots()
a.plot(T,U,label = "u")
a.set_xlabel('t')
a.set_ylabel('u')
a.set_title('Controls');
a.legend()