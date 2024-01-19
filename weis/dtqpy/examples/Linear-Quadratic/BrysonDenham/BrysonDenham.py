# -*- coding: utf-8 -*-
"""
BrysonDenham
Bryson Denham problems

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

# import necessary libraries


import os
import sys

sys.path.insert(1,'/dt-qp-py-project/src')

os.chdir('../../../')
import numpy as np
from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve

#from DTQPy_create_bnds import DTQPy_create_bnds
import matplotlib.pyplot as plt

# empty class as a struct
aux = auxdata()

opts = options()
opts.dt.nt = 1000
opts.solver.function = 'pyoptsparse'

opts.solver.tolerence = 1e-5
#opts.solver.maxiters = 10000

# System Matrices
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])

# objective
L = LQ_objective(left = 1,right = 1,matrix = 1/2)


# UB,LB
UB = [Simple_Bounds() for n in range(3)]
LB = [Simple_Bounds() for n in range(2)]


UB[0].right = 4; UB[0].matrix = np.array([[0],[1]]) # initial states
LB[0].right = 4; LB[0].matrix =  np.array([[0],[1]])
UB[1].right = 5; UB[1].matrix = np.array([[0],[-1]]) # final states
LB[1].right = 5; LB[1].matrix = np.array([[0],[-1]])
UB[2].right = 2; UB[2].matrix = np.array([[1/9],[np.inf]]) # states

# combine
s = setup()

s.t0 = 0; s.tf = 1
s.A = A; s.B = B;
s.Lagrange = L
s.UB = UB; s.LB = LB
s.auxdata = aux

# solve
T,U,Y,P,F,internal,opts = DTQPy_solve(s,opts)

# plot
plt.close('all')
fig,ax = plt.subplots()
ax.plot(T, Y[:,0], label="x1")
ax.plot(T, Y[:,1], label="x2")
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('States');

fig, a = plt.subplots()
a.plot(T,U,label = "u")
a.set_xlabel('t')
a.set_ylabel('u')
a.set_title('Controls');

          
        
        
        
