# -*- coding: utf-8 -*-
"""
LQRscalar

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

import numpy as np
import os.path
import sys

#os.chdir('../../../')

from dtqpy.src.classes.DTQPy_CLASS_OPTS import options
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve
import matplotlib.pyplot as plt
#----------------------------------------
opts = options()
opts.dt.nt = 1000
opts.solver.tolerence = 1e-5
opts.solver.function = 'pyoptsparse'

#---------------------------------------
p = auxdata()

p.a = -1;p.b = 1
p.q = 1
p.r = 11
p.m = 10
p.x0 = 10
t0 = 0; tf = 1

#-------------------------------------
# system dynamics
A = p.a
B = p.b

# Lagrange term
L = [LQ_objective() for n in range(2)]

# controls
L[0].left = 1
L[0].right = 1
L[0].matrix = p.r

# states
L[1].left = 2
L[1].right = 2;
L[1].matrix = p.q

# Mayer term
M = LQ_objective()
M.left = 5
M.right = 5
M.matrix = p.m

# initial conditions
LB = Simple_Bounds();UB = Simple_Bounds()
LB.right = 4;LB.matrix = p.x0
UB.right = 4;UB.matrix = p.x0

# combine 
s = setup()
s.A = A;s.B = B;
s.Lagrange = L;s.Mayer = M
s.t0 = t0; s.tf = tf
s.auxdata = p
s.LB = LB; s.UB = UB

# solve
T,U,Y,P,F,internal,opts = DTQPy_solve(s,opts)

# plot
plt.close('all')
fig,ax = plt.subplots()
ax.plot(T, Y, label="x1")

ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('States');

fig, a = plt.subplots()
a.plot(T,U,label = "u")
a.set_xlabel('t')
a.set_ylabel('u')
a.set_title('Controls');