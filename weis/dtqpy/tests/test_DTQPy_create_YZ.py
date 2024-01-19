# -*- coding: utf-8 -*-
"""
DTQPy_create_YZ_test
test file for DTQPy_crearte YZ

"""
import numpy as np
from DTQPy_CLASS_SETUP import *
from DTQPy_create_YZ import DTQPy_create_YZ
from numpy.matlib import repmat

class dummy:
    pass

internal = dummy()

internal.t0 = 0;
internal.tf = 1
internal.nt = 20;nt = 20

internal.ny = 4;ny = 4
internal.nu = 2;nu = 2
internal.npl = 1;npl=1
internal.nd = 0

t = np.linspace(0,1,internal.nt)

t = t[None].T

internal.t = t
internal.h = np.diff(t,axis=0)
internal.auxdata = []
internal.nx = (ny+nu)*nt + npl

IN = []
IN.append(np.arange(1,nu+1))
IN.append(np.arange(nu+1,nu+ny+1))
IN.append(np.arange(nu+ny+1,nu+ny+npl+1))
IN.append(IN[1])
IN.append(IN[1])
IN.append(IN[0])
IN.append(IN[0])
    
I_stored = np.hstack([np.reshape(np.arange(0,(nu+ny)*nt),(nt,(ny+nu)),order = 'F'),repmat(((nu+ny)*nt+np.arange(0,npl)),nt,1)])
    
internal.IN = IN; internal.I_stored = I_stored

Y = [Simple_Linear_constraints() for n in range(1)]

linearY = [Simple_Linear_Bound() for n in range(2)]

linearY[0].right = 4; linearY[0].matrix = np.array([[0],[1],[0],[0]])
linearY[1].right = 5; linearY[1].matrix = np.array([[0],[-1],[0],[0]])

Y[0].linear = linearY
Y[0].b = 0

Z = [Simple_Linear_constraints() for n in range(2)]

linearZ1 = [Simple_Linear_Bound() for n in range(2)]

linearZ1[0].right = 2; linearZ1[0].matrix = np.array([[-1],[0],[0],[0]])
linearZ1[1].right = 1; linearZ1[1].matrix = np.array([[0],[1/12]])

Z[0].linear = linearZ1
Z[0].b = 0

linearZ2 = [Simple_Linear_Bound() for n in range(2)]

linearZ2[0].right = 2; linearZ2[0].matrix = np.array([[0],[0],[1],[0]])
linearZ2[1].right = 3; linearZ2[1].matrix = np.ones((1,1))*-1

Z[1].linear = linearZ2
Z[1].b = 0

Aeq,beq = DTQPy_create_YZ(Y,internal)

A,b = DTQPy_create_YZ(Z,internal)