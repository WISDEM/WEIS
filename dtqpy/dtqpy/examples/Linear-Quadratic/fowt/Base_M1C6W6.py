#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:33:16 2021

@author: athulsun
"""

from mat4py import loadmat
from scipy.signal import filtfilt
import numpy as np
from scipy.interpolate import interp1d,PchipInterpolator
import matplotlib.pyplot as plt
import os
import sys


# load linear models
LinearModels = loadmat('SS2py.mat')
Chan = LinearModels['Chan']
#breakpoint()
# obtain the size of the arrays
nl = len(Chan)
nx,nx = np.shape(Chan[0]['A'])
nx,nu = np.shape(Chan[0]['B'])
ny = len(LinearModels['OutName'])

OutputName = LinearModels['OutName']

# initialize
Aw = np.zeros((nl,nx,nx))
Bw = np.zeros((nl,nx,nu))
Cw = np.zeros((nl,ny,nx))
Dw = np.zeros((nl,ny,nu))

xw = np.zeros((nx,nl))
uw = np.zeros((nu,nl))
yw = np.zeros((nl,ny))
ws = np.zeros((nl))

# collect 
for i in range(nl):
    Aw[i,:,:] = np.array(Chan[i]['A'])
    Bw[i,:,:] = np.array(Chan[i]['B'])
    Cw[i,:,:] = np.array(Chan[i]['C'])
    Dw[i,:,:] = np.array(Chan[i]['D'])
    
    xw[:,i] = np.squeeze(np.array(Chan[i]['xop']))
    uw[:,i] = np.squeeze(np.array(Chan[i]['uop']))
    yw[i,:] = np.squeeze(np.array(Chan[i]['yop']))
    ws[i] = Chan[i]['WindSpeed']

# construct LPV models
# A matrix   
A_op_pp = PchipInterpolator(ws, Aw, axis = 0)
A_op = lambda w: A_op_pp(w)

# Bmatrix
B_op_pp = PchipInterpolator(ws, Bw, axis = 0)
B_op = lambda w: B_op_pp(w)

# Cmatrix
C_op_pp = PchipInterpolator(ws,Cw,axis = 0)
C_op = lambda w: C_op_pp(w)

# Dmatrix
D_op_pp = PchipInterpolator(ws,Dw,axis = 0)
D_op = lambda w: D_op_pp(w)

# control operating points
Uo_pp = PchipInterpolator(ws,uw,axis = 1)
Uo_fun = lambda w: Uo_pp(w)

# state operating points
Xo_pp = PchipInterpolator(ws, xw, axis = 1)
Xo_fun = lambda w: Xo_pp(w)

# outputs
Yo_pp = PchipInterpolator(ws, yw, axis = 0)
Yo_fun = lambda w: Yo_pp(w)

# first time derivative of state operating points
DXo_pp = Xo_pp.derivative 
DXo_pp = DXo_pp(nu=1)
DXo_fun = lambda w: DXo_pp(w)

Wind_o = loadmat('072720_183300.mat')
Wind_o = Wind_o['Chan']
Wind_speed = np.array(Wind_o['RtVAvgxh'])
tt = np.array(Wind_o['tt'])

filterflag = 1

if filterflag:
    t_f = 1
    dt = tt[2,0]-tt[1,0]
    nb = int(np.floor(t_f/dt))
    b = np.ones((nb,))/nb
    a = b*nb
    Wind_speed = filtfilt(b,1,Wind_speed,axis = 0)
    



# temporary solution, will need to change
from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
breakpoint()
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve

opts = options()

opts.dt.nt = 1000
opts.solver.tolerence = 1e-16
opts.solver.maxiters = 1000000
opts.solver.function = 'pyoptsparse'

time = np.linspace(tt[0],tt[-1],opts.dt.nt)
W_pp = PchipInterpolator(np.squeeze(tt),np.squeeze(Wind_speed))
dW_pp = W_pp.derivative 
dW_pp = dW_pp(nu = 1)

DW_fun = lambda t: dW_pp(t)
W_fun = lambda t: W_pp(t)

DXoDt_fun = lambda t: (-DXo_fun(W_fun(t)).T*DW_fun(t)).T

def BuildLambda(Ax):
    return lambda t: Ax(t)

def TVmat2cell(f,time):
    """
    function to convert nt*nx*nz matrix to nx*nx cell

    """
    # evaluate function
    At = f(time)
    s = np.shape(At)
    
    if len(s) ==4:
        At = np.squeeze(At)
    elif len(s) == 3:
        At = np.squeeze(At)
        At= At.T
            
    #breakpoint()
    
    # get size
    try:
      null,m,n = np.shape(At)
    except:
        null,m = np.shape(At)
        n = 1
    
    # initialize storage
    A = np.empty((m,n),dtype = 'O')
    Aval = np.empty((8,8))
    #breakpoint()

    for i in range(m):
        for j in range(n):
            try:
               Ax = PchipInterpolator(np.squeeze(time),At[:,i,j],axis = 0)
            except:
               Ax = PchipInterpolator(np.squeeze(time),At[:,i],axis = 0)
            # work around, as defining lambda functions in a loop in python is tricky
            A[i,j] = BuildLambda(Ax)
            
    return A

## Disc2 cont

def BuildFunction(w_ops,X):
    Xpp = PchipInterpolator(w_ops,X)
    return lambda w: Xpp(w)

# Generator speed function
GS_fun = BuildFunction(ws,xw[4,:])

# -1*GS function
GSn_fun = BuildFunction(ws,-xw[4,:])

# Generator torque
GT_fun = BuildFunction(ws,uw[1,:])

# -Generator torque
GTn_fun = BuildFunction(ws,-uw[1,:])

# Blade pitch
BP_fun = BuildFunction(ws,uw[2,:])

# Generator power
GP_fun = BuildFunction(ws,-uw[1,:]*xw[4,:])

# State operating point values
r = Xo_fun(ws)

# lambda function to find the values of lambda function at specific indices
indexat = lambda expr,index: expr[index,:]

# get shape
nws,nx,nu = np.shape(Bw)

# initialize
ub = np.ones((nx,1))*np.inf
lb = -np.ones((nx,1))*np.inf

# set ub values for PtfmPitch and Genspeed
ub[0] = np.deg2rad(6)
ub[4] = 0.7913+0.0001

# initialize
UBx = np.empty((nx,1),dtype = 'O')
LBx = np.empty((nx,1),dtype = 'O')

# need this function to define anaonymous functions in a loop in python
def BuildLambdaUB(ub,indexat,Xo_fun,W_fun,i):
    return lambda t: ub - indexat(Xo_fun(W_fun(t)),i)

# build ub and lb functions
for i in range(nx):
    UBx[i,0] = BuildLambdaUB(ub[i],indexat,Xo_fun,W_fun,i)
    LBx[i,0] = BuildLambdaUB(lb[i],indexat,Xo_fun,W_fun,i)

# control bounds
UBc = np.array([[lambda t: W_fun(t)-W_fun(t)],
                [lambda t: max(uw[1,:])-GT_fun(W_fun(t))],
                [lambda t: max(uw[2,:])-BP_fun(W_fun(t))]])

LBc = np.array([[lambda t: W_fun(t)-W_fun(t)],
                [lambda t: min(uw[1,:])-GT_fun(W_fun(t))],
                [lambda t: min(uw[2,:])-BP_fun(W_fun(t))]])

# initial state
X0_n = np.array( [[0.0493],
    [0.1957],
    [0.0000],
    [0.0001],
    [0.7913],
         [0],
         [0],
         [0]])
UBs = X0_n - Xo_fun(W_fun(0))[None].T
LBs = X0_n - Xo_fun(W_fun(0))[None].T

# UB,LB
UB = [Simple_Bounds() for n in range(3)]
LB = [Simple_Bounds() for n in range(3)]

# states
UB[0].right = 2
UB[0].matrix = UBx
LB[0].right = 2
LB[0].matrix = LBx

# control bounds
UB[1].right = 1
UB[1].matrix = UBc
LB[1].right = 1
LB[1].matrix = LBc

# initial state
UB[2].right = 4
UB[2].matrix = UBs
LB[2].right = 4
LB[2].matrix = LBs

# lagrange terms

R1 = 1e-0; R2 = 1e+8

lx = 0

L = [LQ_objective() for n in range(5)]

# uRu
L[lx].left = 1
L[lx].right = 1
L[lx].matrix = np.diag([0,R1,R2])

lx = lx+1

# uPX
L[lx].left = 1
L[lx].right = 2
Lmat = np.zeros((nu,nx)); Lmat[1,4] = -1
L[lx].matrix = Lmat

lx = lx+1

L[lx].left = 0;
L[lx].right = 1
L2mat = np.zeros((1,nu),dtype = 'O')
L2mat[0,1] = lambda t: GSn_fun(W_fun(t))
L[lx].matrix = L2mat

lx = lx+1
L[lx].left = 0
L[lx].right = 2
L3mat = np.zeros((1,nx),dtype = 'O')
L3mat[0,4] = lambda t: GTn_fun(W_fun(t))
L[lx].matrix = L3mat

lx = lx+1

L[lx].left = 0
L[lx].right = 0
L4mat = np.empty((1,1),dtype = 'O')
L4mat[0,0] = lambda t: GP_fun(W_fun(t))
L[lx].matrix = L4mat

# 
scale = Scaling(right = 1, matrix = np.array([1,1e-16,1e-4]))


# setup
s = setup()
s.A = TVmat2cell(lambda t: A_op(W_fun(t)),time)
s.B = TVmat2cell(lambda t: B_op(W_fun(t)),time)
s.d = TVmat2cell(DXoDt_fun,time)
s.Lagrange = L
s.UB = UB
s.LB = LB
s.Scaling = scale
s.t0 = 0
s.tf = 600
#breakpoint()
[T,Ul,Xl,P,F,internal,opts] = DTQPy_solve(s,opts)

# calculate offset
Xo_off = np.squeeze(Xo_fun(W_fun(T))).T 
Uo_off = np.squeeze(Uo_fun(W_fun(T))).T 

# Add offset to estimated states
X =  Xl + Xo_off
U = Ul + Uo_off

# plot
fig, ((ax1,ax2,ax3)) = plt.subplots(3,1,)

# wind
ax1.plot(T,U[:,0])
ax1.set_title('Wind Speed [m/s]')
ax1.set_xlim([0,600])

# torue
ax2.plot(T,U[:,1]/1e+07)
ax2.set_ylim([1.8,2])
ax2.set_title('Gen Torque [MWm]')
ax2.set_xlim([0,600])

# blade pitch
ax3.plot(T,U[:,2])
#ax3.set_ylim([0.2, 0.3])
ax3.set_title('Bld Pitch [rad/s]')
ax3.set_xlim([0,600])

fig.subplots_adjust(hspace = 0.65)

fig2, ((ax1,ax2)) = plt.subplots(2,1)

# PtfmPitch
ax1.plot(T,np.rad2deg(X[:,0]))
ax1.set_xlim([0,600])
ax1.set_title('Ptfm Pitch [deg]')

# FenSpeed
ax2.plot(T,X[:,4])
ax2.set_xlim([0,600])
ax2.set_title('Gen Speed [rad/s]')

fig2.subplots_adjust(hspace = 0.65)
