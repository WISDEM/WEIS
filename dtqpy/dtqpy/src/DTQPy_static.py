#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_static

Evaluate open loop optimal control response for a single static model

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on github)
Primary Contributor: Daniel R. Herber (danielrherber on github)
"""

from mat4py import loadmat
from scipy.signal import filtfilt
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
from dtqpy.src.classes.DTQPy_CLASS_SETUP import *
from dtqpy.src.DTQPy_solve import DTQPy_solve


def DTQPy_static(LinearModels,disturbance,constraints,plot=False):
    
    '''
        Function to compute the open loop optimal control of a linear turbine model, given
        a disturbance and constraints

        Inputs:         LinearModels:  LinearTurbineModel object specified in weis/control
                        disturbance: dictionary with Time and Wind fields
                        constraints: dictionary with key as OpenFAST state or output, value is a list with lower and upper bounds
                        plot: boolean flag whether to plot outputs
    '''
    
    # load linear models
    Aw = np.squeeze(LinearModels.A_ops)
    Bw = np.squeeze(LinearModels.B_ops)
    Cw = np.squeeze(LinearModels.C_ops)
    Dw = np.squeeze(LinearModels.D_ops)
    
    DescOutput = LinearModels.DescOutput
    DescStates = LinearModels.DescStates
    DescCtrl = LinearModels.DescCntrlInpt
    
    # operating points
    xw = LinearModels.x_ops
    uw = LinearModels.u_ops
    yw = LinearModels.y_ops; 
    
    reduce_flag = True
    
    nx,nu = np.shape(Bw)
    
    if reduce_flag:
        nx_new = np.zeros((nx),dtype = 'bool')
        nx_new[0:5] = True
        
        Aw = Aw[nx_new,:]; Aw = Aw[:,nx_new]
        Bw = Bw[nx_new,:]
        Cw = Cw[:,nx_new]
    
        xw = xw[nx_new,:]
        
    nx,nu = np.shape(Bw)
    
    # wind speeds
    ws = LinearModels.u_h[0]
    
    # get state indices
    iGenSpeed = DescStates.index('ED First time derivative of Variable speed generator DOF (internal DOF index = DOF_GeAz), rad/s')
    iPtfmPitch = DescStates.index('ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad')
    
    iGenTorque = DescCtrl.index('ED Generator torque, Nm')
    iBldPitch = DescCtrl.index('ED Extended input: collective blade-pitch command, rad')
    
    Wind_o = disturbance
    Wind_speed = np.array(Wind_o['Wind'])
    tt = np.array(Wind_o['Time'])
    tt = tt[None].T
    
    t0 = tt[0][0]
    tf = tt[-1][0]

    filterflag = 0
    
    # filer the wind input
    if filterflag:                         
        t_f = 1
        dt = tt[2,0]-tt[1,0]
        nb = int(np.floor(t_f/dt))
        b = np.ones((nb,))/nb
        a = b*nb
        Wind_speed = filtfilt(b,1,Wind_speed,axis = 0)
        
    
    opts = options()

    opts.dt.nt = 1000
    opts.solver.tolerence = 1e-6
    opts.solver.maxiters = 150
    opts.solver.function = 'ipopt'
    
    time = np.linspace(tt[0],tt[-1],opts.dt.nt)
    W_pp = PchipInterpolator(np.squeeze(tt),np.squeeze(Wind_speed))
    W_fun = lambda t: W_pp(t)
    
    # extract values
    
    GT_static = uw[iGenTorque,:]
    BP_static = uw[iBldPitch,:]
    GS_static = xw[iGenSpeed,:]
    PP_static = xw[iPtfmPitch,:]
    GP_static = GT_static*GS_static
    
    nx,nu = np.shape(Bw)
    OutputCon_flag = False
    #initialize 
    ub = np.ones((nx,1))*np.inf 
    lb = -np.ones((nx,1))*np.inf 
    
    # set ub values for PtfmPitch and Genspeed
    for const in constraints:
        if const in DescStates:
            iConst = DescStates.index(const)
            ub[iConst] = constraints[const][1]      # max, min would be index 0
            #lb[iConst] = 0
        elif const in DescOutputs:
            iConst = DescOutputs.index(const)
            # do other output constraint things
        else:
            raise Exception(f'{const} not in DescStates or DescOutputs')
            
    GSmax = ub[iGenSpeed][0]
    
    # state bounds
    UBx = ub - xw
    LBx = lb - xw
    
    # control bounds
    UBc = np.array([[lambda t: W_fun(t)-ws],
        [1.962e+07-GT_static],
        [0.3948-BP_static]
        ])
    
    LBc = np.array([[lambda t: W_fun(t)-ws],
        [1421000-GT_static],
        [0-BP_static]
        ])
    
    # initial states
    # initial state
    X0_n = np.zeros((nx,1))
    X0_n[0:5] = np.array( [[0.0493],
        [0.1957],
        [0.0000],
        [0.0001],
        [GSmax]])
    
    L = [LQ_objective() for n in range(5)]
    
    lx = 0
    L[lx].left = 1; L[lx].right = 1; L[lx].matrix = np.diag([0,1e-0,1e+8])
    
    lx+= 1
    L1mat = np.zeros((nu,nx))
    L1mat[iGenTorque,iGenSpeed] = -1
    L[lx].left = 1;L[lx].right = 2;L[lx].matrix = L1mat
    
    lx+=1
    L[lx].left = 0;
    L[lx].right = 1
    L2mat = np.zeros((1,nu))
    L2mat[0,iGenTorque] = -GS_static
    L[lx].matrix = L2mat

    lx = lx+1
    L[lx].left = 0
    L[lx].right = 2
    L3mat = np.zeros((1,nx))
    L3mat[0,iGenSpeed] = -GT_static
    L[lx].matrix = L3mat

    lx = lx+1

    L[lx].left = 0
    L[lx].right = 0
    L4mat = np.empty((1,1))
    L4mat[0,0] = GP_static
    L[lx].matrix = L4mat
      
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
    UB[2].matrix = X0_n - xw 
    LB[2].right = 4
    LB[2].matrix = X0_n -xw
    
    # 
    scale = Scaling(right = 1, matrix = np.array([1,1e+5,1e-0]))


    # setup
    s = setup()
    s.A = Aw
    s.B = Bw
    
        
    s.Lagrange = L
    s.UB = UB
    s.LB = LB
    s.Scaling = scale
    s.t0 = t0
    s.tf = tf
    s.ScaleObjective = True
    
    # solve
    [T,Ul,Xl,P,F,internal,opts] = DTQPy_solve(s,opts)
    
    # Add offset to estimated states
    X = Xl + xw.T;
    U = Ul+uw.T
    
    # Compute output 
    yl = np.zeros((opts.dt.nt,np.shape(yw)[0]))
    for i in range(len(T)):
        t = T[i,0]
        
        
        xl = Xl[i,:]
        ul = Ul[i,:]
        
        yl[i,:] = np.squeeze(np.dot(Cw,xl.T) + np.dot(Dw,ul.T)) 
        
    Y = yl + yw.T
    
    Qty_p = 'SrvD GenPwr, (kW)'
    Yindp = DescOutput.index(Qty_p)
    
    #Y[:,Yindp] *= 10**(-3)
    
    # plot
    if plot:
        fig, ((ax1,ax2,ax3)) = plt.subplots(3,1,)

        # wind
        ax1.plot(T,U[:,0])
        ax1.set_title('Wind Speed [m/s]')
        ax1.set_xlim([t0,tf])

        # torue
        ax2.plot(T,U[:,iGenTorque]/1e+07)
        #ax2.set_ylim([1.8,2])
        ax2.set_title('Gen Torque [MWm]')
        ax2.set_xlim([t0,tf])
        ax2.set_ylim([0.14,2])

        # blade pitch
        ax3.plot(T,U[:,iBldPitch])
        #ax3.set_ylim([0.2, 0.3])
        ax3.set_title('Bld Pitch [rad]')
        ax3.set_xlim([t0,tf])

        fig.subplots_adjust(hspace = 0.65)
        
        # plot states
        fig2, ((ax1,ax2,ax3)) = plt.subplots(3,1)
        
        # PtfmPitch
        ax1.plot(T,np.rad2deg(X[:,iPtfmPitch]))
        ax1.set_xlim([t0,tf])
        ax1.set_title('Ptfm Pitch [deg]')

        # FenSpeed
        ax2.plot(T,X[:,iGenSpeed])
        ax2.set_xlim([t0,tf])
        ax2.set_title('Gen Speed [rad/s]')
        
        ax3.plot(T,Y[:,Yindp])
        ax3.set_xlim([t0,tf])
        ax3.set_title(Qty_p)
        
        fig2.subplots_adjust(hspace = 0.65)
        
        if OutputCon_flag:
            
            n_con = len(Qty)
            
            fig3,ax = plt.subplots(n_con,1)
            
            for i in range(n_con):    
                if n_con == 1:
                    Yind = DescOutput.index(Qty[i])
                    ax.plot(T,Y[:,Yind])
                    ax.set_xlim([t0,tf])
                    ax.set_title(Qty[i])
                else:   
                    Yind = DescOutput.index(Qty[i])
                    ax[i].plot(T,Y[:,Yind])
                    ax[i].set_xlim([t0,tf])
                    ax[i].set_title(Qty[i])
            fig3.subplots_adjust(hspace = 0.65)

        plt.show()
    
    return T,U,X,Y
    
    
    
    
    
    

    
    
    
    