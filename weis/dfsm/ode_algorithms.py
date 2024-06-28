import numpy as np
from weis.dfsm.evaluate_dfsm import evaluate_dfsm
from weis.dfsm.dfsm_utilities import extrapolate_controls

def RK4(x0, dt, tspan, DFSM, param):

    # calculate intervals
    t0 = tspan[0]; tf = tspan[1]

    # initialize
    T = np.arange(t0,tf+dt,dt)
    n = len(T); nx = len(x0)
    X = np.zeros((n,nx))

    ny = param['ny']

    if ny == 0:
        Y = []
    else:
        Y = np.zeros((n,ny))
         
   
    # starting point
    X[0,:] = x0 
    
    # parameter to scale generator speed
    gen_speed_scaling = param['gen_speed_scaling']

    # extract current speed and wave function
    wave_fun = param['wave_fun']
    wind_fun = param['w_fun']

    # evaluate wind/current speed for the time stencil
    WS = wind_fun(T)

    # convert rpm to rad/s
    # OpenFAST stores genspeed as rpm, whearas ROSCO requiers genspeed in rad/s
    rpm2RadSec = 2.0*(np.pi)/60.0


    # initialize extrapolation arrays
    if wave_fun == None:
        U_extrap = np.zeros((2*n-1,3))
        U = np.zeros((n,3))

    else:
        WE = wave_fun(T)
        U_extrap = np.zeros((2*n-1,4))
        U = np.zeros((n,4))

    # storage array
    ind_extrap = 0
    T_extrap = np.zeros((2*n-1,))
    T_extrap[ind_extrap] = T[0]

    # set first entry in the extrapolation 
    U_extrap[ind_extrap,0] = WS[0]
    U_extrap[ind_extrap,1] = param['gen_torque'][0]
    U_extrap[ind_extrap,2] = param['blade_pitch'][0]
    
    # store control
    U[0,DFSM.wind_speed_ind] = WS[0]
    U[0,DFSM.gen_torque_ind] = param['gen_torque'][0]
    U[0,DFSM.blade_pitch_ind] = param['blade_pitch'][0]

    UK4 = []

    if not(wave_fun == None):
        U[0,DFSM.wave_elev_ind] = WE[0]
        U_extrap[ind_extrap,DFSM.wave_elev_ind] = WE[0]
   
    # loop through and evaluate states
    for h in range(1,n):

        t_n = T[h-1]
        x_n = X[h-1,:]

        # get from previous time step
        u_k1 = U[h-1,:]

        # inputs for DFSM
        inputs = np.hstack([u_k1,x_n])

        k1 = evaluate_dfsm(DFSM,inputs,'deriv')

        # if h == 1:
        #     u_k2 = u_k1 # zero order hold for the first step
        # else:
        #     #breakpoint()
        #     u2_ =  u_k1
        #     u1_ = U_extrap[ind_extrap-1,:]
        #     t2_ = T_extrap[ind_extrap]
        #     t1_ = T_extrap[ind_extrap-1]
        #     u_k2 = extrapolate_controls(t_n + dt/2,[u1_,u2_],[t1_,t2_])
        #     u_k2[0] = wind_fun(t_n + dt/2)

        #     if not(wave_fun == None):
        #         u_k2[3] = wave_fun(t_n + dt/2)
        u_k2 = u_k1

        # add values to exrap array
        ind_extrap += 1
        U_extrap[ind_extrap] = u_k2
        T_extrap[ind_extrap] = t_n + dt/2


        xn_k2 = x_n + dt/2*k1
        inputs = np.hstack([u_k2,xn_k2])
        k2 = evaluate_dfsm(DFSM,inputs,'deriv')

        u_k3 = u_k2
        xn_k3 =  x_n + dt/2*k2
        inputs = np.hstack([u_k3,xn_k3])
        k3 = evaluate_dfsm(DFSM,inputs,'deriv')

        
        u2_ = U_extrap[ind_extrap,:]
        u1_ = U_extrap[ind_extrap-1,:]
        t2_ = T_extrap[ind_extrap]
        t1_ = T_extrap[ind_extrap-1]
        u_k4 = extrapolate_controls(t_n + dt,[u1_, u2_],[t1_, t2_])
        u_k4[DFSM.wind_speed_ind] = wind_fun(t_n + dt)

        if not(wave_fun == None):
            u_k4[DFSM.wave_elev_ind] = wave_fun(t_n + dt)

        
        xn_k4 = x_n + dt*k3
        inputs = np.hstack([u_k4,xn_k4])
        k4 = evaluate_dfsm(DFSM,inputs,'deriv')

        x_step = x_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        inputs = np.hstack([u_k4,x_step])

        if ny > 0:
            Y[h,:] = evaluate_dfsm(DFSM,inputs,'outputs')
        
        X[h,:] = x_step

        turbine_state = {}

        if h == n:
            turbine_state['iStatus'] = -1
        else:
            turbine_state['iStatus'] = 1

        turbine_state['bld_pitch'] = np.deg2rad(U[h-1,DFSM.blade_pitch_ind])
        turbine_state['gen_torque'] = U[h-1,DFSM.gen_torque_ind]*1000
        turbine_state['t'] = t_n
        turbine_state['dt'] = dt 
        turbine_state['ws'] = WS[h]
        turbine_state['num_blades'] = int(3)
        turbine_state['gen_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling
        turbine_state['gen_eff'] = param['VS_GenEff']/100
        turbine_state['rot_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio']
        turbine_state['Yaw_fromNorth'] = 0
        turbine_state['Y_MeasErr'] = 0

        if not(DFSM.FA_Acc_ind_s == None):
            turbine_state['FA_Acc'] = X[h,DFSM.FA_Acc_ind_s]/1

        elif not(DFSM.FA_Acc_ind_o == None):
            turbine_state['FA_Acc'] = Y[h,DFSM.FA_Acc_ind_o]/1
        
        if not(DFSM.NacIMU_FA_Acc_ind_s == None):
            turbine_state['NacIMU_FA_Acc'] = X[h,DFSM.NacIMU_FA_Acc_ind_s]*np.deg2rad(1)

        elif not(DFSM.NacIMU_FA_Acc_ind_o == None):
            turbine_state['NacIMU_FA_Acc'] = Y[h,DFSM.NacIMU_FA_Acc_ind_o]*np.deg2rad(1)

        # call ROSCO to get control values
        gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)

        # convert to right units
        gen_torque = gen_torque/1000
        bld_pitch = np.rad2deg(bld_pitch)

        U[h,DFSM.wind_speed_ind] = WS[h]
        U[h,DFSM.gen_torque_ind] = gen_torque 
        U[h,DFSM.blade_pitch_ind] = bld_pitch

        if not(wave_fun == None):
            U[h,DFSM.wave_elev_ind] = WE[h]

        ind_extrap += 1
        U_extrap[ind_extrap,:] = U[h,:]
        T_extrap[ind_extrap] = t_n + dt

    return T,X,U,Y, T_extrap, U_extrap
