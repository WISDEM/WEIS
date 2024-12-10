import numpy as np
from weis.dfsm.evaluate_dfsm import evaluate_dfsm
from weis.dfsm.dfsm_utilities import extrapolate_controls

# convert rpm to rad/s
# OpenFAST stores genspeed as rpm, whearas ROSCO requiers genspeed in rad/s
rpm2RadSec = 2.0*(np.pi)/60.0
KWatt2Watt = 1000

def RK4(x0, dt, tspan, DFSM, param):

    # Implementation of the 4th order runge-kutta method
    # calculate intervals
    t0 = tspan[0]; tf = tspan[1]

    # initialize
    T = np.arange(t0,tf+dt,dt)
    nt = len(T); nx = len(x0)

    # states array
    X = np.zeros((nt,nx))

    # Check the number of outputs in the DFSM model
    ny = param['ny']

    # initalize storage array for outputs
    if ny == 0:
        Y = []
    else:
        Y = np.zeros((nt,ny))
   
    # starting point
    X[0,:] = x0 
    
    # parameter to scale generator speed
    gen_speed_scaling = param['gen_speed_scaling']

    # extract current speed and wave function
    wave_fun = param['wave_fun']
    wind_fun = param['w_fun']

    # evaluate wind/current speed for the time stencil
    WS = wind_fun(T)

    # initialize extrapolation arrays
    if wave_fun == None:
        U_extrap = np.zeros((2*nt-1,3))
        U = np.zeros((nt,3))

    else:
        WE = wave_fun(T)
        U_extrap = np.zeros((2*nt-1,4))
        U = np.zeros((nt,4))

    # storage array
    ind_extrap = 0
    T_extrap = np.zeros((2*nt-1,))
    T_extrap[ind_extrap] = T[0]

    # set first entry in the extrapolation 
    U_extrap[ind_extrap,0] = WS[0]
    U_extrap[ind_extrap,1] = param['gen_torque'][0]
    U_extrap[ind_extrap,2] = param['blade_pitch'][0]
    
    # store control
    U[0,DFSM.wind_speed_ind] = WS[0]
    U[0,DFSM.gen_torque_ind] = param['gen_torque'][0]
    U[0,DFSM.blade_pitch_ind] = param['blade_pitch'][0]


    if not(wave_fun == None):
        U[0,DFSM.wave_elev_ind] = WE[0]
        U_extrap[ind_extrap,DFSM.wave_elev_ind] = WE[0]
   
    # loop through and evaluate states
    for h in range(1,nt):

        t_n = T[h-1]
        x_n = X[h-1,:]

        # get from previous time step
        u_k1 = U[h-1,:]

        # inputs for DFSM
        inputs = np.hstack([u_k1,x_n])

        k1 = evaluate_dfsm(DFSM,inputs,'deriv')

        # zero order hold beteen time steps
        u_k2 = u_k1

        # add values to exrap array
        ind_extrap += 1
        U_extrap[ind_extrap] = u_k2
        T_extrap[ind_extrap] = t_n + dt/2

        # second step 
        # k2 = f(t_n + h/2,x_n + h*k1/2)
        xn_k2 = x_n + dt/2*k1
        inputs = np.hstack([u_k2,xn_k2])
        k2 = evaluate_dfsm(DFSM,inputs,'deriv')

        # Third step
        # k3 = f(t_n + h/2,x_n + h*k2/2)
        u_k3 = u_k2
        xn_k3 =  x_n + dt/2*k2
        inputs = np.hstack([u_k3,xn_k3])
        k3 = evaluate_dfsm(DFSM,inputs,'deriv')

        # extrapolate and find the value of the controls at tn + dt
        u2_ = U_extrap[ind_extrap,:]
        u1_ = U_extrap[ind_extrap-1,:]
        t2_ = T_extrap[ind_extrap]
        t1_ = T_extrap[ind_extrap-1]
        u_k4 = extrapolate_controls(t_n + dt,[u1_, u2_],[t1_, t2_])
        u_k4[DFSM.wind_speed_ind] = wind_fun(t_n + dt)

        if not(wave_fun == None):
            u_k4[DFSM.wave_elev_ind] = wave_fun(t_n + dt)

        # fourth step
        # k3 = f(t_n + h,x_n + hk3)
        xn_k4 = x_n + dt*k3
        inputs = np.hstack([u_k4,xn_k4])
        k4 = evaluate_dfsm(DFSM,inputs,'deriv')

        # runge kutta step
        # x_n+1 = x_n + h/6*(k1 + 2k2 + 2k3 + k4)
        x_step = x_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        # evaluate outputs at t_n+1 using the stimate states and extrapolated controls
        inputs = np.hstack([u_k4,x_step])

        if ny > 0:
            Y[h,:] = evaluate_dfsm(DFSM,inputs,'outputs')
        
        X[h,:] = x_step

        # Initialize turbine_state dict to pass the necessary information to ROSCO
        turbine_state = {}

        # operating status of the turbine
        if h == nt:
            turbine_state['iStatus'] = -1
        else:
            turbine_state['iStatus'] = 1

        turbine_state['bld_pitch'] = np.deg2rad(U[h-1,DFSM.blade_pitch_ind]) # blade pitch
        turbine_state['gen_torque'] = U[h-1,DFSM.gen_torque_ind]*KWatt2Watt # generator torque
        turbine_state['t'] = t_n # previous time step
        turbine_state['dt'] = dt # step size
        turbine_state['ws'] = WS[h] # estimate wind speed
        turbine_state['num_blades'] = int(3) # number of blades
        turbine_state['gen_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling # generator speed
        turbine_state['gen_eff'] = param['VS_GenEff']/100 # generator efficiency
        turbine_state['rot_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio'] # rotor speed
        turbine_state['Yaw_fromNorth'] = 0 # yaw
        turbine_state['Y_MeasErr'] = 0

        if not(DFSM.FA_Acc_ind_s == None):
            turbine_state['FA_Acc'] = X[h,DFSM.FA_Acc_ind_s] # tower-top acceleration if it is modelled as a state

        elif not(DFSM.FA_Acc_ind_o == None):
            turbine_state['FA_Acc'] = Y[h,DFSM.FA_Acc_ind_o] # tower top acceleration if it is modelled as an output
    
        if not(DFSM.NacIMU_FA_Acc_ind_s == None):
            turbine_state['NacIMU_FA_Acc'] = X[h,DFSM.NacIMU_FA_Acc_ind_s]*np.deg2rad(1)

        elif not(DFSM.NacIMU_FA_Acc_ind_o == None):
            turbine_state['NacIMU_FA_Acc'] = Y[h,DFSM.NacIMU_FA_Acc_ind_o]*np.deg2rad(1)

        # call ROSCO to get control values
        gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)

        # convert to right units
        gen_torque = gen_torque/KWatt2Watt
        bld_pitch = np.rad2deg(bld_pitch)

        # store
        U[h,DFSM.wind_speed_ind] = WS[h]
        U[h,DFSM.gen_torque_ind] = gen_torque 
        U[h,DFSM.blade_pitch_ind] = bld_pitch

        if not(wave_fun == None):
            U[h,DFSM.wave_elev_ind] = WE[h]

        # update the extrapolation array with the control values calculated using ROSCO
        ind_extrap += 1
        U_extrap[ind_extrap,:] = U[h,:]

    return T,X,U,Y



def ABM4(x0, dt, tspan, DFSM, param):

#----------------------------------------------------------------------------------------------------------------------------------
# This subroutine implements the fourth-order Adams-Bashforth-Moulton Method (RK4) for numerically integrating ordinary 
# differential equations:
#
#   Let f(t, x) = xdot denote the time (t) derivative of the continuous states (x). 
#
#   Adams-Bashforth Predictor: \n
#   x^p(t+dt) = x(t)  + (dt / 24.) * ( 55.*f(t,x) - 59.*f(t-dt,x) + 37.*f(t-2.*dt,x) - 9.*f(t-3.*dt,x) )
#
#   Adams-Moulton Corrector: \n
#   x(t+dt) = x(t)  + (dt / 24.) * ( 9.*f(t+dt,x^p) + 19.*f(t,x) - 5.*f(t-dt,x) + 1.*f(t-2.*dt,x) )
#
#  See, e.g.,
#  https://en.wikiversity.org/wiki/Adams-Bashforth_and_Adams-Moulton_methods#Predictor%E2%80%93corrector_method
#
#  or
#
#  K. E. Atkinson, "An Introduction to Numerical Analysis", 1989, John Wiley & Sons, Inc, Second Edition.
#----------------------------------------------------------------------------------------------------------------------------------

    # calculate intervals
    t0 = tspan[0]; tf = tspan[1]

    # initialize
    T = np.arange(t0,tf+dt,dt)
    nt = len(T); nx = len(x0)

    # array to store states
    X = np.zeros((nt,nx))

    # array to store derivative function evaluations
    F = np.zeros((nt,nx))


    # Check the number of outputs in the DFSM model
    ny = param['ny']

    # initalize storage array for outputs
    if ny == 0:
        Y = []
    else:
        Y = np.zeros((nt,ny))
   
    # starting point
    X[0,:] = x0 
    
    # parameter to scale generator speed
    gen_speed_scaling = param['gen_speed_scaling']

    # extract current speed and wave function
    wave_fun = param['wave_fun']
    wind_fun = param['w_fun']

    # evaluate wind/current speed for the time stencil
    WS = wind_fun(T)

    # initialize control arrays
    if wave_fun == None:
        U = np.zeros((nt,3))
    else:
        WE = wave_fun(T)
        U = np.zeros((nt,4))

    # store control
    U[0,DFSM.wind_speed_ind] = WS[0]
    U[0,DFSM.gen_torque_ind] = param['gen_torque'][0]
    U[0,DFSM.blade_pitch_ind] = param['blade_pitch'][0]

    if not(wave_fun == None):
        U[0,DFSM.wave_elev_ind] = WE[0]

    inputs = np.hstack([U[0,:],X[0,:]])

    # evaluate the derivative function value and store
    F[0,:] = evaluate_dfsm(DFSM,inputs,'deriv')


    # loop through and evaluate states
    for h in range(1,nt):

        # for the first three time steps use Euler forward

        if h <=3:

            t_n = T[h-1]
            x_n = X[h-1,:]

            # extract derivative function value
            f_n = F[h-1,:]

            # calculate X at the next time step
            x_step = x_n + dt*f_n

            u_ = U[h-1,:]
            inputs = np.hstack([u_,x_step])

        else:

            t_n = T[h-1]
            x_n = X[h-1,:]

            f_n = F[h-1,:]
            f_n1 = F[h-2,:]
            f_n2 = F[h-3,:]
            f_n3 = F[h-4,:]

            # predictor step
            P = x_n + dt/24*(55*f_n -59*f_n1 + 37*f_n2 - 9*f_n3)

            # corrector step
            u_ = U[h-1,:]
            inputs_ = np.hstack([u_,P])
            f_nP = evaluate_dfsm(DFSM,inputs_,'deriv')

            x_step = x_n + dt/24*(9*f_nP + 19*f_n - 5*f_n1 + f_n2)

            inputs = np.hstack([u_,x_step])

        if ny > 0:
            Y[h,:] = evaluate_dfsm(DFSM,inputs,'outputs')
        
        X[h,:] = x_step

        # Initialize turbine_state dict to pass the necessary information to ROSCO
        turbine_state = {}

        # operating status of the turbine
        if h == nt:
            turbine_state['iStatus'] = -1
        else:
            turbine_state['iStatus'] = 1

        turbine_state['bld_pitch'] = np.deg2rad(U[h-1,DFSM.blade_pitch_ind]) # blade pitch
        turbine_state['gen_torque'] = U[h-1,DFSM.gen_torque_ind]*KWatt2Watt # generator torque
        turbine_state['t'] = t_n # previous time step
        turbine_state['dt'] = dt # step size
        turbine_state['ws'] = WS[h] # estimate wind speed
        turbine_state['num_blades'] = int(3) # number of blades
        turbine_state['gen_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling # generator speed
        turbine_state['gen_eff'] = param['VS_GenEff']/100 # generator efficiency
        turbine_state['rot_speed'] = X[h,DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio'] # rotor speed
        turbine_state['Yaw_fromNorth'] = 0 # yaw
        turbine_state['Y_MeasErr'] = 0

        if not(DFSM.FA_Acc_ind_s == None):
            turbine_state['FA_Acc'] = X[h,DFSM.FA_Acc_ind_s] # tower-top acceleration if it is modelled as a state

        elif not(DFSM.FA_Acc_ind_o == None):
            turbine_state['FA_Acc'] = Y[h,DFSM.FA_Acc_ind_o] # tower top acceleration if it is modelled as an output
    
        if not(DFSM.NacIMU_FA_Acc_ind_s == None):
            turbine_state['NacIMU_FA_Acc'] = X[h,DFSM.NacIMU_FA_Acc_ind_s]*np.deg2rad(1)

        elif not(DFSM.NacIMU_FA_Acc_ind_o == None):
            turbine_state['NacIMU_FA_Acc'] = Y[h,DFSM.NacIMU_FA_Acc_ind_o]*np.deg2rad(1)

        # call ROSCO to get control values
        gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)

        # convert to right units
        gen_torque = gen_torque/KWatt2Watt
        bld_pitch = np.rad2deg(bld_pitch)

        # store
        U[h,DFSM.wind_speed_ind] = WS[h]
        U[h,DFSM.gen_torque_ind] = gen_torque 
        U[h,DFSM.blade_pitch_ind] = bld_pitch

        if not(wave_fun == None):
            U[h,DFSM.wave_elev_ind] = WE[h]

        # evaluate the derivative function
        inputs = np.hstack([U[h,:],X[h,:]])
        f_step = evaluate_dfsm(DFSM,inputs,'deriv')

        F[h,:] = f_step

    return T,X,U,Y