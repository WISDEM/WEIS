import numpy as np 
from dfsm.evaluate_dfsm import evaluate_dfsm

def run_sim_ROSCO(t,x,DFSM,param):
    
    turbine_state = {}
    dt = t - param['time'][-1]
    param['dt'].append(dt)
    param['time'].append(t)
    
    if dt == 0:
        dt = 1e-4

    # extract data from param dict
    w = param['w_fun'](t)
    rpm2RadSec = 2.0*(np.pi)/60.0
    gen_speed_scaling = param['gen_speed_scaling']
    
    

    # populate turbine state dictionary
    if t == param['tf']:
        turbine_state['iStatus'] = -1
    else:
        turbine_state['iStatus'] = 1
        
    # first step
    # if  t == param['t0']:
    #     turbine_state['bld_pitch'] = np.deg2rad(param['bp_init'])
    #     turbine_state['gen_torque'] = param['gen_torque'][-1]*1000
        
    # else:
    turbine_state['bld_pitch'] = np.deg2rad(param['blade_pitch'][-1])
    turbine_state['gen_torque'] = param['gen_torque'][-1]*1000
    
        
    turbine_state['t'] = t
    turbine_state['dt'] = np.abs(dt)
    turbine_state['ws'] = w
    turbine_state['num_blades'] = int(2)
    turbine_state['gen_speed'] = x[DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling
    turbine_state['gen_eff'] = param['VS_GenEff']/100
    turbine_state['rot_speed'] = x[DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio']
    turbine_state['Yaw_fromNorth'] = 0
    turbine_state['Y_MeasErr'] = 0
    
    if not(DFSM.FA_Acc_ind == None):
        turbine_state['FA_Acc'] = x[DFSM.FA_Acc_ind]
        
    if not(DFSM.NacIMU_FA_Acc_ind == None):
        turbine_state['NacIMU_FA_Acc'] = x[DFSM.NacIMU_FA_Acc_ind]
    
    # call ROSCO to get control values
    gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)
    
    # convert to right units
    gen_torque = gen_torque/1000
    bld_pitch = np.rad2deg(bld_pitch)
    
    if param['wave_fun']  == None:
        u = np.array([w,gen_torque,bld_pitch])
        
    else:
        wv = param['wave_fun'](t)
        u = np.array([w,gen_torque,bld_pitch,wv])
    
    # update param list
    param['gen_torque'].append(gen_torque)
    param['blade_pitch'].append(bld_pitch)
    #param['time'].append(t)
    
    # combine
    inputs = np.hstack([u,x])
    
    # evaluate dfsm
    dx = evaluate_dfsm(DFSM,inputs,'deriv')
    
    return dx