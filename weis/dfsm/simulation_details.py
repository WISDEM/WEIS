import os
import numpy as np
from pCrunch import read
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt
import time as timer
import pickle


class SimulationDetails:
    
    def __init__(self,OF_output_files,reqd_states,reqd_controls,reqd_outputs,scale_args = {}
                 ,filter_args = {},add_dx2 = True,tmin = 0,tmax = None,linear_model_file = None,region = None,OF_file_type = 'outb'):
        
        # initialize
        self.OF_output_files = OF_output_files
        self.OF_file_type = OF_file_type
        self.reqd_states = reqd_states
        self.reqd_controls = reqd_controls
        self.reqd_outputs = reqd_outputs
        self.scale_args = scale_args
        self.filter_args = filter_args
        self.add_dx2 = add_dx2
        self.tmin = tmin
        self.tmax = tmax 
        self.linear_model_file = linear_model_file
        self.region = region
        
        # get number of simulations
        self.n_sim = len(OF_output_files)
        
        # get number of states,inputs and outputs
        self.n_states = len(reqd_states)
        self.n_controls = len(reqd_controls)
        self.n_outputs = len(reqd_outputs)
        
        # number of inputs and outputs for the DFSM model
        
        # if we add the second derivatives
        if add_dx2:
            self.n_states = self.n_states*2
        
        # model inputs
        self.n_model_inputs = self.n_states + self.n_controls
        self.n_deriv = self.n_states
        
    def scale_quantities(self,states,controls,outputs):
        
        # extract scale arguments
        scale_args = self.scale_args
        
        # scale states
        if 'state_scaling_factor' in scale_args.keys():
            states = states/scale_args['state_scaling_factor']
            
        # scale controls
        if 'control_scaling_factor' in scale_args.keys():
            controls = controls/scale_args['control_scaling_factor']
        
        # scale outputs
        if self.n_outputs > 0:
            if 'output_scaling_factor' in scale_args.keys():
                outputs = outputs/scale_args['output_scaling_factor']
            
        return states,controls,outputs
    
    def filter_signal(self,t_f,time,signal):

        '''
        Function to filter the given signal
        '''
        
        dt = time[1]-time[0]
        nb = int(np.floor(t_f/dt))
        b = np.ones((nb,))/nb;a = 1
        signal = filtfilt(b,a,signal,axis = 0)
        
        return signal
    
    def notch_filter(self,omega,time,signal):
        
        nt = len(time)
        dt = time[1]-time[0]
        BetaDen = 0.25
        BetaNum = 0
        signal_filt = np.zeros(nt)
        
        K = 2/dt
        b2 = (K**2 + 2*omega*BetaNum*K + omega**2)/(K**2+2*omega*BetaDen*K+omega**2)
        b1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
        b0 = (K**2 - 2*omega*BetaNum*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
        a1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
        a0 = (K**2 - 2*omega*BetaDen*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)

        for i_ in range(nt):
            
            # first iteration
            if i_ == 0:

                OutputSignalLast1 = signal[i_]
                OutputSignalLast2 = signal[i_]

                InputSignalLast1 = signal[i_]
                InputSignalLast2 = signal[i_]

            signal_filt[i_] = b2*signal[i_] + b1*InputSignalLast1 + b0*InputSignalLast2 - a1*OutputSignalLast1 - a0*OutputSignalLast2

            # update
            InputSignalLast2 = InputSignalLast1 
            InputSignalLast1 = signal[i_]

            OutputSignalLast2 = OutputSignalLast1 
            OutputSignalLast1 = signal_filt[i_]
            
        return signal_filt
    
    def filter_quantities(self,time,states,controls,outputs):
        
        # extract filter arguments
        filter_args = self.filter_args
        
        if 'state_filter_flag' in  filter_args.keys():
            for idx,flag in enumerate(filter_args['state_filter_flag']):
                if flag:
                    
                    for ifilt,filt_type in enumerate(filter_args['state_filter_type'][idx]):
                    
                        if filt_type == 'filtfilt':
                            t_f = filter_args['state_filter_tf'][idx][ifilt]
                            states[:,idx] = self.filter_signal(t_f,time,states[:,idx])
                            
                        elif ifilt == 'notch':
                            
                            corner_freq = filter_args['state_filter_tf'][idx][ifilt]
                            states[:,idx] = self.notch_filter(corner_freq, time, states[:,idx])
                        
        if 'control_filter_flag' in  filter_args.keys():        
            for idx,flag in enumerate(filter_args['control_filter_flag']):
                if flag:
                    t_f = filter_args['control_filter_tf'][idx]
                    controls[:,idx] = self.filter_signal(t_f,time,controls[:,idx])
                
        if 'output_filter_flag' in  filter_args.keys():
            if self.n_outputs > 0:
                for idx,flag in enumerate(filter_args['output_filter_flag']):
                    if flag:
                        t_f = filter_args['output_filter_tf'][idx]
                        outputs[:,idx] = self.filter_signal(t_f,time,outputs[:,idx])
                    
        return states,controls,outputs
                  
        
        
    def load_openfast_sim(self):
        
        FAST_sim = []
        
        # loop through and extract openfast file
        for sim_idx,file_name in enumerate(self.OF_output_files):
           
            # load output file
            if self.OF_file_type == 'outb':
                FAST_out = read(file_name)

            elif self.OF_file_type == 'pkl':
                with open(file_name,'rb') as handle:
                    FAST_out = pickle.load(handle)
            
            # extract time
            time = FAST_out['Time']
            time = time - np.min(time)
            
            tmin = self.tmin
            tmax = self.tmax 
            
            t_ind = time>=tmin
            
            time = time[t_ind]
            
            
            # number of points
            nt = len(time)
            dt = time[1]-time[0]
            # extract states
            states = np.zeros((nt,len(self.reqd_states)))
            
            for ix,state_name in enumerate(self.reqd_states):
                states[:,ix] = FAST_out[state_name][t_ind]
                
            # extract controls
            controls = np.zeros((nt,self.n_controls))
            
            for iu,control_name in enumerate(self.reqd_controls):
                controls[:,iu] = FAST_out[control_name][t_ind]
            
            # extract outputs
            if self.n_outputs > 0:
                
                outputs = np.zeros((nt,self.n_outputs))
                
                for iy,output_name in enumerate(self.reqd_outputs):
                    outputs[:,iy] = FAST_out[output_name][t_ind]
                    
            else:
                
                outputs = []

            if dt < 0.01:
                dt_ = 0.01
                t0 = time[0];tf = time[-1]
                time_ = np.arange(t0,tf+dt_,dt_)

                cs_s = CubicSpline(time,states,axis = 0)
                states = cs_s(time_)

                cs_u = CubicSpline(time,controls,axis = 0)
                controls = cs_u(time_)
                
                if self.n_outputs > 0:
                    cs_y = CubicSpline(time,outputs)
                    outputs = cs_y(time_)

                time = time_


                    
            # scale the inputs and outputs according to the options present in scale_args
            if len(self.scale_args) > 0:
                states,controls,outputs = self.scale_quantities(states,controls,outputs)
                
            if len(self.filter_args) > 0:
                states,controls,outputs = self.filter_quantities(time,states,controls,outputs)
                
            # construct polynomial approximation
            states_pp = CubicSpline(time,states)
            dx_pp = states_pp.derivative
            
            # evaluate first time derivative
            dx_pp1 = dx_pp(nu = 1)
            
            # evaluate second time derivative
            dx_pp2 = dx_pp(nu = 2)
            
            # evaluate state derivatives
            state_derivatives = dx_pp1(time)
            
            # evaluate the second time derivatives
            state_derivatives2 = dx_pp2(time)
            
            if self.add_dx2:
                
                states = np.hstack([states,state_derivatives])
                state_derivatives = np.hstack([state_derivatives,state_derivatives2])
                
                dx1_names = ['d' + s_name for s_name in self.reqd_states]
                dx2_names = ['d' + dx for dx in dx1_names]
                dx_names = dx1_names + dx2_names
                
            else:
                
                dx1_names = []
                dx_names = ['d' + s_name for s_name in self.reqd_states] 

                #self.reqd_states = self.reqd_states + dx_names
                
            # find index of genspeed
            try:
                self.gen_speed_ind = self.reqd_states.index('GenSpeed')
                
            except ValueError:
                self.gen_speed_ind = None
                
            # find index of FA_Acc
            try:
                self.FA_Acc_ind_s = self.reqd_states.index('YawBrTAxp')
                
            except ValueError:
                self.FA_Acc_ind_s = None
                
            # find index of genspeed
            try:
                self.NacIMU_FA_Acc_ind_s = self.reqd_states.index('NcIMURAys')
                
            except ValueError:
                self.NacIMU_FA_Acc_ind_s = None

            # find index of FA_Acc
            try:
                self.FA_Acc_ind_o = self.reqd_outputs.index('YawBrTAxp')
                
            except ValueError:
                self.FA_Acc_ind_o = None
                
            # find index of genspeed
            try:
                self.NacIMU_FA_Acc_ind_o = self.reqd_outputs.index('NcIMURAys')
                
            except ValueError:
                self.NacIMU_FA_Acc_ind_o = None

            # find the index of wind/current speed
            try:
                self.wind_speed_ind = self.reqd_controls.index('RtVAvgxh')

            except ValueError:
                self.wind_speed_ind = None 

            # find gen torque
            try:
                self.gen_torque_ind = self.reqd_controls.index('GenTq')

            except ValueError:
                self.gen_torque_ind = None

            # find blade pitch index
            try:
                self.blade_pitch_ind = self.reqd_controls.index('BldPitch1')

            except ValueError:
                self.blade_pitch_ind = None

            # find wave elev index
            try:
                self.wave_elev_ind = self.reqd_controls.index('Wave1Elev')

            except ValueError:
                self.wave_elev_ind = None
                
            # initialize storage dict
            sim_detail = {'sim_idx': sim_idx,
                         'n_states': self.n_states,
                        'states': states,
                        'state_names': self.reqd_states + dx1_names,
                        'n_controls': self.n_controls,
                        'controls': controls,
                        'control_names': self.reqd_controls,
                        'n_model_inputs': self.n_model_inputs,
                        'n_outputs': self.n_outputs,
                        'outputs': outputs,
                        'output_names': self.reqd_outputs,
                        'n_deriv': self.n_states,
                        'state_derivatives': state_derivatives,
                        'dx_names':dx_names,
                        'nt': len(time),
                        'time': time,
                        'w_mean': np.round(np.mean(FAST_out['Wind1VelX']),decimals = 2)
                      }
            
            FAST_sim.append(sim_detail)
            
        self.FAST_sim = FAST_sim
