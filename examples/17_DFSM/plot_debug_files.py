import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.integrate import trapz,solve_ivp
from scipy.interpolate import CubicSpline
from pCrunch.io import load_FAST_out
from ROSCO_toolbox.ofTools.fast_io import output_processing
from ROSCO_toolbox.ofTools.util import spectral
from matplotlib.pyplot import cm


def filtering_function(filter_type,time,signal_uf,filter_params,dt = None):

    nt = len(time)

    if dt == None:
        dt = time[1] - time[0]


    # initialize storage array
    signal_f = np.zeros((nt,))


    if filter_type == 'lpf':

        omega = filter_params['omega']
        

        for i_ in range(nt):

            if i_ == 0:
                # first iteration
                OutputSignalLast = signal_uf[i_]
                InputSignalLast = signal_uf[i_]

                dt = 0.0021

                b1 = omega*dt
                b0 = omega*dt
                a1 = omega*dt + 2
                a0 = omega*dt - 2

            # filter signal
            signal_f[i_] = 1/a1*(-a0*OutputSignalLast + b1*signal_uf[i_] + b0*InputSignalLast )

            # update
            InputSignalLast = signal_uf[i_]
            OutputSignalLast = signal_f[i_]

    elif filter_type == 'notch':

        # extract parameters
        omega = filter_params['omega']
        BetaNum = filter_params['BetaNum']
        BetaDen = filter_params['BetaDen']

        for i_ in range(nt):
            
          # first iteration
          if i_ == 0:

               OutputSignalLast1 = signal_uf[i_]
               OutputSignalLast2 = signal_uf[i_]

               InputSignalLast1 = signal_uf[i_]
               InputSignalLast2 = signal_uf[i_]

               dt = 0.0021

               K = 2/dt
               b2 = (K**2 + 2*omega*BetaNum*K + omega**2)/(K**2+2*omega*BetaDen*K+omega**2)
               b1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
               b0 = (K**2 - 2*omega*BetaNum*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
               a1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
               a0 = (K**2 - 2*omega*BetaDen*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)

          

          signal_f[i_] = b2*signal_uf[i_] + b1*InputSignalLast1 + b0*InputSignalLast2 - a1*OutputSignalLast1 - a0*OutputSignalLast2

          # update
          InputSignalLast2 = InputSignalLast1 
          InputSignalLast1 = signal_uf[i_]

          OutputSignalLast2 = OutputSignalLast1 
          OutputSignalLast1 = signal_f[i_]

    elif filter_type == 'SecLPF':

        # extract parameters
        omega = filter_params['omega']
        damp = filter_params['damp']
        
        

        for i_ in range(nt):

            # first iteration
            if i_ == 0:

                dt = 1e-4

                OutputSignalLast1 = signal_uf[i_]
                OutputSignalLast2 = signal_uf[i_]

                InputSignalLast1 = signal_uf[i_]
                InputSignalLast2 = signal_uf[i_]

                a2 = dt**2*omega**2 + 4 + 4*damp*omega*dt 
                a1 = 2*dt**2*omega**2 - 8
                a0 = dt**2*omega**2 + 4 - 4*damp*omega*dt 
 
                b2 = dt**2*omega**2
                b1 = 2*dt**2*omega**2
                b0 = dt**2*omega**2 

            signal_f[i_] = 1/a2*(b2*signal_uf[i_] + b1*InputSignalLast1 + b0*InputSignalLast2 - a1*OutputSignalLast1 - a0*OutputSignalLast2)

            # save signals for next step
            InputSignalLast2 = InputSignalLast1
            InputSignalLast1 = signal_uf[i_]
            OutputSignalLast2 = OutputSignalLast1
            OutputSignalLast1 = signal_f[i_]

    elif filter_type == 'hpf':

        omega = filter_params['omega']

        for i_ in range(nt):

            # first iteration
          if i_ == 0:

                OutputSignalLast = signal_uf[i_]
                InputSignalLast = signal_uf[i_]

                dt = 1e-4

          else:

               dt = np.abs(time[i_] - time[i_-1])
          if dt == 0:
               dt = 1e-4
          k = 2/dt 

          signal_f[i_] = k/(omega + k)*signal_uf[i_] - k/(omega + k)*InputSignalLast - (omega - k)/(omega + k)*OutputSignalLast

          InputSignalLast = signal_uf[i_]
          OutputSignalLast = signal_f[i_]

    return signal_f

if __name__ == '__main__':
    
    # get path to current file
    run_dir  = os.path.dirname(os.path.realpath(__file__)) 
    print(run_dir)

    
    # file prefix and suffix
    prefix = 'sim_test.RO'
    suffix = '.outb'
    db_suffix = '.RO.dbg2'


    # load outputs
    
    opRO = output_processing.output_processing()
    debug_file = run_dir + os.sep + prefix + db_suffix
    
    debug_output = opRO.load_fast_out(debug_file,tmin = 0)[0]

    filter_params_lpf = {'omega':15.008}
    filter_params_notch = {'BetaDen':0.25, 'BetaNum' : 0,'omega': 2.42}
    filter_params_hpf = {'omega':0.015}
    filter_params_SecLPF = {'omega':0.6613,'damp':1.00} 


    
    time_debug = debug_output['Time']
    t_ind = time_debug >= 00
    FA_Acc = debug_output['FA_Acc'][t_ind]
    FA_AccF = debug_output['FA_AccF'][t_ind]
    GenSpeed = debug_output['GenSpeed'][t_ind]
    GenSpeedF = debug_output['GenSpeedF'][t_ind]
    BlPitch = np.rad2deg(debug_output['BlPitch'][t_ind])
    FL_PitCom = np.rad2deg(debug_output['Fl_PitCom'][t_ind])

    #breakpoint()
    BldPitch_wo = BlPitch - FL_PitCom

#     FA_Acc_f = filtering_function('SecLPF',time_debug,FA_Acc,filter_params_SecLPF,dt = 1e-4)
#     FA_Acc_f = filtering_function('hpf',time_debug,FA_Acc_f,filter_params_hpf,dt = 1e-4)
#     FA_Acc_f = filtering_function('notch',time_debug,FA_Acc_f,filter_params_notch,dt = 1e-4)

    
    #GenSpeed_n = filtering_function('notch',time_debug,GenSpeed_lpf,filter_params_notch,dt=1e-4)

    #---------------------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_title('BladePitch [deg]')
    
    
    ax.plot(time_debug[t_ind],BldPitch_wo,label = 'PI')
    ax.plot(time_debug[t_ind],FL_PitCom,label = 'FL')
    ax.plot(time_debug[t_ind],BlPitch,label = 'PI + FL')
    ax.set_xlim([time_debug[0],time_debug[-1]])
    ax.legend(ncol = 3)


    #---------------------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_title('FA_Acc [m/s2]')
    
    
    ax.plot(time_debug[t_ind],FA_Acc,label = 'unfiltered')
    ax.plot(time_debug[t_ind],FA_AccF,label = 'filtered')
    #ax.plot(time_debug,FA_Acc_f,label = 'test-filter')
    ax.set_xlim([time_debug[0],time_debug[-1]])
    ax.legend(ncol = 2)

    #---------------------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_title('GenSpeed [rad/s]')
    
    
    ax.plot(time_debug[t_ind],GenSpeed,label = 'unfiltered')
    ax.plot(time_debug[t_ind],GenSpeedF,label = 'filtered')
    ax.set_xlim([time_debug[0],time_debug[-1]])
    ax.legend(ncol = 3)

    xf_1,FFT_gs_uf,_ = spectral.fft_wrap(time_debug[t_ind],GenSpeed,averaging = 'Welch',averaging_window= 'hamming')
    xf_2,FFT_gs_f,_ = spectral.fft_wrap(time_debug[t_ind],GenSpeedF,averaging = 'Welch',averaging_window= 'hamming')

    fig1_psd,ax1_psd = plt.subplots(1)

    ax1_psd.loglog(xf_1*6.2,np.sqrt(FFT_gs_uf),label = 'unfiltered')
    ax1_psd.loglog(xf_2*6.2,np.sqrt(FFT_gs_f),label = 'filtered')
    ax1_psd.vlines(2.42,ymin = 0.001,ymax = 100,color = 'k')
    ax1_psd.legend(ncol = 2)
    ax1_psd.set_xlabel('Freq. [Hz]')
    ax1_psd.set_title('PSD GenSpeed')
    ax1_psd.grid()

    #breakpoint()
    plt.show()
    

