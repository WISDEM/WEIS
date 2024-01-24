import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.integrate import trapz,solve_ivp
from scipy.interpolate import CubicSpline
from pCrunch.io import load_FAST_out
from ROSCO_toolbox.ofTools.fast_io import output_processing
from ROSCO_toolbox.ofTools.util import spectral
from matplotlib.pyplot import cm




if __name__ == '__main__':
    
     # get path to current file
    run_dir  = os.path.dirname(os.path.dirname(os.path.realpath(__file__)) )

    # folder name
    fol_name = '/home/athulsun/WEIS-AKS/examples/17_DFSM'  #'/test/openfast_runs'
    outputs_dir =  fol_name
    
    # file prefix and suffix
    prefix1 = 'RM1_4'
    prefix2 = 'simDEBUG.RO'
  
    suffix = '.outb'
    db_suffix = '.RO.dbg2'    

    # load outputs
    op_RO2 = output_processing.output_processing()
    debug_file1 = outputs_dir + os.sep + prefix1 + db_suffix
    debug_file2 = outputs_dir + os.sep + prefix2 + db_suffix
    
    #debug_output1 = op_RO2.load_fast_out(debug_file1,tmin = 0)[0]
    debug_output2 = op_RO2.load_fast_out(debug_file2,tmin = 0)[0]

    # time_OF = debug_output1['Time']
    # GenSpeed_OF = debug_output1['GenSpeed']
    # GenSpeedF_OF = debug_output1['GenSpeedF']
    
    time_DFSM = debug_output2['Time']
    GenSpeed_DFSM = debug_output2['GenSpeed']
    GenSpeedF_DFSM = debug_output2['GenSpeedF']
    
    #--------------------------------------------------------
#     fig,ax = plt.subplots(1)
    
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('FA_Acc [m/s2]')
    
#     ax.plot(time_debug,FA_Acc,label = 'FA_Acc')
#     ax.plot(time_debug,FA_AccF,'--',label = 'FA_AccF')
#     ax.set_xlim([time_debug[0],time_debug[-1]])
#     ax.legend(ncol = 2)
    
#     #-------------------------------------------------------------------
#     fig,ax = plt.subplots(1)

#     xf,FFT_uf,_ = spectral.fft_wrap(time_debug,FA_Acc,averaging = 'Welch',averaging_window= 'hamming')
#     xf,FFT_f,_ = spectral.fft_wrap(time_debug,FA_AccF,averaging = 'Welch',averaging_window= 'hamming')
    
#     ax.loglog(xf,np.sqrt(FFT_uf),label = 'unfiltered')
#     ax.loglog(xf,np.sqrt(FFT_f),label = 'filtered')
#     ax.set_title('FA_Acc')
#     ax.set_xlabel('Freq [Hz]')
#     ax.set_xlim([np.min(xf),np.max(xf)])
#     ax.axvline(2.42/6.28,linestyle = ':',color = 'k',alpha = 0.7)
#     ax.legend(ncol = 2)

    #--------------------------------------------------------------------
    # gen speed 
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('GenSpeed [rad/s]')
    
    ax.plot(time_DFSM,GenSpeed_DFSM,label = 'GenSpeed')
    ax.plot(time_DFSM,GenSpeedF_DFSM,'--',label = 'GenSpeedF')
    ax.set_xlim([100,time_DFSM[-1]])
    ax.set_ylim([60,68])
    ax.legend(ncol = 2)
    
    #-------------------------------------------------------------------
    fig,ax = plt.subplots(1)


    xf,FFT_uf,_ = spectral.fft_wrap(time_DFSM,GenSpeed_DFSM,averaging = 'Welch',averaging_window= 'hamming')
    xf,FFT_f,_ = spectral.fft_wrap(time_DFSM,GenSpeedF_DFSM,averaging = 'Welch',averaging_window= 'hamming')
    
    ax.loglog(xf,np.sqrt(FFT_uf),label = 'unfiltered')
    ax.loglog(xf,np.sqrt(FFT_f),label = 'filtered')
    ax.set_title('GenSpeed')
    ax.set_xlabel('Freq [Hz]')
    ax.set_xlim([np.min(xf),np.max(xf)])
    ax.axvline(2.42/6.28,linestyle = ':',color = 'k',alpha = 0.7)
    ax.legend(ncol = 2)

    plt.show()
    
    
     # notch filter

    BetaDen = 0.25
    BetaNum = 0
    nf = 3
    omega =  2.42 #np.linspace(3,2,nf)
    
    nt = len(time_DFSM)
    
    # K = 2/dt
    # b2 = (K**2 + 2*omega*BetaNum*K + omega**2)/(K**2+2*omega*BetaDen*K+omega**2)
    # b1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
    # b0 = (K**2 - 2*omega*BetaNum*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
    # a1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
    # a0 = (K**2 - 2*omega*BetaDen*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)

    # for i_ in range(nt):
        
    #     # first iteration
    #     if i_ == 0:

    #         OutputSignalLast1 = GenSpeed_uf[i_]
    #         OutputSignalLast2 = GenSpeed_uf[i_]

    #         InputSignalLast1 = GenSpeed_uf[i_]
    #         InputSignalLast2 = GenSpeed_uf[i_]

    #     GenSpeed_filt[i_] = b2*GenSpeed_uf[i_] + b1*InputSignalLast1 + b0*InputSignalLast2 - a1*OutputSignalLast1 - a0*OutputSignalLast2

    #     # update
    #     InputSignalLast2 = InputSignalLast1 
    #     InputSignalLast1 = GenSpeed_uf[i_]

    #     OutputSignalLast2 = OutputSignalLast1 
    #     OutputSignalLast1 = GenSpeed_filt[i_]

    # ax.plot(time,GenSpeed_filt,label = '$\\omega$ = ' + (str.format('{0:.1f}',omega)))

    # xf,FFT_f,_ = spectral.fft_wrap(time,GenSpeed_filt,averaging = 'Welch',averaging_window= 'hamming')

    # ax1.loglog(xf*6.2,np.sqrt(FFT_f))

    # # ax1.legend(ncol = 6)
    # ax.legend(ncol = 6)
    
   

    
    
    
