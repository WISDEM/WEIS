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
    prefix = 'simDEBUG'
  
    suffix = '.outb'
    db_suffix = '.RO.RO.dbg2'    

    # load outputs
    op_RO2 = output_processing.output_processing()
    debug_file = outputs_dir + os.sep + prefix + db_suffix
    debug_output = op_RO2.load_fast_out(debug_file,tmin = 0)[0]

    time_debug = debug_output['Time']
    FA_Acc = debug_output['FA_Acc']
    FA_AccF = debug_output['FA_AccF']
    GenSpeed = debug_output['GenSpeed']
    GenSpeedF = debug_output['GenSpeedF']
    
    #--------------------------------------------------------
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('FA_Acc [m/s2]')
    
    ax.plot(time_debug,FA_Acc,label = 'FA_Acc')
    ax.plot(time_debug,FA_AccF,'--',label = 'FA_AccF')
    ax.set_xlim([time_debug[0],time_debug[-1]])
    ax.legend(ncol = 2)
    
    #-------------------------------------------------------------------
    fig,ax = plt.subplots(1)

    xf,FFT_uf,_ = spectral.fft_wrap(time_debug,FA_Acc,averaging = 'Welch',averaging_window= 'hamming')
    xf,FFT_f,_ = spectral.fft_wrap(time_debug,FA_AccF,averaging = 'Welch',averaging_window= 'hamming')
    
    ax.loglog(xf,np.sqrt(FFT_uf),label = 'unfiltered')
    ax.loglog(xf,np.sqrt(FFT_f),label = 'filtered')
    ax.set_title('FA_Acc')
    ax.set_xlabel('Freq [Hz]')
    ax.set_xlim([np.min(xf),np.max(xf)])
    ax.axvline(2.42/6.28,linestyle = ':',color = 'k',alpha = 0.7)
    ax.legend(ncol = 2)
    
    #--------------------------------------------------------------------
    # gen speed 
    fig,ax = plt.subplots(1)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('GenSpeed [rad/s]')
    
    ax.plot(time_debug,GenSpeed,label = 'GenSpeed')
    ax.plot(time_debug,GenSpeedF,'--',label = 'GenSpeedF')
    ax.set_xlim([time_debug[0],time_debug[-1]])
    ax.legend(ncol = 2)
    
    #-------------------------------------------------------------------
    fig,ax = plt.subplots(1)


    xf,FFT_uf,_ = spectral.fft_wrap(time_debug,GenSpeed,averaging = 'Welch',averaging_window= 'hamming')
    xf,FFT_f,_ = spectral.fft_wrap(time_debug,GenSpeedF,averaging = 'Welch',averaging_window= 'hamming')
    
    ax.loglog(xf,np.sqrt(FFT_uf),label = 'unfiltered')
    ax.loglog(xf,np.sqrt(FFT_f),label = 'filtered')
    ax.set_title('GenSpeed')
    ax.set_xlabel('Freq [Hz]')
    ax.set_xlim([np.min(xf),np.max(xf)])
    ax.axvline(2.42/6.28,linestyle = ':',color = 'k',alpha = 0.7)
    ax.legend(ncol = 2)
    
    
    
