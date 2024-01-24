import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.signal import lfilter,bode,TransferFunction,periodogram
from scipy.fft import fft,fftfreq
from ROSCO_toolbox.ofTools.util import spectral

from pCrunch.io import load_FAST_out



if __name__ == '__main__':

    # get path to current file
    run_dir  = os.path.dirname(os.path.realpath(__file__)) 

    # location to openfast files
    of_files = run_dir + os.sep + 'outputs'+ os.sep+ 'simple_sim_TR' #+ os.sep +'openfast_runs' + os.sep + 'rank_0' 
    prefix = 'RM1_'
    n = '4' # select a load case in the rated region
    suffix = '.outb'

    # select type of filter
    filter_type = 'notch'

    if filter_type == 'lpf':

        # low pass filter
        
        # corner frequency
        nf = 10
        omega_list =  [15.008]
        nf = len(omega_list)

    elif filter_type == 'notch':

        # notch filter

        BetaDen = 0.25
        BetaNum = 0
        nf = 3
        omega_list =  [2.42] #np.linspace(3,2,nf)
        nf = len(omega_list)

    # get path to .outb file
    file_name = of_files + os.sep + prefix + n + suffix
    
    # load outputs
    outputs = load_FAST_out(file_name)[0]

    # required channels
    reqd_channels = ['Wind1VelX',"GenSpeed",'BldPitch1']
    chan_units = ['[m/s]','[rpm]','[deg]']

    # number of channels
    nchannels = len(reqd_channels)
    time = outputs['Time'];
    
    time = time - np.min(time)

    t_transient = 00
    t_ind = time >= t_transient

    time = time[t_ind]
    time = time - t_transient
    nt = len(time)
    t0 = time[0];tf = time[-1]
    dt = 1e-5; #time[1]-time[0]


    # initialize storage array
    channels = np.zeros((nt,nchannels))

    for idx,chan in enumerate(reqd_channels):
        channels[:,idx] = outputs[chan][t_ind]

    # extract unfiltered generated speed
    GenSpeed_uf = channels[:,1]

    # calculate FFT of unfiltered signal
    xf,FFT_uf,_ = spectral.fft_wrap(time,GenSpeed_uf,averaging = 'Welch',averaging_window= 'hamming')

    fig1,ax1 = plt.subplots(1)

    ax1.loglog(xf*6.2,np.sqrt(FFT_uf))
    ax1.set_title('dt = 1e-5')
    ax1.vlines(omega_list,ymin = 0.001,ymax = 100,color = 'k')
    #breakpoint()
    ax1.set_xlabel('Freq. (rad/s)')
    ax1.set_ylabel('PSD '+reqd_channels[1])
    ax1.set_ylim([0.001,100])
    #ax1.set_xlim([2,5])
    plt.grid()

    # x_ind = np.logical_and(xf*6.2>=2.3,xf*6.2<=2.4)

    # y_int = np.sqrt(FFT_uf)[x_ind]

    # y_ind = np.argmax(y_int)

    # print(6.2*xf[y_ind])

    # breakpoint()
    # initialize plot
    fig,ax = plt.subplots(1)
    ax.plot(time,GenSpeed_uf,linewidth = 2,color = 'k',label = 'unfiltered')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('GenSpeed [rpm]')
    ax.set_xlim([t0,tf])
    

    # fig2,(ax1,ax2) = plt.subplots(2,1)

    # loop through the different values and 
    for i in range(nf):

        # initialize
        GenSpeed_filt = np.zeros((nt,1))

        if filter_type == 'lpf':

            omega = omega_list[i]
            b1 = omega*dt
            b0 = omega*dt
            a1 = omega*dt + 2
            a0 = omega*dt - 2
            b = [b0,b1]
            a = [a0,a1]

            for i_ in range(nt):

                if i_ == 0:
                    # first iteration
                    OutputSignalLast = GenSpeed_uf[i_]
                    InputSignalLast = GenSpeed_uf[i_]

                # filter signal
                GenSpeed_filt[i_] = 1/a1*(-a0*OutputSignalLast + b1*GenSpeed_uf[i_] + b0*InputSignalLast )

                # update
                InputSignalLast = GenSpeed_uf[i_]
                OutputSignalLast = GenSpeed_filt[i_]

            ax.plot(time,GenSpeed_filt,label = '$\\omega$ = ' + (str.format('{0:.1f}',omega)))

            xf,FFT_f,_ = spectral.fft_wrap(time,GenSpeed_filt,averaging = 'Welch',averaging_window= 'hamming')

            ax1.loglog(xf*6.2,np.sqrt(FFT_f))

        elif filter_type == 'notch':

            omega = omega_list[i]

            K = 2/dt
            b2 = (K**2 + 2*omega*BetaNum*K + omega**2)/(K**2+2*omega*BetaDen*K+omega**2)
            b1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
            b0 = (K**2 - 2*omega*BetaNum*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
            a1 = (2*omega**2 - 2*K**2)/(K**2 + 2*omega*BetaDen*K + omega**2)
            a0 = (K**2 - 2*omega*BetaDen*K + omega**2)/(K**2 + 2*omega*BetaDen*K + omega**2)

            for i_ in range(nt):
                
                # first iteration
                if i_ == 0:

                    OutputSignalLast1 = GenSpeed_uf[i_]
                    OutputSignalLast2 = GenSpeed_uf[i_]

                    InputSignalLast1 = GenSpeed_uf[i_]
                    InputSignalLast2 = GenSpeed_uf[i_]

                GenSpeed_filt[i_] = b2*GenSpeed_uf[i_] + b1*InputSignalLast1 + b0*InputSignalLast2 - a1*OutputSignalLast1 - a0*OutputSignalLast2

                # update
                InputSignalLast2 = InputSignalLast1 
                InputSignalLast1 = GenSpeed_uf[i_]

                OutputSignalLast2 = OutputSignalLast1 
                OutputSignalLast1 = GenSpeed_filt[i_]

            ax.plot(time,GenSpeed_filt,label = '$\\omega$ = ' + (str.format('{0:.1f}',omega)))

            xf,FFT_f,_ = spectral.fft_wrap(time,GenSpeed_filt,averaging = 'Welch',averaging_window= 'hamming')

            ax1.loglog(xf*6.2,np.sqrt(FFT_f))

    # ax1.legend(ncol = 6)
    ax.legend(ncol = 6)
    
    plt.show()

    