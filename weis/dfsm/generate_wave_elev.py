import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from rosco.toolbox.ofTools.util import spectral


def jonswap(Hs,Tp,df,f_high,f_low):

    fp = 1/Tp 

    Tp_o_Hs = Tp/np.sqrt(Hs)

    if (Tp_o_Hs) < 3.6:
        gamma = 5
    elif (Tp_o_Hs >= 3.6) and (Tp_o_Hs <= 5):
        gamma = np.exp(5.75 -1.15*Tp_o_Hs)
    elif (Tp_o_Hs > 5):
        gamma = 1

    v_freq = np.arange(f_low,f_high,df)
    nf = len(v_freq)

    S = np.zeros((nf,))

    for i in range(nf):

        freq = v_freq[i]

        if freq <= fp :
            sigma = 0.07
        else:
            sigma = 0.09

        S[i] = 0.3125*(Hs**2)*Tp*((freq/fp)**(-5))*np.exp(-1.25*(freq/fp)**(-4))*(1 -0.287*np.log (gamma))*(gamma)**(np.exp(-0.5*(((freq/fp )-1)*(1/sigma))**2))

    return v_freq,S
        

def generate_wave_elev(time,Hs,Tp,rng):
    '''
    Function to generate wave profiles for the given significant height and period
    using the JONSWAP spectrum

    Code adopted from: http://emmanuel.branlard.free.fr/work/papers/html/2010wral4/Branlard-2010-WindTimesSeriesGeneration.pdf

    inputs:
    time: time mesh
    Hs: significant wave height
    Tp: significant period
    rng: random number generator with specified seed

    returns
    eta: wave elevation time series 
    
    '''

    # number of time points
    nt = len(time)

    # hard code frequency cut off
    f_high = 0.5 # [hz]
    f_low = 0.0250000585 #[hz]

    df = 0.0005

    v_freq,S = jonswap(Hs,Tp,df,f_high,f_low)

    nf = len(v_freq)
    
    v_phase = rng.random((nf,))*2*pi

    v_amp = np.sqrt(2*S*df)

    eta = np.zeros((nt,))

    for i in range(nt):

        eta[i] = np.sum(v_amp*np.cos(2*pi*v_freq*time[i] + v_phase))

    return eta


if __name__ == '__main__':

    # test for function 'generate_wave_elev'

    # wave height and perisod corresponding to SSS
    Hs = 6.3 # meter
    Tp = 11.5 # seconds

    # generate time profile
    t0 = 0
    dt = 0.01
    tf = 600

    time = np.arange(t0,tf+dt,dt)

    # number of waves
    n_wave = 3

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 14
    fontsize_axlabel = 18
    fontsize_tick = 12

    # initialize two plots. one for time series of the wave profile, and one for PSD of the generated wave

    # initialize time series plot
    fig,ax1 = plt.subplots(1)
    ax1.tick_params(labelsize=fontsize_tick)
    ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
    ax1.set_ylabel('Wave Elevation [m]',fontsize = fontsize_axlabel)
    ax1.set_xlim([t0,tf])

    # initialize PSD plot
    fig,ax2 = plt.subplots(1)
    ax2.tick_params(labelsize=fontsize_tick)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Freq. [Hz]',fontsize = fontsize_axlabel)
    ax2.set_ylabel('PSD',fontsize = fontsize_axlabel)

    # initialize random number generator
    rng = np.random.default_rng(12345)

    # loop through and generate wave profiles
    for i in range(n_wave):

        # generate wave profile
        eta = generate_wave_elev(time,Hs,Tp,rng)

        # plot time series
        ax1.plot(time,eta,label = 'wave_' + str(i))

        # eavluate PSD response
        fq,y,_ = spectral.fft_wrap(time,eta,averaging = 'welch',averaging_window='Hamming', output_type='psd')

        # plot PSD
        ax2.plot(fq,np.sqrt(y),label = 'wave_' + str(i))
    
    # add legend
    ax1.legend(ncol = 1,fontsize = fontsize_legend)
    ax2.legend(ncol = 1,fontsize = fontsize_legend)

    plt.show()


