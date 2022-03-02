import numpy as np
import matplotlib.pyplot as plt
from weis.dlc_driver.turbulence_models import IEC_TurbulenceModels
from scipy.special import modstruve, iv

from ROSCO_toolbox.ofTools.util import spectral
from weis.aeroelasticse.turbsim_file import TurbSimFile


def IECKaimal(f, V_ref, HH, Class, Categ, TurbMod, R):
    
    ###### Initialize IEC Wind parameters #######
    iec_turb = IEC_TurbulenceModels()
    iec_turb.z_hub = HH
    iec_turb.Turbine_Class = Class
    iec_turb.Turbulence_Class = Categ
    iec_turb.setup()

    # Compute wind turbulence standard deviation (invariant with height)
    if TurbMod == 'NTM':
        sigma_1 = iec_turb.NTM(V_ref)
    elif TurbMod == 'ETM':
        sigma_1 = iec_turb.ETM(V_ref)
    elif TurbMod == 'EWM':
        sigma_1 = iec_turb.EWM(V_ref)
    else:
        raise Exception("Wind model must be either NTM, ETM, or EWM. While you wrote " + TurbMod)

    # Compute turbulence scale parameter Annex C3 of IEC 61400-1-2019
    # Longitudinal
    if HH <= 60:
        L_1 = .7 * HH
    else:
        L_1 = 42.
    sigma_u = sigma_1
    L_u = 8.1 * L_1
    # Lateral
    sigma_v =  0.8 * sigma_1
    L_v = 2.7 * L_1 
    # Upward
    sigma_w =  0.5 * sigma_1
    L_w = 0.66 * L_1 

    U = (4*L_u/V_ref)*sigma_u**2/((1+6*f*L_u/V_ref)**(5./3.))
    V = (4*L_v/V_ref)*sigma_v**2/((1+6*f*L_v/V_ref)**(5./3.))
    W = (4*L_w/V_ref)*sigma_w**2/((1+6*f*L_w/V_ref)**(5./3.))

    kappa = 12 * np.sqrt((f/V_ref)**2 + (0.12 / L_u)**2)

    Rot = (2*U / (R * kappa)**3) * \
        (modstruve(1,2*R*kappa) - iv(1,2*R*kappa) - 2/np.pi + \
            R*kappa * (-2 * modstruve(-2,2*R*kappa) + 2 * iv(2,2*R*kappa) + 1) )

    # set NaNs to 0
    Rot[np.isnan(Rot)] = 0
    
    # Formulas from Section 6.3 of IEC 61400-1-2019
    # S_1_f = 0.05 * sigma_1**2. * (L_1 / V_hub) ** (-2./3.) * f **(-5./3)
    # S_2_f = S_3_f = 4. / 3. * S_1_f
    # sigma_k = np.sqrt(np.trapz(S_1_f, f))
    # print(sigma_k)
    # print(sigma_u)

    return U, V, W, Rot

if __name__=="__main__":

    # Average wind speed at hub height
    V_hub = 6.
    # Wind turbine hub height
    HH = 150.
    # IEC Turbine Wind Speed Class, can be I, II, or III
    Class = 'I'
    # IEC Turbine Wind Turbulence Category, can be A, B, or C
    Categ = 'B'
    # Wind turbulence model, it can be NTM = normal turbulence, ETM = extreme turbulence, or EWM = extreme wind
    TurbMod = 'NTM'
    # Rotor radius, used to calculate rotor average wind spectrum
    R = 120.
    # Frequency range
    f=np.arange(0.0015873015873015873015873015873, 20.00001, 0.0015873015873015873015873015873)
    f=np.logspace(-2,1)

    U, V, W, S_r = IECKaimal(f, V_hub, HH, Class, Categ, TurbMod, R)




    # load turbsim file and compute spectra of rot avg wind speed
    # ts_file = TurbSimFile('/Users/dzalkind/Tools/WEIS-2/wind/IEA-15MW/level3_NTM_U12.000000_Seed600.0.bts')
    ts_filename   = '/Users/dzalkind/Tools/WEIS-2/wind/IEA-15MW/level3_NTM_U6.000000_Seed600.0.bts'
    ts_file = TurbSimFile(ts_filename)
    ts_file.compute_rot_avg(120)

    dist = {}
    dist['Time'] = ts_file['t']
    dist['Wind'] = ts_file['rot_avg'][0]
    dist['HH'] = ts_file['u'][0,:,12,12]


    # y,fq= op.plot_spectral([dist],[('Wind',0)],averaging='Welch',averaging_window='Hamming')
    fq, y, _ = spectral.fft_wrap(
                    dist['Time'], dist['Wind'], averaging='Welch', averaging_window='Hamming', output_type='psd')

    plt.plot(fq, (y), '-', label = 'U_TS')

    plt.plot(f, U, '-', label = 'U')
    plt.plot(f, V, '--', label = 'V')
    plt.plot(f, W, ':', label = 'W')
    plt.plot(f, S_r, label = 'S_rot')
    plt.yscale('log')
    plt.xscale('log')

    plt.xlim([1e-2,10])
    plt.grid('True')

    plt.xlabel('Freq. (Hz)')
    plt.ylabel('PSD')


    # How much energy are we capturing with f < 0.3
    if False:
        f = f[~np.isnan(S_r)]
        s_r = np.sqrt(S_r[~np.isnan(S_r)])

        tot_eng = np.trapz(s_r,f)

        eng = np.array([np.trapz(s_r[:i],f[:i]) for i in range(len(f))]) / tot_eng
        
        plt.plot(f,eng * 1e2,label='Perc. Energy')

    plt.legend()
    plt.show() 