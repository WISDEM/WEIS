import numpy as np
import matplotlib.pyplot as plt
from weis.aeroelasticse.pyIECWind import pyIECWind_extreme

def IECKaimal(f, V_ref, HH, Class, Categ, TurbMod):
    
    ###### Initialize IEC Wind parameters #######
    iec_wind = pyIECWind_extreme()
    iec_wind.z_hub = HH
    iec_wind.Turbine_Class = Class
    iec_wind.Turbulence_Class = Categ
    iec_wind.setup()

    # Compute wind turbulence standard deviation (invariant with height)
    if TurbMod == 'NTM':
        sigma_1 = iec_wind.NTM(V_ref)
    elif TurbMod == 'ETM':
        sigma_1 = iec_wind.ETM(V_ref)
    elif TurbMod == 'EWM':
        sigma_1 = iec_wind.EWM(V_ref)
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

    U= (4*L_u/V_ref)*sigma_u**2/((1+6*f*L_u/V_ref)**(5./3.))
    V= (4*L_v/V_ref)*sigma_v**2/((1+6*f*L_v/V_ref)**(5./3.))
    W= (4*L_w/V_ref)*sigma_w**2/((1+6*f*L_w/V_ref)**(5./3.))
    
    # Formulas from Section 6.3 of IEC 61400-1-2019
    # S_1_f = 0.05 * sigma_1**2. * (L_1 / V_hub) ** (-2./3.) * f **(-5./3)
    # S_2_f = S_3_f = 4. / 3. * S_1_f
    # sigma_k = np.sqrt(np.trapz(S_1_f, f))
    # print(sigma_k)
    # print(sigma_u)

    return U, V, W

if __name__=="__main__":

    # Average wind speed at hub height
    V_hub = 10.
    # Wind turbine hub height
    HH = 100.
    # IEC Turbine Wind Speed Class, can be I, II, or III
    Class = 'I'
    # IEC Turbine Wind Turbulence Category, can be A, B, or C
    Categ = 'A'
    # Wind turbulence model, it can be NTM = normal turbulence, ETM = extreme turbulence, or EWM = extreme wind
    TurbMod = 'NTM'
    # Frequency range
    f=np.arange(0.0015873015873015873015873015873, 20.00001, 0.0015873015873015873015873015873)


    U, V, W = IECKaimal(f, V_hub, HH, Class, Categ, TurbMod)

    plt.plot(f, U, '-', label = 'U')
    plt.plot(f, V, '--', label = 'V')
    plt.plot(f, W, ':', label = 'W')
    plt.yscale('log')
    plt.legend()
    plt.show() 