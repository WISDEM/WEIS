# RAFT's rotor class

import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import ccblade        as CCBlade  # for cloned ccblade
except:
    import wisdem.ccblade as CCblade  # for conda install wisdem



# a class for the rotor structure, aerodynamics, and control in RAFT
class Rotor:

    def __init__(self):
        '''
        '''
        
        # (not worrying about rotor structure/mass yet, just aero)
        
        

        # >>>> Pietro's CCblade input preparation code here? <<<<
        
        ccblade = CCBlade(
            r,                         # radius stations [m]
            chord,                     # chord length at each station [m]
            theta,                     # ...
            af,                        # airfoil data structure - polars, use WISDEM YAML
            Rhub,                      #
            Rtip,                      #
            B,                         #
            rho,                       #
            mu,                        #
            precone,                   # not in rotor block of WISDEM YAML
            tilt,                      # not in rotor block of WISDEM YAML
            yaw,                       #
            shearExp,                  #
            hub_height,                #
            nSector,                   #
            precurve,                  #
            precurveTip,               #
            tiploss=tiploss,           #
            hubloss=hubloss,           #
            wakerotation=wakerotation, #
            usecd=usecd,               #
            derivatives=True,          #
        )


    def runCCblade(self):
        '''
        '''

        # >>>> Peitro's code for running CCblade and extracting out the values and deriviatives here? <<<<
    
    
        #self.J = ...



    def calcAeroContributions(self, nw=0, U_amplitude=[])
        '''Calculates stiffness, damping, and added mass contributions to the system matrices
        from rotor aerodynamics. Results are w.r.t. platform reference point assuming rigid tower,
        constant rotor speed, and no controls.
        '''
        
        Uinf = 10.  # inflow wind speed (m/s) <<< eventually should be consistent with rest of RAFT
        Hhub = 100.
    
        # extract derivatives of interest, interpolated for the current wind speed
        dT_dU  = np.interp(Uinf, self.Uhub, self.J["T", "Uhub"     ])
        dT_dOm = np.interp(Uinf, self.Uhub, self.J["T", "Omega_rpm"])
        dT_dPi = np.interp(Uinf, self.Uhub, self.J["T", "pitch_deg"])
        dQ_dU  = np.interp(Uinf, self.Uhub, self.J["Q", "Uhub"     ])
        dQ_dOm = np.interp(Uinf, self.Uhub, self.J["Q", "Omega_rpm"])
        dQ_dPi = np.interp(Uinf, self.Uhub, self.J["Q", "pitch_deg"])        
        # wish list       
        # dMy_dU  = np.interp(Uinf, self.Uhub, self.J["My", "Uhub"     ])  # overturning moment about hub
        # dMy_dShearExp = 
        # ...
        
        # coefficients to be filled in
        A_aero = np.zeros([6,6])                        # added mass
        B_aero = np.zeros([6,6])                        # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF
        
        # calculate contribution to system matrices - assuming rigid body and no control to start with        
        B_aero[0,0] += dT_dU                            # surge damping
        B_aero[0,4] += dT_dU*Hhub                       # 
        B_aero[4,0] += dT_dU*Hhub                       # 
        B_aero[4,4] += dT_dU*Hhub**2                    # pitch damping
        
        # calculate wind excitation force/moment spectra
        for i in range(nw):                             # loop through each frequency component
            F_aero[0,i] = U_amplitude[i]*dT_dU             # surge excitation
            F_aero[4,i] = U_amplitude[i]*dT_dU*Hhub        # pitch excitation
            #F_aero[7,i] = U_amplitude*dQ_dU            # rotor torque excitation
        
        
        return A_aero, B_aero, C_aero, F_aero0, F_aero
        


    def calcAeroServoContributions(self, nw=0, U_amplitude=[])
        '''Calculates stiffness, damping, and added mass contributions to the system matrices
        from rotor aerodynamics coupled with turbine control system. 
        Results are w.r.t. platform reference point assuming rigid tower.
        '''
        
        Uinf = 10.  # inflow wind speed (m/s) <<< eventually should be consistent with rest of RAFT
        Hhub = 100.
    
        # extract derivatives of interest, interpolated for the current wind speed
        dT_dU  = np.interp(Uinf, self.Uhub, self.J["T", "Uhub"     ])
        dT_dOm = np.interp(Uinf, self.Uhub, self.J["T", "Omega_rpm"])
        dT_dPi = np.interp(Uinf, self.Uhub, self.J["T", "pitch_deg"])
        dQ_dU  = np.interp(Uinf, self.Uhub, self.J["Q", "Uhub"     ])
        dQ_dOm = np.interp(Uinf, self.Uhub, self.J["Q", "Omega_rpm"])
        dQ_dPi = np.interp(Uinf, self.Uhub, self.J["Q", "pitch_deg"])        
        
        # coefficients to be filled in
        A_aero = np.zeros([6,6])                        # added mass
        B_aero = np.zeros([6,6])                        # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF
        
        # calculate contribution to system matrices
        
        #...
        
        # calculate wind excitation force/moment spectra (will this change with control?)
        for i in range(nw):                             # loop through each frequency component
            F_aero[0,i] = U_amplitude[i]*dT_dU             # surge excitation
            F_aero[4,i] = U_amplitude[i]*dT_dU*Hhub        # pitch excitation        
        
        return A_aero, B_aero, C_aero, F_aero0, F_aero