# RAFT's rotor class

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


from raft.pyIECWind         import pyIECWind_extreme

from scipy.interpolate      import PchipInterpolator
from scipy.special          import modstruve, iv


from raft.helpers                import rotationMatrix

from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

import pickle

if False:
    thrust_psd = pickle.load( open( "/Users/dzalkind/Tools/RAFT/designs/rotors/thrust_psd.p", "rb" ) )

'''
try:
    import ccblade        as CCBlade, CCAirfoil  # for cloned ccblade
except:
    import wisdem.ccblade as CCblade, CCAirfoil  # for conda install wisdem
'''

# global constants
raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
rad2deg = 57.2958
rpm2radps = 0.1047

# a class for the rotor structure, aerodynamics, and control in RAFT
class Rotor:

    def __init__(self, turbine, w, old=False):
        '''
        >>>> add mean offset parameters add move this to runCCBlade<<<<
        '''

        # Should inherit these from raft_model or _env?
        self.w = np.array(w)
               
        self.Zhub       = turbine['Zhub']           # [m]
        self.shaft_tilt = turbine['shaft_tilt']     # [deg]        
        self.overhang   = turbine['overhang'] 
        self.R_rot      = turbine['blade']['Rtip']  # rotor radius [m]
        #yaw  = 0        

        # Set some turbine params, this can come from WEIS/WISDEM or an external input
        if old:
            self.rot_from_weis = yaml.load(open(os.path.join(raft_dir,'designs/rotors/IEA-15MW_WEIS.yaml'),'r'))
            self.Uhub = np.array(self.rot_from_weis['wt_ops']['v'])
            self.Omega_rpm = np.array(self.rot_from_weis['wt_ops']['omega_op']) / rpm2radps
            self.pitch_deg = np.array(self.rot_from_weis['wt_ops']['pitch_op']) * rad2deg
            self.I_drivetrain = 3.2e8
        else:   # planning on this option going forward
            #self.Uhub = np.array([14.])
            #self.Omega_rpm = np.array([7.56])
            #self.pitch_deg = np.array([13.78])
            self.Uhub      = np.array(turbine['wt_ops']['v'])
            self.Omega_rpm = np.array(turbine['wt_ops']['omega_op']) / rpm2radps
            self.pitch_deg = np.array(turbine['wt_ops']['pitch_op']) * rad2deg
            self.I_drivetrain = float(turbine['I_drivetrain'])


        # Set default control gains
        self.kp_0 = np.zeros_like(self.Uhub)
        self.ki_0 = np.zeros_like(self.Uhub)       
        self.k_float = 0 # np.zeros_like(self.Uhub), right now this is a single value, but this may change    <<< what is this?


        # Set CCBlade flags
        tiploss = True # Tip loss model True/False
        hubloss = True # Hub loss model, True/False
        wakerotation = True # Wake rotation, True/False
        usecd = True # Use drag coefficient within BEMT, True/False

        # Set discretization parameters
        nSector = 4 # [-] - number of equally spaced azimuthal positions where CCBlade should be interrogated. The results are averaged across the n positions. 4 is a good first guess
        n_span = 30 # [-] - number of blade stations along span
        grid = np.linspace(0., 1., n_span) # equally spaced grid along blade span, root=0 tip=1


        # ----- AIRFOIL STUFF ------
        n_aoa = 200 # [-] - number of angles of attack to discretize airfoil polars - MUST BE MULTIPLE OF 4
        n_af = len(turbine["airfoils"])
        af_used     = [ b for [a,b] in turbine['blade']["airfoils"] ]
        af_position = [ a for [a,b] in turbine['blade']["airfoils"] ]
        #af_used     = turbine['blade']["airfoils"]["labels"]
        #af_position = turbine['blade']["airfoils"]["grid"]
        n_af_span = len(af_used)
        
        # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
        aoa = np.unique(np.hstack([np.linspace(-180, -30, int(n_aoa/4.0 + 1)), 
                                   np.linspace( -30,  30, int(n_aoa/2.0),),
                                   np.linspace(  30, 180, int(n_aoa/4.0 + 1))]))

        af_name = n_af * [""]
        r_thick = np.zeros(n_af)
        for i in range(n_af):
            af_name[i] = turbine["airfoils"][i]["name"]
            r_thick[i] = turbine["airfoils"][i]["relative_thickness"]
            

        cl = np.zeros((n_af, n_aoa, 1))
        cd = np.zeros((n_af, n_aoa, 1))
        cm = np.zeros((n_af, n_aoa, 1))

        # Interp cl-cd-cm along predefined grid of angle of attack
        for i in range(n_af):
        
            #cl[i, :, 0] = np.interp(aoa, turbine["airfoils"][i]["alpha"], turbine["airfoils"][i]["c_l"])
            #cd[i, :, 0] = np.interp(aoa, turbine["airfoils"][i]["alpha"], turbine["airfoils"][i]["c_d"])
            #cm[i, :, 0] = np.interp(aoa, turbine["airfoils"][i]["alpha"], turbine["airfoils"][i]["c_m"])

            polar_table = np.array(turbine["airfoils"][i]['data'])
            
            cl[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,1])
            cd[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,2])
            cm[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,3])

            #plt.figure()
            #plt.plot(polar_table[:,0], polar_table[:,1])
            #plt.plot(polar_table[:,0], polar_table[:,2])
            #plt.title(af_name[i])
            
            if abs(cl[i, 0, 0] - cl[i, -1, 0]) > 1.0e-5:
                print("WARNING: Ai " + af_name[i] + " has the lift coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
                cl[i, 0, 0] = cl[i, -1, 0]
            if abs(cd[i, 0, 0] - cd[i, -1, 0]) > 1.0e-5:
                print("WARNING: Airfoil " + af_name[i] + " has the drag coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
                cd[i, 0, 0] = cd[i, -1, 0]
            if abs(cm[i, 0, 0] - cm[i, -1, 0]) > 1.0e-5:
                print("WARNING: Airfoil " + af_name[i] + " has the moment coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
                cm[i, 0, 0] = cm[i, -1, 0]


        # Interpolate along blade span using a pchip on relative thickness
        r_thick_used = np.zeros(n_af_span)
        cl_used = np.zeros((n_af_span, n_aoa, 1))
        cd_used = np.zeros((n_af_span, n_aoa, 1))
        cm_used = np.zeros((n_af_span, n_aoa, 1))

        for i in range(n_af_span):
            for j in range(n_af):
                if af_used[i] == af_name[j]:
                    r_thick_used[i] = r_thick[j]
                    cl_used[i, :, :] = cl[j, :, :]
                    cd_used[i, :, :] = cd[j, :, :]
                    cm_used[i, :, :] = cm[j, :, :]
                    break

        # Pchip does have an associated derivative method built-in:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.PchipInterpolator.derivative
        spline = PchipInterpolator
        rthick_spline = spline(af_position, r_thick_used)
        # GB: HAVE TO TALK TO PIETRO ABOUT THIS
        #r_thick_interp = rthick_spline(grid[1:-1])
        r_thick_interp = rthick_spline(grid)

        # Spanwise interpolation of the airfoil polars with a pchip
        r_thick_unique, indices = np.unique(r_thick_used, return_index=True)
        cl_spline = spline(r_thick_unique, cl_used[indices, :, :])
        self.cl_interp = np.flip(cl_spline(np.flip(r_thick_interp)), axis=0)
        cd_spline = spline(r_thick_unique, cd_used[indices, :, :])
        self.cd_interp = np.flip(cd_spline(np.flip(r_thick_interp)), axis=0)
        cm_spline = spline(r_thick_unique, cm_used[indices, :, :])
        self.cm_interp = np.flip(cm_spline(np.flip(r_thick_interp)), axis=0)
                
        self.aoa = aoa
        
        # split out blade geometry info from table 
        geometry_table = np.array(turbine['blade']['geometry'])
        blade_r         = geometry_table[:,0]
        blade_chord     = geometry_table[:,1]
        blade_theta     = geometry_table[:,2]
        blade_precurve  = geometry_table[:,3]
        blade_presweep  = geometry_table[:,4]
        
        af = []
        for i in range(self.cl_interp.shape[0]):
            af.append(CCAirfoil(self.aoa, [], self.cl_interp[i,:,:],self.cd_interp[i,:,:],self.cm_interp[i,:,:]))
        
        self.ccblade = CCBlade(
            blade_r,                        # (m) locations defining the blade along z-axis of blade coordinate system
            blade_chord,                    # (m) corresponding chord length at each section
            blade_theta,                    # (deg) corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---positive twist decreases angle of attack.
            af,                             # CCAirfoil object
            turbine['Rhub'],                # (m) radius of hub
            turbine['blade']['Rtip'],       # (m) radius of tip
            turbine['nBlades'],             # number of blades
            turbine['rho_air'],             # (kg/m^3) freestream fluid density
            turbine['mu_air'],              # (kg/m/s) dynamic viscosity of fluid
            turbine['precone'],             # (deg) hub precone angle
            self.shaft_tilt,                # (deg) hub tilt angle
            0.0,                            # (deg) nacelle yaw angle
            turbine['shearExp'],            # shear exponent for a power-law wind profile across hub
            turbine['Zhub'],                # (m) hub height used for power-law wind profile.  U = Uref*(z/hubHt)**shearExp
            nSector,                        # number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to 1 if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
            blade_precurve,                 # (m) location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            turbine['blade']['precurveTip'],# (m) location of blade pitch axis in x-direction at the tip (analogous to Rtip)
            blade_presweep,                 # (m) location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            turbine['blade']['presweepTip'],# (m) location of blade pitch axis in y-direction at the tip (analogous to Rtip)
            tiploss=tiploss,                # if True, include Prandtl tip loss model
            hubloss=hubloss,                # if True, include Prandtl hub loss model
            wakerotation=wakerotation,      # if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
            usecd=usecd,                    # If True, use drag coefficient in computing induction factors (always used in evaluating distributed loads from the induction factors).
            derivatives=True,               # if True, derivatives along with function values will be returned for the various methods
        )

        # pull control gains out of dictionary
        self.setControlGains(turbine)


    def runCCBlade(self, Uhub, ptfm_pitch=0, yaw_misalign=0):
        '''This performs a single CCBlade evaluation at specified conditions.
        
        ptfm_pitch
            mean platform pitch angle to be included in rotor tilt angle [rad]
        yaw_misalign
            turbine yaw misalignment angle [deg]
        '''
        
        # find turbine operating point at the provided wind speed
        Omega_rpm = np.interp(Uhub, self.Uhub, self.Omega_rpm)  # rotor speed [rpm]
        pitch_deg = np.interp(Uhub, self.Uhub, self.pitch_deg)  # blade pitch angle [deg]
        
        # adjust rotor angles based on provided info (I think this intervention in CCBlade should work...)
        self.ccblade.tilt = np.deg2rad(self.shaft_tilt) + ptfm_pitch
        self.ccblade.yaw  = np.deg2rad(yaw_misalign)
        
        # evaluate aero loads and derivatives with CCBlade
        loads, derivs = self.ccblade.evaluate(Uhub, Omega_rpm, pitch_deg, coefficients=True)

        # organize and save the relevant outputs...
        outputs = {}
        
        outputs["P"] = loads["P"]
        outputs["Mb"] = loads["Mb"]
        outputs["CP"] = loads["CP"]
        outputs["CMb"] = loads["CMb"]
        outputs["Fhub"] = np.array( [loads["T" ][0], loads["Y"  ][0], loads["Z"  ][0]])
        outputs["Mhub"] = np.array( [loads["Q" ][0], loads["My" ][0], loads["Mz" ][0]])
        outputs["CFhub"] = np.array([loads["CT"][0], loads["CY" ][0], loads["CZ" ][0]])
        outputs["CMhub"] = np.array([loads["CQ"][0], loads["CMy"][0], loads["CMz"][0]])


        print(f"Wind speed: {Uhub} m/s, Aerodynamic power coefficient: {loads['CP'][0]:4.3f}")

        J={} # Jacobian/derivatives

        dP = derivs["dP"]
        J["P", "r"] = dP["dr"]
        # J["P", "chord"] = dP["dchord"]
        # J["P", "theta"] = dP["dtheta"]
        # J["P", "Rhub"] = np.squeeze(dP["dRhub"])
        # J["P", "Rtip"] = np.squeeze(dP["dRtip"])
        # J["P", "hub_height"] = np.squeeze(dP["dhubHt"])
        # J["P", "precone"] = np.squeeze(dP["dprecone"])
        # J["P", "tilt"] = np.squeeze(dP["dtilt"])
        # J["P", "yaw"] = np.squeeze(dP["dyaw"])
        # J["P", "shearExp"] = np.squeeze(dP["dshear"])
        # J["P", "V_load"] = np.squeeze(dP["dUinf"])
        # J["P", "Omega_load"] = np.squeeze(dP["dOmega"])
        # J["P", "pitch_load"] = np.squeeze(dP["dpitch"])
        # J["P", "precurve"] = dP["dprecurve"]
        # J["P", "precurveTip"] = dP["dprecurveTip"]
        # J["P", "presweep"] = dP["dpresweep"]
        # J["P", "presweepTip"] = dP["dpresweepTip"]

        dQ = derivs["dQ"]
        J["Q","Uhub"]      = np.atleast_1d(np.diag(dQ["dUinf"]))
        J["Q","pitch_deg"] = np.atleast_1d(np.diag(dQ["dpitch"]))
        J["Q","Omega_rpm"] = np.atleast_1d(np.diag(dQ["dOmega"]))

        dT = derivs["dT"]
        J["T","Uhub"]      = np.atleast_1d(np.diag(dT["dUinf"]))
        J["T","pitch_deg"] = np.atleast_1d(np.diag(dT["dpitch"]))
        J["T","Omega_rpm"] = np.atleast_1d(np.diag(dT["dOmega"]))

        # dT = derivs["dT"]
        # .J["Fhub", "r"][0,:] = dT["dr"]     # 0 is for thrust force, 1 would be y, 2 z
        # .J["Fhub", "chord"][0,:] = dT["dchord"]
        # .J["Fhub", "theta"][0,:] = dT["dtheta"]
        # .J["Fhub", "Rhub"][0,:] = np.squeeze(dT["dRhub"])
        # .J["Fhub", "Rtip"][0,:] = np.squeeze(dT["dRtip"])
        # .J["Fhub", "hub_height"][0,:] = np.squeeze(dT["dhubHt"])
        # .J["Fhub", "precone"][0,:] = np.squeeze(dT["dprecone"])
        # .J["Fhub", "tilt"][0,:] = np.squeeze(dT["dtilt"])
        # .J["Fhub", "yaw"][0,:] = np.squeeze(dT["dyaw"])
        # .J["Fhub", "shearExp"][0,:] = np.squeeze(dT["dshear"])
        # .J["Fhub", "V_load"][0,:] = np.squeeze(dT["dUinf"])
        # .J["Fhub", "Omega_load"][0,:] = np.squeeze(dT["dOmega"])
        # .J["Fhub", "pitch_load"][0,:] = np.squeeze(dT["dpitch"])
        # .J["Fhub", "precurve"][0,:] = dT["dprecurve"]
        # .J["Fhub", "precurveTip"][0,:] = dT["dprecurveTip"]
        # .J["Fhub", "presweep"][0,:] = dT["dpresweep"]
        # .J["Fhub", "presweepTip"][0,:] = dT["dpresweepTip"]

        self.J = J

        return loads, derivs        

    def calcAeroContributions(self, case):
        '''Calculates stiffness, damping, added mass, and excitation coefficients
        from rotor aerodynamics. Results are w.r.t. nonrotating hub reference frame
        and assume constant rotor speed and no controls.
        '''
        
        Uinf = case['wind_speed']  # inflow wind speed (m/s) 

        # extract derivatives of interest, interpolated for the current wind speed
        dT_dU  = np.interp(Uinf, self.Uhub, self.J["T", "Uhub"     ])
        #dT_dOm = np.interp(Uinf, self.Uhub, self.J["T", "Omega_rpm"])
        #dT_dPi = np.interp(Uinf, self.Uhub, self.J["T", "pitch_deg"])
        #dQ_dU  = np.interp(Uinf, self.Uhub, self.J["Q", "Uhub"     ])
        #dQ_dOm = np.interp(Uinf, self.Uhub, self.J["Q", "Omega_rpm"])
        #dQ_dPi = np.interp(Uinf, self.Uhub, self.J["Q", "pitch_deg"])
        # wish list
        # dMy_dU  = np.interp(Uinf, self.Uhub, self.J["My", "Uhub"     ])  # overturning moment about hub
        # dMy_dShearExp =
        # ...

        # coefficients to be filled in
        nw = len(self.w)
        A_aero = np.zeros([6,6,nw])                     # added mass
        B_aero = np.zeros([6,6,nw])                     # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF

        # calculate hub aero coefficients (in nonrotating hub reference frame) - assuming rigid body and no control to start with
        B_aero[0,0,:] += dT_dU                          # surge damping

        # calculate wind excitation force/moment spectra (in nonrotating hub reference frame)
        #for i in range(len(self.w)):                    # loop through each frequency component
        #    F_aero[0,i] = U_amplitude[i]*dT_dU          # surge excitation
        #    F_aero[7,i] = U_amplitude*dQ_dU            # rotor torque excitation

        # calculate steady aero forces and moments
        F_aero0 = np.hstack((self.outputs["Fhub"],self.outputs["Mhub"]))

        return A_aero, B_aero, C_aero, F_aero0, F_aero


    def setControlGains(self,turbine):
        '''
        Use flipped sign version of ROSCO
        '''

        # Convert gain-scheduling wrt pitch to wind speed
        pc_angles = np.array(turbine['pitch_control']['GS_Angles']) * rad2deg
        self.kp_0 = np.interp(self.pitch_deg,pc_angles,turbine['pitch_control']['GS_Kp'],left=0,right=0)
        self.ki_0 = np.interp(self.pitch_deg,pc_angles,turbine['pitch_control']['GS_Ki'],left=0,right=0)
        self.k_float = -turbine['pitch_control']['Fl_Kp']

        # Torque control
        self.kp_tau = -turbine['torque_control']['VS_KP']
        self.ki_tau = -turbine['torque_control']['VS_KI']
        self.Ng     = turbine['gear_ratio']
            


    def calcAeroServoContributions(self, case, ptfm_pitch=0):
        '''Calculates stiffness, damping, added mass, and excitation coefficients
        from rotor aerodynamics coupled with turbine controls.
        Results are w.r.t. nonrotating hub reference frame.
        
         ptfm_pitch
            mean platform pitch angle to be included in rotor tilt angle [rad]
        '''
        
        loads, derivs = self.runCCBlade(case['wind_speed'], ptfm_pitch=ptfm_pitch, yaw_misalign=case['yaw_misalign'])
        
        Uinf = case['wind_speed']  # inflow wind speed (m/s) <<< eventually should be consistent with rest of RAFT
        
        # >>>> make wind spectrum from info in case <<<<<   
            
        # extract derivatives of interest
        dT_dU  = np.atleast_1d(np.diag(derivs["dT"]["dUinf"]))
        dT_dOm = np.atleast_1d(np.diag(derivs["dT"]["dOmega"])) / rpm2radps
        dT_dPi = np.atleast_1d(np.diag(derivs["dT"]["dpitch"])) * rad2deg
        dQ_dU  = np.atleast_1d(np.diag(derivs["dQ"]["dUinf"]))
        dQ_dOm = np.atleast_1d(np.diag(derivs["dQ"]["dOmega"])) / rpm2radps
        dQ_dPi = np.atleast_1d(np.diag(derivs["dQ"]["dpitch"])) * rad2deg

        # Pitch control gains at Uinf (Uinf), flip sign to translate ROSCO convention to this one
        kp_U    = -np.interp(Uinf, self.Uhub, self.kp_0) 
        ki_U    = -np.interp(Uinf, self.Uhub, self.ki_0) 

        # Torque control gains, need to get these from somewhere
        kp_tau = self.kp_tau * (kp_U == 0)  #     -38609162.66552     ! VS_KP				- Proportional gain for generator PI torque controller [1/(rad/s) Nm]. (Only used in the transitional 2.5 region if VS_ControlMode =/ 2)
        ki_tau = self.kp_tau  * (kp_U == 0)   #    -4588245.18720      ! VS_KI	
        
        a_aer = np.zeros_like(self.w)
        b_aer = np.zeros_like(self.w)
        C   = np.zeros_like(self.w,dtype=np.complex_)
        C2  = np.zeros_like(self.w,dtype=np.complex_)
        D   = np.zeros_like(self.w,dtype=np.complex_)
        E   = np.zeros_like(self.w,dtype=np.complex_)

        # Roots of characteristic equation, helps w/ debugging
        p = np.array([-self.I_drivetrain, (dQ_dOm + kp_U * dQ_dPi - self.Ng * kp_tau), ki_U* dQ_dPi - self.Ng * ki_tau])
        r = np.roots(p)

        for iw, omega in enumerate(self.w):
            
            # Denominator of control transfer function
            D[iw] = self.I_drivetrain * omega**2 + (dQ_dOm + kp_U * dQ_dPi - self.Ng * kp_tau) * 1j * omega + ki_U* dQ_dPi - self.Ng * ki_tau

            # control transfer function
            C[iw] = 1j * omega * (dQ_dU - self.k_float * dQ_dPi / self.Zhub) / D[iw]

            # Thrust transfer function
            E[iw] = ((dT_dOm + kp_U * dT_dPi) * 1j * omega + ki_U * dT_dPi )

            # alternative for debugging
            C2[iw] = C[iw] / (1j * omega)

            # Complex aero damping
            T = 1j * omega * (dT_dU - self.k_float * dT_dPi / self.Zhub) - ( E[iw] * C[iw])

            # Aerodynamic coefficients
            a_aer[iw] = -(1/omega**2) * np.real(T)
            b_aer[iw] = (1/omega) * np.imag(T)
        
        # coefficients to be filled in
        nw = len(self.w)
        A_aero = np.zeros([6,6,nw])                     # added mass
        B_aero = np.zeros([6,6,nw])                     # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF
        
        # calculate contribution to system matrices      
        A_aero[0,0,:] = a_aer
        B_aero[0,0,:] = b_aer
                
        # calculate steady aero forces and moments
        F_aero0 = np.array([loads["T" ][0], loads["Y"  ][0], loads["Z"  ][0],
                            loads["My" ][0], loads["Q" ][0], loads["Mz" ][0] ])
                            
        # calculate wind excitation force/moment spectra
        _,_,_,S_rot = self.IECKaimal(case)

        V_w = np.sqrt(S_rot)
        T_0 = loads["T" ][0]
        T_w1 = dT_dU * V_w
        T_w2 = (E * C * V_w) / (1j * self.w)

        T_ext = T_w1 + T_w2

        # print('here')
        if False:
            plt.plot(self.w, V_w, label = 'S_rot')
            plt.yscale('log')
            plt.xscale('log')

            plt.xlim([1e-2,10])
            plt.grid('True')

            plt.xlabel('Freq. (Hz)')
            plt.ylabel('PSD')


            plt.plot(thrust_psd.fq_0 * 2 * np.pi,thrust_psd.psd_0)
            plt.plot(self.w, np.abs(T_ext))
            # plt.plot(self.w, abs(T_w2))

            plt.show()




        #for i in range(nw):                             # loop through each frequency component
        #    F_aero[0,i] = U_amplitude[i]*dT_dU             # surge excitation
        #    F_aero[4,i] = U_amplitude[i]*dT_dU*Zhub        # pitch excitation
        
        
        return A_aero, B_aero, C_aero, F_aero0, F_aero, a_aer, b_aer #  B_aero, C_aero, F_aero0, F_aero
        
        
    def plot(self, ax, r_ptfm=[0,0,0], R_ptfm=np.eye(3), azimuth=0):
        '''Draws the rotor on the passed axes, considering optional platform offset and rotation matrix, and rotor azimuth angle'''

        # ----- blade geometry ----------

        m = len(self.ccblade.chord)

        # lists to be filled with coordinates for plotting
        X = []
        Y = []
        Z = []        
        
        # generic airfoil for now
        afx = np.array([ 0.0 , -0.16, 0.0 ,  0.0 ])
        afy = np.array([-0.25,  0.  , 0.75, -0.25])
        npts = len(afx)
        
        # should add real airfoil shapes, and twist     
        for i in range(m):
            for j in range(npts):
                X.append(self.ccblade.chord[i]*afx[j])
                Y.append(self.ccblade.chord[i]*afy[j])
                Z.append(self.ccblade.r[i])            
                #X.append(self.ccblade.chord[i+1]*afx[j])
                #Y.append(self.ccblade.chord[i+1]*afy[j])
                #Z.append(self.ccblade.r[i+1]) 
                
        P = np.array([X, Y, Z])
        
        # ----- rotation matricse ----- 
        # (blade pitch would be a -rotation about local z)
        R_precone = rotationMatrix(0, -self.ccblade.precone, 0)  
        R_azimuth = [rotationMatrix(azimuth + azi, 0, 0) for azi in 2*np.pi/3.*np.arange(3)]
        R_tilt    = rotationMatrix(0, np.deg2rad(self.shaft_tilt), 0)   # # define x as along shaft downwind, y is same as ptfm y
        
        # ----- transform coordinates -----
        for ib in range(3):
        
            P2 = np.matmul(R_precone, P)
            P2 = np.matmul(R_azimuth[ib], P2)
            P2 = np.matmul(R_tilt, P2)
            P2 = P2 + np.array([-self.overhang, 0, self.Zhub])[:,None] # PRP to tower-shaft intersection point
            P2 = np.matmul(R_ptfm, P2) + np.array(r_ptfm)[:,None]
          
            # drawing airfoils                            
            #for ii in range(m-1):
            #    ax.plot(P2[0, npts*ii:npts*(ii+1)], P2[1, npts*ii:npts*(ii+1)], P2[2, npts*ii:npts*(ii+1)])  
            # draw outline
            ax.plot(P2[0, 0:-1:npts], P2[1, 0:-1:npts], P2[2, 0:-1:npts], 'k') # leading edge  
            ax.plot(P2[0, 2:-1:npts], P2[1, 2:-1:npts], P2[2, 2:-1:npts], 'k')  # trailing edge
            
            
        #for j in range(m):
        #    linebit.append(ax.plot(Xs[j::m], Ys[j::m], Zs[j::m]            , color='k'))  # station rings
        #
        #return linebit
        
    def IECKaimal(self, case):        # 
        # Set inputs (f, V_ref, HH, Class, Categ, TurbMod, R)
        f = self.w / 2 / np.pi    # frequency in Hz
        HH = self.Zhub
        R = self.R_rot
        V_ref = case['wind_speed']
        
        ###### Initialize IEC Wind parameters #######
        iec_wind = pyIECWind_extreme()
        iec_wind.z_hub = HH
        
        if isinstance(case['turbulence'],str):
            # If a string, the options are I, II, III, IV
            Class = ''
            for char in case['turbulence']:
                if char == 'I' or char == 'V':
                    Class += char
                else:
                    break
            
            if not Class:
                raise Exception("Turbulence class must start with I, II, III, or IV, while you wrote " + case['turbulence'])
            else:
                Categ = char
                iec_wind.Turbulence_Class = Categ

            try:
                TurbMod = case['turbulence'].split('_')[1]
            except:
                raise Exception("Error reading the turbulence model. You wrote " + case['turbulence'])

            iec_wind.Turbine_Class = Class
        
        # set things up (use default values if not specified in the above)
        iec_wind.setup()
        
        # Can set iec_wind.I_ref here if wanted, NTM used then
        if isinstance(case['turbulence'],float):
            iec_wind.I_ref = case['turbulence']    # this overwrites the value set in setup method
            TurbMod = 'NTM'

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

if __name__=='__main__':
    fname_design = os.path.join(raft_dir,'designs/VolturnUS-S.yaml')
    # from raft.runRAFT import loadTurbineYAML
    # turbine = loadTurbineYAML(fname_design)

    # open the design YAML file and parse it into a dictionary for passing to raft
    with open(fname_design) as file:
        design = yaml.load(file, Loader=yaml.FullLoader)

    # Turbine rotor
    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp'] = design['site']['shearExp']

    # Set up thrust verification cases
    print('here')
    UU = [8,12,16]
    for i_case in range(len(UU)):
        design['cases']['data'][i_case][0] = UU[i_case]     # wind speed
        design['cases']['data'][i_case][1] = 0   # wind heading
        design['cases']['data'][i_case][2] = 'IB_NTM'    # turbulence
        design['cases']['data'][i_case][3] = 'operating'    # turbulence
        design['cases']['data'][i_case][4] = 0    # turbulence
        design['cases']['data'][i_case][5] = 'JONSWAP'    # turbulence
        design['cases']['data'][i_case][6] = 8    # turbulence
        design['cases']['data'][i_case][7] = 2    # turbulence
        design['cases']['data'][i_case][8] = 0    # turbulence


    rr = Rotor(design['turbine'],np.linspace(0.05,3)) #, old=True)
    # rr.runCCBlade()
    # rr.setControlGains(design['turbine'])  << now called in Rotor init


    if True:

        # loop through each case
        nCases = len(design['cases']['data'])
        for iCase in range(nCases):
        
            print("  Running case")
            print(design['cases']['data'][iCase])
        
            # form dictionary of case parameters
            case = dict(zip( design['cases']['keys'], design['cases']['data'][iCase]))   

            rr.calcAeroServoContributions(case)

    if True:

        UU = np.linspace(4,24)
        a_aer_U = np.zeros_like(UU)
        b_aer_U = np.zeros_like(UU)

        for iU, Uinf in enumerate(UU):
            # rr.V = Uinf
            case['wind_speed'] = Uinf
            _,_,_,_,_, a_aer, b_aer = rr.calcAeroServoContributions(case)

            a_aer_U[iU] = np.interp(2 * np.pi / 30, rr.w, a_aer)
            b_aer_U[iU] = np.interp(2 * np.pi / 30, rr.w, b_aer)


        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(2,1)


        ax1[0].plot(UU,a_aer_U)
        ax1[0].set_ylabel('a_aer @ 30 sec.')
        ax1[0].grid(True)

        ax1[1].plot(UU,b_aer_U)
        ax1[1].set_ylabel('b_aer @ 30 sec.')
        ax1[1].grid(True)

        ax1[1].set_xlabel('U (m/s)')

    plt.show()


    # ax1[0].plot(ww,a_aer)
    # ax1[0].set_ylabel('a_aer')

    # ax1[1].plot(ww,b_aer)
    # ax1[1].set_ylabel('b_aer')

    # fig1.legend(('gains * 0','gains * 1','gains * 2'))

    # ax1[1].set_xlabel('frequency (rad/s)')

    # ax2[0].plot(ww,np.abs(C))
    # ax2[0].set_ylabel('mag(C)')

    # ax2[1].plot(ww,np.angle(C))
    # ax2[1].set_ylabel('phase(C)')

