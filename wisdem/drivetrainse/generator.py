"""generator.py
Created by Latha Sethuraman, Katherine Dykes. 
Copyright (c) NREL. All rights reserved.

Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

import openmdao.api as om
import numpy as np
import wisdem.drivetrainse.generator_models as gm
import wisdem.commonse.fileIO as fio

#----------------------------------------------------------------------------------------------

class Constraints(om.ExplicitComponent):
    """
    Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded.
    
    Parameters
    ----------
    u_allow_s : float, [m]
        
    u_as : float, [m]
        
    z_allow_s : float, [m]
        
    z_as : float, [m]
        
    y_allow_s : float, [m]
        
    y_as : float, [m]
        
    b_allow_s : float, [m]
        
    b_st : float, [m]
        
    u_allow_r : float, [m]
        
    u_ar : float, [m]
        
    y_allow_r : float, [m]
        
    y_ar : float, [m]
        
    z_allow_r : float, [m]
        
    z_ar : float, [m]
        
    b_allow_r : float, [m]
        
    b_arm : float, [m]
        
    TC1 : float, [m**3]
        
    TC2r : float, [m**3]
        
    TC2s : float, [m**3]
        
    B_g : float, [T]
        
    B_smax : float, [T]
        
    K_rad : float
        
    K_rad_LL : float
        
    K_rad_UL : float
        
    D_ratio : float
        
    D_ratio_LL : float
        
    D_ratio_UL : float
        
    
    Returns
    -------
    con_uas : float, [m]
        
    con_zas : float, [m]
        
    con_yas : float, [m]
        
    con_bst : float, [m]
        
    con_uar : float, [m]
        
    con_yar : float, [m]
        
    con_zar : float, [m]
        
    con_br : float, [m]
        
    TCr : float, [m**3]
        
    TCs : float, [m**3]
        
    con_TC2r : float, [m**3]
        
    con_TC2s : float, [m**3]
        
    con_Bsmax : float, [T]
        
    K_rad_L : float
        
    K_rad_U : float
        
    D_ratio_L : float
        
    D_ratio_U : float
        
    
    """
    def setup(self):
        self.add_input('u_allow_s', val=0.0, units='m')
        self.add_input('u_as', val=0.0, units='m')
        self.add_input('z_allow_s', val=0.0, units='m')
        self.add_input('z_as', val=0.0, units='m')
        self.add_input('y_allow_s', val=0.0, units='m')
        self.add_input('y_as', val=0.0, units='m')
        self.add_input('b_allow_s', val=0.0, units='m')
        self.add_input('b_st', val=0.0, units='m')
        self.add_input('u_allow_r', val=0.0, units='m')
        self.add_input('u_ar', val=0.0, units='m')
        self.add_input('y_allow_r', val=0.0, units='m')
        self.add_input('y_ar', val=0.0, units='m')
        self.add_input('z_allow_r', val=0.0, units='m')
        self.add_input('z_ar', val=0.0, units='m')
        self.add_input('b_allow_r', val=0.0, units='m')
        self.add_input('b_arm', val=0.0, units='m')
        self.add_input('TC1', val=0.0, units='m**3')
        self.add_input('TC2r', val=0.0, units='m**3')
        self.add_input('TC2s', val=0.0, units='m**3')
        self.add_input('B_g', val=0.0, units='T')
        self.add_input('B_smax', val=0.0, units='T')
        self.add_input('K_rad', val=0.0)
        self.add_input('K_rad_LL', val=0.0)
        self.add_input('K_rad_UL', val=0.0)
        self.add_input('D_ratio', val=0.0)
        self.add_input('D_ratio_LL', val=0.0)
        self.add_input('D_ratio_UL', val=0.0)

        self.add_output('con_uas', val=0.0, units='m')
        self.add_output('con_zas', val=0.0, units='m')
        self.add_output('con_yas', val=0.0, units='m')
        self.add_output('con_bst', val=0.0, units='m')
        self.add_output('con_uar', val=0.0, units='m')
        self.add_output('con_yar', val=0.0, units='m')
        self.add_output('con_zar', val=0.0, units='m')
        self.add_output('con_br', val=0.0, units='m')
        self.add_output('TCr', val=0.0, units='m**3')
        self.add_output('TCs', val=0.0, units='m**3')
        self.add_output('con_TC2r', val=0.0, units='m**3')
        self.add_output('con_TC2s', val=0.0, units='m**3')
        self.add_output('con_Bsmax', val=0.0, units='T')
        self.add_output('K_rad_L', val=0.0)
        self.add_output('K_rad_U', val=0.0)
        self.add_output('D_ratio_L', val=0.0)
        self.add_output('D_ratio_U', val=0.0)
        
        
    def compute(self, inputs, outputs):
        outputs['con_uas'] = inputs['u_allow_s'] - inputs['u_as']
        outputs['con_zas'] = inputs['z_allow_s'] - inputs['z_as']
        outputs['con_yas'] = inputs['y_allow_s'] - inputs['y_as']
        outputs['con_bst'] = inputs['b_allow_s'] - inputs['b_st']   #b_st={'units':'m'}
        outputs['con_uar'] = inputs['u_allow_r'] - inputs['u_ar']
        outputs['con_yar'] = inputs['y_allow_r'] - inputs['y_ar']
        outputs['con_TC2r'] = inputs['TC2s'] - inputs['TC1']
        outputs['con_TC2s'] = inputs['TC2s'] - inputs['TC1']
        outputs['con_Bsmax'] = inputs['B_g'] - inputs['B_smax']
        outputs['con_zar'] = inputs['z_allow_r'] - inputs['z_ar']
        outputs['con_br'] = inputs['b_allow_r'] - inputs['b_arm'] # b_r={'units':'m'}
        outputs['TCr'] = inputs['TC2r'] - inputs['TC1']
        outputs['TCs'] = inputs['TC2s'] - inputs['TC1']
        outputs['K_rad_L'] = inputs['K_rad'] - inputs['K_rad_LL']
        outputs['K_rad_U'] = inputs['K_rad'] - inputs['K_rad_UL']
        outputs['D_ratio_L'] = inputs['D_ratio'] - inputs['D_ratio_LL']
        outputs['D_ratio_U'] = inputs['D_ratio'] - inputs['D_ratio_UL']

#----------------------------------------------------------------------------------------------
        
class MofI(om.ExplicitComponent):
    """
    Compute moments of inertia.
    
    Parameters
    ----------
    R_out : float, [m]
        Outer radius
    stator_mass : float, [kg]
        Total rotor mass
    rotor_mass : float, [kg]
        Total rotor mass
    generator_mass : float, [kg]
        Actual mass
    len_s : float, [m]
        Stator core length
    
    Returns
    -------
    generator_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    rotor_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the rotor about its center of mass
    stator_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the stator about its center of mass
    
    """
    def setup(self):
        self.add_input('R_out', val=0.0, units ='m')
        self.add_input('stator_mass', val=0.0, units='kg')
        self.add_input('rotor_mass', val=0.0, units='kg')
        self.add_input('generator_mass', val=0.0, units='kg')
        self.add_input('len_s', val=0.0, units='m')

        self.add_output('generator_I', val=np.zeros(3), units='kg*m**2')
        self.add_output('rotor_I', val=np.zeros(3), units='kg*m**2')
        self.add_output('stator_I', val=np.zeros(3), units='kg*m**2')
        
    def compute(self, inputs, outputs):
        R_out = inputs['R_out']
        Mass  = inputs['generator_mass']
        m_stator = inputs['stator_mass']
        m_rotor  = inputs['rotor_mass']
        len_s = inputs['len_s']

        I = np.zeros(3)
        I[0] = 0.50*Mass*R_out**2
        I[1] = I[2] = 0.5*I[0] + Mass*len_s**2 / 12.
        outputs['generator_I'] = I
        coeff = m_stator / Mass if m_stator > 0.0 else 0.5
        outputs['stator_I'] = coeff * I
        coeff = m_rotor / Mass if m_rotor > 0.0 else 0.5
        outputs['rotor_I'] = coeff * I
#----------------------------------------------------------------------------------------------
        
class Cost(om.ExplicitComponent):
    """
    Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded.
    
    Parameters
    ----------
    C_Cu : float, [USD/kg]
        Specific cost of copper
    C_Fe : float, [USD/kg]
        Specific cost of magnetic steel/iron
    C_Fes : float, [USD/kg]
        Specific cost of structural steel
    C_PM : float, [USD/kg]
        Specific cost of Magnet
    Copper : float, [kg]
        Copper mass
    Iron : float, [kg]
        Iron mass
    mass_PM : float, [kg]
        Magnet mass
    Structural_mass : float, [kg]
        Structural mass
    
    Returns
    -------
    generator_cost : float, [USD]
        Total cost
    
    """
    def setup(self):
        
        # Specific cost of material by type
        self.add_input('C_Cu', val=0.0, units='USD/kg')
        self.add_input('C_Fe', val=0.0, units='USD/kg')
        self.add_input('C_Fes', val=0.0, units='USD/kg')
        self.add_input('C_PM', val=0.0, units='USD/kg')
        
        # Mass of each material type
        self.add_input('Copper', val=0.0, units='kg')
        self.add_input('Iron', val=0.0, units='kg')
        self.add_input('mass_PM', val=0.0, units='kg')
        self.add_input('Structural_mass', val=0.0, units='kg')

        # Outputs
        self.add_output('generator_cost', val=0.0, units='USD')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs):
        Copper          = inputs['Copper']
        Iron            = inputs['Iron']
        mass_PM         = inputs['mass_PM']
        Structural_mass = inputs['Structural_mass']
        C_Cu            = inputs['C_Cu']
        C_Fes           = inputs['C_Fes']
        C_Fe            = inputs['C_Fe']
        C_PM            = inputs['C_PM']
                
        # Material cost as a function of material mass and specific cost of material
        K_gen            = Copper*C_Cu + Iron*C_Fe + C_PM*mass_PM #%M_pm*K_pm; # 
        Cost_str         = C_Fes*Structural_mass
        outputs['generator_cost'] = K_gen + Cost_str
        
#----------------------------------------------------------------------------------------------
class Generator(om.Group):

    def initialize(self):
        genTypes = ['scig','dfig','eesg','pmsg_arms','pmsg_disc','pmsg_outer']
        self.options.declare('topLevelFlag', default=True)
        self.options.declare('design', values=genTypes + [m.upper() for m in genTypes])
    
    def setup(self):
        topLevelFlag = self.options['topLevelFlag']
        genType      = self.options['design']
        
        ivc = om.IndepVarComp()
        sivc = om.IndepVarComp()

        ivc.add_output('B_r', val=1.2, units='T')
        ivc.add_output('P_Fe0e', val=1.0, units='W/kg')
        ivc.add_output('P_Fe0h', val=4.0, units='W/kg')
        ivc.add_output('S_N', val=-0.002)
        ivc.add_output('alpha_p', val=0.5*np.pi*0.7)
        ivc.add_output('b_r_tau_r', val=0.45)
        ivc.add_output('b_ro', val=0.004, units='m')
        ivc.add_output('b_s_tau_s', val=0.45)
        ivc.add_output('b_so', val=0.004, units='m')
        ivc.add_output('cofi', val=0.85)
        ivc.add_output('freq', val=60, units='Hz')
        ivc.add_output('h_i', val=0.001, units='m')
        ivc.add_output('h_sy0', val=0.0)
        ivc.add_output('h_w', val=0.005, units='m')
        ivc.add_output('k_fes', val=0.9)
        ivc.add_output('k_fillr', val=0.7)
        ivc.add_output('k_fills', val=0.65)
        ivc.add_output('k_s', val=0.2)
        ivc.add_discrete_output('m', val=3)
        ivc.add_output('mu_0', val=np.pi*4e-7, units='m*kg/s**2/A**2')
        ivc.add_output('mu_r', val=1.06, units='m*kg/s**2/A**2')
        ivc.add_output('p', val=3.0)
        ivc.add_output('phi', val=np.deg2rad(90), units='rad')
        ivc.add_discrete_output('q1', val=6)
        ivc.add_discrete_output('q2', val=4)
        ivc.add_output('ratio_mw2pp', val=0.7)
        ivc.add_output('resist_Cu', val=1.8e-8*1.4, units='ohm/m')
        ivc.add_output('sigma', val=40e3, units='Pa')
        ivc.add_output('y_tau_p', val=1.0)
        ivc.add_output('y_tau_pr', val=10. / 12)

        ivc.add_output('I_0', val=0.0, units='A')
        ivc.add_output('d_r', val=0.0, units='m')
        ivc.add_output('h_m', val=0.0, units='m')
        ivc.add_output('h_0', val=0.0, units ='m')
        ivc.add_output('h_s', val=0.0, units='m')
        ivc.add_output('len_s', val=0.0, units='m')
        ivc.add_output('n_r', val=0.0)
        ivc.add_output('rad_ag', val=0.0, units='m')
        ivc.add_output('t_wr', val=0.0, units='m')
        
        ivc.add_output('n_s', val=0.0)
        ivc.add_output('b_st', val=0.0, units='m')
        ivc.add_output('d_s', val=0.0, units='m')
        ivc.add_output('t_ws', val=0.0, units='m')
        
        ivc.add_output('rho_Copper', val=0.0, units='kg*m**-3')
        ivc.add_output('rho_Fe', val=0.0, units='kg*m**-3')
        ivc.add_output('rho_Fes', val=0.0, units='kg*m**-3')
        ivc.add_output('rho_PM', val=0.0, units='kg*m**-3')

        ivc.add_output('C_Cu',  val=0.0, units='USD/kg')
        ivc.add_output('C_Fe',  val=0.0, units='USD/kg')
        ivc.add_output('C_Fes', val=0.0, units='USD/kg')
        ivc.add_output('C_PM',  val=0.0, units='USD/kg')
        
        if genType.lower() in ['pmsg_outer']:
            ivc.add_output('r_g',0.0, units ='m')
            ivc.add_output('N_c',0.0)
            ivc.add_output('b',0.0)
            ivc.add_output('c',0.0)
            ivc.add_output('E_p',0.0, units ='V')
            ivc.add_output('h_yr', val=0.0, units ='m')
            ivc.add_output('h_ys', val=0.0, units ='m')
            ivc.add_output('h_sr',0.0,units='m',desc='Structural Mass')
            ivc.add_output('h_ss',0.0, units ='m')
            ivc.add_output('t_r',0.0, units ='m')
            ivc.add_output('t_s',0.0, units ='m')
            
            ivc.add_output('u_allow_pcent',0.0)
            ivc.add_output('y_allow_pcent',0.0)
            ivc.add_output('z_allow_deg',0.0,units='deg')
            ivc.add_output('B_tmax',0.0, units='T')

            # These are part of sivc and only get added if topLevelFlag
            sivc.add_output('P_mech', 0.0, units='W')
            sivc.add_output('y_sh', units ='m')
            sivc.add_output('theta_sh', 0.0, units='rad')
            sivc.add_output('D_nose',0.0, units ='m')
            sivc.add_output('y_bd', units ='m')
            sivc.add_output('theta_bd', 0.0, units='rad')
            
        if genType.lower() in ['eesg','pmsg_arms','pmsg_disc']:
            ivc.add_output('tau_p', val=0.0, units='m')
            ivc.add_output('h_ys',  val=0.0, units='m')
            ivc.add_output('h_yr',  val=0.0, units='m')
            ivc.add_output('b_arm',   val=0.0, units='m')
            
        elif genType.lower() in ['scig','dfig']:
            ivc.add_output('B_symax', val=0.0, units='T')
            ivc.add_output('S_Nmax', val=-0.2)
        
        if topLevelFlag:
            self.add_subsystem('ivc', ivc, promotes=['*'])
        
            sivc.add_output('machine_rating', 0.0, units='W')
            sivc.add_output('rated_rpm', 0.0, units='rpm')
            sivc.add_output('rated_torque', 0.0, units='N*m')
            sivc.add_output('D_shaft', val=0.0, units='m')
            sivc.add_output('E', val=0., units='Pa')
            sivc.add_output('G', val=0., units='Pa')
            self.add_subsystem('sivc', sivc, promotes=['*'])

        # Easy Poisson ratio assuming isotropic
        self.add_subsystem('poisson', om.ExecComp('v = 0.5*E/G - 1.0', E={'units':'Pa'}, G={'units':'Pa'}), promotes=['*'])
        
        # Add generator design component and cost
        if genType.lower() == 'scig':
            mygen = gm.SCIG
            
        elif genType.lower() == 'dfig':
            mygen = gm.DFIG
            
        elif genType.lower() == 'eesg':
            mygen = gm.EESG
            
        elif genType.lower() == 'pmsg_arms':
            mygen = gm.PMSG_Arms
            
        elif genType.lower() == 'pmsg_disc':
            mygen = gm.PMSG_Disc
            
        elif genType.lower() == 'pmsg_outer':
            mygen = gm.PMSG_Outer
            
        self.add_subsystem('generator', mygen(), promotes=['*'])
        self.add_subsystem('mofi', MofI(), promotes=['*'])
        self.add_subsystem('gen_cost', Cost(), promotes=['*'])
        self.add_subsystem('constr', Constraints(), promotes=['*'])

