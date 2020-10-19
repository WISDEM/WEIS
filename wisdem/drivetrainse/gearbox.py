import openmdao.api as om
import numpy as np
from scipy.optimize import minimize

# Application factor to include ring/housing/carrier weight
Kr = 0.4

#-----------------------------------

def V_planetary(U, B, K):
    sunU = 0.5*U - 1.0
    V = 1./U + 1./U/B + 1./B/(0.5*U-1.0) + sunU + sunU**2 + K*(U-1.)**2/B + K*(U-1.)**2/B/sunU
    return V
#-----------------------------------

def V_parallel(U):
    V = (1. + 1./U + U + U**2)
    return V
#-----------------------------------
def volumeEEP(x, n_planets, torque, Kr1=Kr, Kr2=Kr):
    # Safety factor?
    Kgamma = [1.1 if m < 5 else 1.35 for m in n_planets]

    # Individual stage torques
    Q_stage = torque / np.cumprod(x)

    # Volume
    V = (Q_stage[0] * Kgamma[0] * V_planetary(x[0], n_planets[0], Kr1) +
         Q_stage[1] * Kgamma[1] * V_planetary(x[1], n_planets[1], Kr2) +
         Q_stage[2]             * V_parallel( x[2])/np.prod(x) )
    return 2*V
#-----------------------------------

def volumeEPP(x, n_planets, torque, Kr1=Kr):
    # Safety factor?
    Kgamma = [1.1 if m < 5 else 1.35 for m in n_planets]

    # Individual stage torques
    Q_stage = torque / np.cumprod(x)

    V = (Q_stage[0] * Kgamma[0] * V_planetary(x[0], n_planets[0], Kr1) +
         Q_stage[1]             * V_parallel( x[1])/np.prod(x[:2]) +
         Q_stage[2]             * V_parallel( x[2])/np.prod(x) )
    return 2*V
#-----------------------------------
        

class Gearbox(om.ExplicitComponent):
    """
    Gearbox class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
    Approach computes the gearbox weight based on a surface durability criteria.
    
    Parameters
    ----------
    gear_configuration : string
        3-letter string of Es or Ps to denote epicyclic or parallel gear configuration
    n_planets : numpy array[3]
        number of planets in each stage
    gear_ratio : float
        overall gearbox speedup ratio
    D_rotor : float, [m]
        rotor diameter
    Q_rotor : float, [N*m]
        rotor torque at rated power
    s_gearbox : float, [m]
        gearbox position along x-axis
    
    Returns
    -------
    stage_masses : numpy array[3], [kg]
        individual gearbox stage gearbox_masses
    gearbox_mass : float, [kg]
        overall component mass
    gearbox_cm : numpy array[3], [m]
        Gearbox center of mass [x,y,z] measure along shaft from bedplate
    gearbox_I : numpy array[3], [kg*m**2]
        Gearbox mass moments of inertia [Ixx, Iyy, Izz] around its center of mass
    L_gearbox : float, [m]
        length of gearbox
    H_gearbox : float, [m]
        height of gearbox
    D_gearbox : float, [m]
        diameter of gearbox
    
    """

    def initialize(self):
        self.options.declare('direct_drive', default=True)
        
    def setup(self):
        self.add_discrete_input('gear_configuration', val='eep')
        #self.add_discrete_input('shaft_factor', val='normal')
        self.add_discrete_input('planet_numbers', val=[3, 3, 0])
        self.add_input('gear_ratio', val=1.0)
        self.add_input('rotor_diameter', val=0.0, units='m')
        self.add_input('rated_torque', val=0.0, units='N*m')
        self.add_input('machine_rating', val=0.0, units='kW')

        self.add_output('stage_ratios', val=np.zeros(3))
        self.add_output('gearbox_mass', 0.0, units='kg')
        self.add_output('gearbox_I', np.zeros(3), units='kg*m**2')
        self.add_output('L_gearbox', 0.0, units='m')
        self.add_output('D_gearbox', 0.0, units='m')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        if self.options['direct_drive']:
            outputs['stage_ratios'] = np.zeros(3)
            outputs['gearbox_mass'] = outputs['D_gearbox'] = outputs['L_gearbox'] = 0.0
            outputs['gearbox_I'] = np.zeros(3)            
            return

        # Unpack inputs
        config       = discrete_inputs['gear_configuration']
        #shaft_factor = discrete_inputs['shaft_factor']
        n_planets    = np.maximum(1.0, np.array(discrete_inputs['planet_numbers']))
        gear_ratio   = inputs['gear_ratio']
        D_rotor      = inputs['rotor_diameter']
        torque       = inputs['rated_torque']
        rating       = inputs['machine_rating']

        # Known configuration checks
        if not config.lower() in ['eep','eep_2','eep_3','epp']:
            raise ValueError('Invalid value for gearbox_configuration.  Must be one of: eep, eep_2, eep_3, epp')
        n_stage = 3

        # Optimize stage ratios

        # Use double sided constraints to hack inequality constraints as COBYLA seems to work better than SLSQP here
        def constr1(x, ratio):
            return np.prod(x) - ratio
        def constr2(x, ratio):
            return ratio - np.prod(x)

        x0     = gear_ratio ** (1.0 / n_stage) * np.ones(n_stage)
        bounds = [[2.01, 20.0],[2.01, 20.0],[2.01, 20.0]]
        const  = [{},{}]
        const[0]['type'] = 'ineq'
        const[0]['fun']  = constr1
        const[0]['args'] = [gear_ratio]
        const[1]['type'] = 'ineq'
        const[1]['fun']  = constr2
        const[1]['args'] = [gear_ratio]
        method = 'cobyla'
        tol = 1e-3
        
        if config.lower() == 'eep':
            bounds[2][0] = 1.0
            result = minimize(lambda x: volumeEEP(x, n_planets, torque), x0, method=method, #bounds=bounds,
                              tol=tol, constraints=const, options={'maxiter': 100} )
            ratio_stage = result.x

        elif config == 'eep_3':
            # fixes last stage ratio at 3
            const[0]['args'] = const[1]['args'] = [gear_ratio/3.0]
            bounds[2][0] = 1.0
            result = minimize(lambda x: volumeEEP(np.r_[x, 3.0], n_planets, torque), x0[:2], method=method, #bounds=bounds,
                              tol=tol, constraints=const, options={'maxiter': 100} )
            ratio_stage = np.r_[result.x, 3.0]

        elif config == 'eep_2':
            # fixes final stage ratio at 2
            const[0]['args'] = const[1]['args'] = [gear_ratio/2.0]
            bounds[2][0] = 1.0
            result = minimize(lambda x: volumeEEP(np.r_[x, 2.0], n_planets, torque), x0[:2], method=method, #bounds=bounds,
                              tol=tol, constraints=const, options={'maxiter': 100} )
            ratio_stage = np.r_[result.x, 2.0]

        elif config == 'epp':
            bounds[1][0] = 1.0
            bounds[2][0] = 1.0
            result = minimize(lambda x: volumeEPP(x, n_planets, torque), x0, method=method, #bounds=bounds,
                              tol=tol, constraints=const, options={'maxiter': 100} )
            ratio_stage = result.x

        # Get final volume
        if config.lower().find('eep') >= 0:
            vol = volumeEEP(ratio_stage, n_planets, torque)
        else:
            vol = volumeEPP(ratio_stage, n_planets, torque)

        # Cumulative Kfactor scaling based on values reported in Nejad's paper even though that was probably done with the buggy version of DriveSE
        K = np.mean([48.82/15216.504, 53.69/17401.453])

        # Shaft length factor
        Kshaft = 1.0 #if shaft_factor == 'normal' else 1.25

        # All factors into the mass
        m_gearbox = K*Kshaft * vol.sum()

        # Other gearbox elements that are just estimates: shrink disc and carrier
        m_shrink_disc = rating / 3.
        m_carrier = 8e3
        m_gearbox += m_shrink_disc + m_carrier
        
        # calculate mass properties
        L_gearbox = (0.012 * D_rotor)
        R_gearbox = 0.5 * 0.75 * 0.015 * D_rotor

        I = np.zeros(3)
        I[0] = 0.5*R_gearbox**2
        I[1:] = (1./12.)*(3*R_gearbox**2 + L_gearbox**2)
        
        # Store outputs
        outputs['stage_ratios'] = ratio_stage
        outputs['gearbox_mass'] = m_gearbox
        outputs['gearbox_I'] = I*m_gearbox
        outputs['D_gearbox'] = 2*R_gearbox
        outputs['L_gearbox'] = L_gearbox

