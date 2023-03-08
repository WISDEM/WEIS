'''
Controller tuning script.

Nikhar J. Abbas
January 2020
'''

from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox.inputs.validation import load_rosco_yaml
from ROSCO_toolbox.linear.robust_scheduling import rsched_driver
from ROSCO_toolbox.utilities import list_check, DISCON_dict
import numpy as np
from openmdao.api import ExplicitComponent, Group
from wisdem.ccblade.ccblade import CCAirfoil, CCBlade
import yaml, os

weis_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),'../..'))

class ServoSE_ROSCO(Group):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        if not modeling_options['Level3']['from_openfast']:        #haven't already computed
            self.add_subsystem('aeroperf_tables',   Cp_Ct_Cq_Tables(modeling_options   = modeling_options), promotes = ['v_min', 'v_max','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])

        self.add_subsystem('tune_rosco',        TuneROSCO(modeling_options = modeling_options, opt_options=opt_options), promotes = ['v_min', 'v_max', 'rho', 'omega_min', 'tsr_operational', 'rated_power', 'r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_Ctrl', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'mu'])

        if not modeling_options['Level3']['from_openfast']:        #haven't already computed
            # Connect ROSCO for Rotor Performance tables
            self.connect('aeroperf_tables.Cp',              'tune_rosco.Cp_table')
            self.connect('aeroperf_tables.Ct',              'tune_rosco.Ct_table')
            self.connect('aeroperf_tables.Cq',              'tune_rosco.Cq_table')
            self.connect('aeroperf_tables.pitch_vector',    'tune_rosco.pitch_vector')
            self.connect('aeroperf_tables.tsr_vector',      'tune_rosco.tsr_vector')
            self.connect('aeroperf_tables.U_vector',        'tune_rosco.U_vector')



class TuneROSCO(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.opt_options = self.options['opt_options']
        rosco_init_options = self.modeling_options['ROSCO']
        rotorse_init_options = self.modeling_options['WISDEM']['RotorSE']
        n_pc     = rotorse_init_options['n_pc']

        # Input parameters
        self.controller_params = {}
        # Controller Flags
        self.controller_params['LoggingLevel'] = rosco_init_options['LoggingLevel']
        self.controller_params['F_LPFType'] = rosco_init_options['F_LPFType']
        self.controller_params['F_NotchType'] = rosco_init_options['F_NotchType']
        self.controller_params['IPC_ControlMode'] = rosco_init_options['IPC_ControlMode']
        self.controller_params['VS_ControlMode'] = rosco_init_options['VS_ControlMode']
        self.controller_params['PC_ControlMode'] = rosco_init_options['PC_ControlMode']
        self.controller_params['Y_ControlMode'] = rosco_init_options['Y_ControlMode']
        self.controller_params['SS_Mode'] = rosco_init_options['SS_Mode']
        self.controller_params['WE_Mode'] = rosco_init_options['WE_Mode']
        self.controller_params['PS_Mode'] = rosco_init_options['PS_Mode']
        self.controller_params['SD_Mode'] = rosco_init_options['SD_Mode']
        self.controller_params['Fl_Mode'] = rosco_init_options['Fl_Mode']
        self.controller_params['DAC_Mode'] = rosco_init_options['DAC_Mode']
        self.controller_params['DAC_Model'] = rosco_init_options['DAC_Model'] #TODO bem: need to check if this really needed to be added
        self.controller_params['PC_GS_n'] = rosco_init_options['PC_GS_n']
        self.controller_params['WS_GS_n'] = rosco_init_options['WS_GS_n']

        # Necessary parameters
        # Turbine parameters
        self.add_input('rotor_inertia',     val=0.0,        units='kg*m**2',        desc='Rotor inertia')
        self.add_input('rho',               val=0.0,        units='kg/m**3',        desc='Air Density')
        self.add_input('R',                 val=0.0,        units='m',              desc='Rotor Radius')
        self.add_input('gear_ratio',        val=0.0,                                desc='Gearbox Ratio')
        self.add_input('rated_rotor_speed', val=0.0,        units='rad/s',          desc='Rated rotor speed')
        self.add_input('rated_power',       val=0.0,        units='W',              desc='Rated power')
        self.add_input('rated_torque',     val=0.0,                units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_input('v_rated',           val=0.0,        units='m/s',            desc='Rated wind speed')
        self.add_input('v_min',             val=0.0,        units='m/s',            desc='Minimum wind speed (cut-in)')
        self.add_input('v_max',             val=0.0,        units='m/s',            desc='Maximum wind speed (cut-out)')
        self.add_input('max_pitch_rate',    val=0.0,        units='rad/s',          desc='Maximum allowed blade pitch rate')
        self.add_input('max_torque_rate',   val=0.0,        units='N*m/s',          desc='Maximum allowed generator torque rate')
        self.add_input('tsr_operational',   val=0.0,                                desc='Operational tip-speed ratio')
        self.add_input('omega_min',         val=0.0,        units='rad/s',          desc='Minimum rotor speed')
        self.add_input('flap_freq',         val=0.0,        units='Hz',             desc='Blade flapwise first natural frequency')
        self.add_input('edge_freq',         val=0.0,        units='Hz',             desc='Blade edgewise first natural frequency')
        self.add_input('gearbox_efficiency',val=1.0,                                desc='Gearbox efficiency')
        self.add_input('generator_efficiency', val=1.0,                  desc='Generator efficiency')
        self.add_input('TowerHt',           val=1.0,        units='m',              desc='Tower height')
        #
        self.add_input('max_pitch',         val=0.0,        units='rad',            desc='')
        self.add_input('min_pitch',         val=0.0,        units='rad',            desc='')
        self.add_input('vs_minspd',         val=0.0,        units='rad/s',          desc='')
        self.add_input('ss_vsgain',         val=0.0,                                desc='')
        self.add_input('ss_pcgain',         val=0.0,                                desc='')
        self.add_input('ps_percent',        val=0.0,                                desc='')
        # Rotor Power
        if self.modeling_options['WISDEM']['RotorSE']['flag']:
            self.n_pitch    = n_pitch   = rotorse_init_options['n_pitch_perf_surfaces']
            self.n_tsr      = n_tsr     = rotorse_init_options['n_tsr_perf_surfaces']
            self.n_U        = n_U       = rotorse_init_options['n_U_perf_surfaces']
        else:
            self.n_pitch    = n_pitch   = self.modeling_options['ROSCO']['n_pitch']
            self.n_tsr      = n_tsr     = self.modeling_options['ROSCO']['n_tsr']
            self.n_U        = n_U       = self.modeling_options['ROSCO']['n_U']

        self.add_input('Cp_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero power coefficient')
        self.add_input('Ct_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero thrust coefficient')
        self.add_input('Cq_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero torque coefficient')
        self.add_input('pitch_vector',      val=np.zeros(n_pitch),              units='rad',    desc='Pitch vector used')
        self.add_input('tsr_vector',        val=np.zeros(n_tsr),                                desc='TSR vector used')
        self.add_input('U_vector',          val=np.zeros(n_U),                  units='m/s',    desc='Wind speed vector used')

        # For cc-blade & DAC device tuning
        rotorse_options = self.modeling_options['WISDEM']['RotorSE']
        self.n_span     = n_span       = rotorse_options['n_span']
        # self.n_af       = n_af         = af_init_options['n_af'] # Number of airfoils
        self.n_aoa      = n_aoa        = rotorse_options['n_aoa']# Number of angle of attacks
        self.n_Re       = n_Re         = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab      = n_tab        = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_dac      = n_dac        = rotorse_options['n_dac']
        self.add_input('r',             val=np.zeros(n_span),               units='m',          desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span),               units='m',          desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span),               units='deg',        desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,                            units='m',          desc='hub radius')
        self.add_input('Rtip',          val=0.0,                            units='m',          desc='tip radius')
        self.add_input('hub_height',    val=0.0,                            units='m',          desc='hub height')
        self.add_input('precone',       val=0.0,                            units='deg',        desc='precone angle', )
        self.add_input('tilt',          val=0.0,                            units='deg',        desc='shaft tilt', )
        self.add_input('yaw',           val=0.0,                            units='deg',        desc='yaw error', )
        self.add_input('precurve',      val=np.zeros(n_span),               units='m',          desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                            units='m',          desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(n_span),               units='m',          desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                            units='m',          desc='presweep at tip')
        self.add_input('airfoils_cl',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',  val=np.zeros((n_aoa)),              units='deg',        desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',   val=np.zeros((n_Re)),                                   desc='Reynolds numbers of polars')
        self.add_input('airfoils_Ctrl', val=np.zeros((n_span, n_Re, n_tab)),                    desc='Airfoil control paremeter (i.e. flap angle or LEMS height)')
        self.add_discrete_input('nBlades',         val=0,                                       desc='number of blades')
        self.add_input('mu',            val=1.81e-5,                        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',      val=0.0,                                                desc='shear exponent')
        self.add_input('delta_max_pos', val=np.zeros(n_dac),                                    desc='1D array of the max DAC parameter values.')
        self.add_discrete_input('nSector',      val=4,                                          desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',      val=True,                                       desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',      val=True,                                       desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,                                       desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',        val=True,                                       desc='use drag coefficient in computing induction factors')

        # Controller Tuning Parameters
        if rosco_init_options['linmodel_tuning']['type'] == 'robust':
            n_PC = 1
        else:
            n_PC = len(rosco_init_options['U_pc'])
        self.add_input('zeta_pc',           val=np.zeros(n_PC),                                 desc='Pitch controller damping ratio')
        self.add_input('omega_pc',          val=np.zeros(n_PC),        units='rad/s',           desc='Pitch controller natural frequency')
        self.add_input('stability_margin',  val=0.0,                                            desc='Maximum stability margin for robust scheduling')
        self.add_input('omega_pc_max',      val=0.0,                                            desc='Maximum allowable omega margin for robust scheduling')
        self.add_input('twr_freq',          val=0.0,        units='Hz',                         desc='Tower natural frequency')
        self.add_input('ptfm_freq',         val=0.0,        units='rad/s',                      desc='Platform natural frequency')
        self.add_output('VS_Kp',            val=0.0,        units='s',                          desc='Generator torque control proportional gain at first point in schedule')
        self.add_output('VS_Ki',            val=0.0,                                            desc='Generator torque control integral gain at first point in schedule')
        self.add_input('Kp_float',          val=0.0,        units='s',                          desc='Floating feedback gain')
        self.add_input('zeta_vs',           val=0.0,                                            desc='Generator torque controller damping ratio')
        self.add_input('omega_vs',          val=0.0,        units='rad/s',                      desc='Generator torque controller natural frequency')
        if rosco_init_options['DAC_Mode'] > 0:
            self.add_input('dac_kp_norm',   val=0.0,                                    desc='Flap controller normalized gain') #TODO bem: may need to change naming convention at some point
            self.add_input('dac_tau',       val=0.0,            units='s',              desc='Flap controller integral gain time constant')
        self.add_input('IPC_Kp1p',          val=0.0,            units='s',              desc='Individual pitch controller 1p proportional gain')
        self.add_input('IPC_Ki1p',          val=0.0,                                    desc='Individual pitch controller 1p integral gain')
        # Outputs for constraints and optimizations
        self.add_output('dactune_coeff1',   val=0.0,            units='rad/s',          desc='First coefficient in denominator of flap controller tuning model')
        self.add_output('dactune_coeff2',   val=0.0,            units='(rad/s)**2',     desc='Second coefficient in denominator of flap controller tuning model')
        self.add_output('PC_Kp',            val=0.0,            units='rad',            desc='Pitch control proportional gain at first pitch angle in schedule')
        self.add_output('PC_Ki',            val=0.0,            units='rad',            desc='Pitch control integral gain at first pitch angle in schedule')
        self.add_output('DAC_Kp',           val=0.0,            units='rad',            desc='Flap control proportional gain') #TODO bem: may need to change naming convention at some point 
        self.add_output('DAC_Ki',           val=0.0,            units='rad',            desc='Flap control integral gain')

        self.add_output('PC_GS_angles',     val=np.zeros(rosco_init_options['PC_GS_n']), units='rad', desc='Gain-schedule table: pitch angles')
        self.add_output('PC_GS_Kp',         val=np.zeros(rosco_init_options['PC_GS_n']), units='s',   desc='Gain-schedule table: pitch controller kp gains')
        self.add_output('PC_GS_Ki',         val=np.zeros(rosco_init_options['PC_GS_n']),              desc='Gain-schedule table: pitch controller ki gains')
        self.add_output('Fl_Kp',            val=0.0,            desc='Floating feedback gain')

        # self.add_output('VS_Rgn2K',     val=0.0, units='N*m/(rad/s)**2',      desc='Generator torque constant in Region 2 (HSS side), [N-m/(rad/s)^2]')

    def compute(self,inputs,outputs, discrete_inputs, discrete_outputs):
        '''
        Call ROSCO toolbox to define controller
        '''
        rosco_init_options   = self.modeling_options['ROSCO']
        # Add control tuning parameters to dictionary
        rosco_init_options['omega_pc']    = inputs['omega_pc'].tolist()
        rosco_init_options['zeta_pc']     = inputs['zeta_pc'].tolist()
        rosco_init_options['omega_vs']    = float(inputs['omega_vs'])
        rosco_init_options['zeta_vs']     = float(inputs['zeta_vs'])
        if rosco_init_options['DAC_Mode'] > 0:
            #self.modeling_options['Level3']['ServoDyn']['AfCmode'] = 5
            if rosco_init_options['DAC_Mode'] > 3: # used for bang-bang controller
                self.modeling_options['Level3']['AeroDyn']['AFTabMod'] = 3
            rosco_init_options['dac_kp_norm'] = float(inputs['dac_kp_norm'])
            rosco_init_options['dac_tau']  = float(inputs['dac_tau'])
        else:
            rosco_init_options['omega_dac'] = 0.0
            rosco_init_options['zeta_dac']  = 0.0
        rosco_init_options['max_pitch']   = float(inputs['max_pitch'])
        rosco_init_options['min_pitch']   = float(inputs['min_pitch'])
        rosco_init_options['vs_minspd']   = float(inputs['vs_minspd'])
        rosco_init_options['ss_vsgain']   = float(inputs['ss_vsgain'])
        rosco_init_options['ss_pcgain']   = float(inputs['ss_pcgain'])
        rosco_init_options['ps_percent']  = float(inputs['ps_percent'])
        rosco_init_options['IPC_Kp1p']    = max(0.0, float(inputs['IPC_Kp1p']))
        rosco_init_options['IPC_Ki1p']    = max(0.0, float(inputs['IPC_Ki1p']))
        rosco_init_options['IPC_Kp2p']    = 0.0 # 2P optimization is not currently supported
        rosco_init_options['IPC_Kp2p']    = 0.0

        if rosco_init_options['DAC_Mode'] > 0:
            rosco_init_options['dac_maxval']  = float(inputs['delta_max_pos'])

        # If Kp_float is a design variable, do not automatically tune it
        if self.opt_options['design_variables']['control']['servo']['pitch_control']['Kp_float']['flag']:
            rosco_init_options['Kp_float'] = float(inputs['Kp_float'])
            rosco_init_options['tune_Fl'] = False

        # Define necessary turbine parameters
        WISDEM_turbine = type('', (), {})()
        WISDEM_turbine.v_min        = float(inputs['v_min'])
        WISDEM_turbine.J            = float(inputs['rotor_inertia'])
        WISDEM_turbine.rho          = float(inputs['rho'])
        WISDEM_turbine.rotor_radius = float(inputs['R'])
        WISDEM_turbine.Ng           = float(inputs['gear_ratio'])
        # Incoming value already has gearbox eff included, so have to separate it out
        WISDEM_turbine.GenEff       = float(inputs['generator_efficiency'] / inputs['gearbox_efficiency']) * 100.
        WISDEM_turbine.GBoxEff      = float(inputs['gearbox_efficiency']) * 100.
        WISDEM_turbine.rated_rotor_speed   = float(inputs['rated_rotor_speed'])
        WISDEM_turbine.rated_power  = float(inputs['rated_power'])
        WISDEM_turbine.rated_torque = float(inputs['rated_torque']) / WISDEM_turbine.Ng * float(inputs['gearbox_efficiency'])
        WISDEM_turbine.max_torque   = WISDEM_turbine.rated_torque * 1.1  # TODO: make this an input if studying constant power
        WISDEM_turbine.v_rated      = float(inputs['rated_rotor_speed'])*float(inputs['R']) / float(inputs['tsr_operational'])
        WISDEM_turbine.v_min        = float(inputs['v_min'])
        WISDEM_turbine.v_max        = float(inputs['v_max'])
        WISDEM_turbine.max_pitch_rate   = float(inputs['max_pitch_rate'])
        WISDEM_turbine.min_pitch_rate   = -float(inputs['max_pitch_rate'])
        WISDEM_turbine.TSR_operational  = float(inputs['tsr_operational'])
        WISDEM_turbine.max_torque_rate  = float(inputs['max_torque_rate'])
        WISDEM_turbine.TowerHt          = float(inputs['TowerHt'])
        WISDEM_turbine.bld_edgewise_freq = float(inputs['edge_freq']) * 2 * np.pi

        # Floating Feedback Filters
        rosco_init_options['twr_freq']      = float(inputs['twr_freq']) * 2 * np.pi
        rosco_init_options['ptfm_freq']     = float(inputs['ptfm_freq'])

        # Load Cp tables
        self.Cp_table       = WISDEM_turbine.Cp_table = np.squeeze(inputs['Cp_table'])
        self.Ct_table       = WISDEM_turbine.Ct_table = np.squeeze(inputs['Ct_table'])
        self.Cq_table       = WISDEM_turbine.Cq_table = np.squeeze(inputs['Cq_table'])
        self.pitch_vector   = WISDEM_turbine.pitch_initial_rad = inputs['pitch_vector']
        self.tsr_vector     = WISDEM_turbine.TSR_initial = inputs['tsr_vector']

        # self.Cp_table       = WISDEM_turbine.Cp_table = self.Cp_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        # self.Ct_table       = WISDEM_turbine.Ct_table = self.Ct_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        # self.Cq_table       = WISDEM_turbine.Cq_table = self.Cq_table.reshape(len(self.pitch_vector),len(self.tsr_vector))

        RotorPerformance = ROSCO_turbine.RotorPerformance
        WISDEM_turbine.Cp   = RotorPerformance(self.Cp_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Ct   = RotorPerformance(self.Ct_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Cq   = RotorPerformance(self.Cq_table,self.pitch_vector,self.tsr_vector)

        # Load blade info to pass to dac controller tuning process
        if rosco_init_options['DAC_Mode'] >= 1:
            # Create airfoils
            af = [None]*self.n_span
            for i in range(self.n_span):
                if self.n_tab > 1:
                    ref_tab = int(np.floor(self.n_tab/2))
                    af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,ref_tab], inputs['airfoils_cd'][i,:,:,ref_tab], inputs['airfoils_cm'][i,:,:,ref_tab])
                else:
                    af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])

            # Initialize CCBlade as cc_rotor object
            WISDEM_turbine.cc_rotor = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])

            # Load aerodynamic performance data for blades
            WISDEM_turbine.af_data = [{} for i in range(self.n_span)]
            for i in range(self.n_span):
                # Check number of dac device positions for each airfoil section
                if self.n_tab > 1:
                    if inputs['airfoils_Ctrl'][i,0,0] == inputs['airfoils_Ctrl'][i,0,1]:
                        n_tabs = 1  # If all Ctrl dac parameter values are identical then no dac devices
                    else:
                        n_tabs = self.n_tab
                else:
                    n_tabs = 1
                # Save data for each dac device position
                for j in range(n_tabs):
                    WISDEM_turbine.af_data[i][j] = {}
                    WISDEM_turbine.af_data[i][j]['NumTabs'] = n_tabs
                    WISDEM_turbine.af_data[i][j]['Ctrl']    = inputs['airfoils_Ctrl'][i,0,j]
                    WISDEM_turbine.af_data[i][j]['Alpha']   = np.array(inputs['airfoils_aoa']).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cl']      = np.array(inputs['airfoils_cl'][i,:,0,j]).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cd']      = np.array(inputs['airfoils_cd'][i,:,0,j]).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cm']      = np.array(inputs['airfoils_cm'][i,:,0,j]).flatten().tolist()

            # Save some more airfoil info
            WISDEM_turbine.span     = inputs['r']
            WISDEM_turbine.chord    = inputs['chord']
            WISDEM_turbine.twist    = inputs['theta']
            WISDEM_turbine.bld_flapwise_freq = float(inputs['flap_freq']) * 2*np.pi
            WISDEM_turbine.bld_flapwise_damp = self.modeling_options['ROSCO']['Bld_FlpDamp']

        else:
            WISDEM_turbine.bld_flapwise_freq = 0
            WISDEM_turbine.bld_flapwise_damp = 1

        # Instantiate controller
        controller = ROSCO_controller.Controller(rosco_init_options)

        # Stability margin based analysis
        if rosco_init_options['linmodel_tuning']['type'] == 'robust':
            # Scheduling options
            scheduling_options = { 'driver': 'optimization',
                            'windspeed': rosco_init_options['U_pc'],
                            'stability_margin': inputs['stability_margin'][0],
                            'omega': [0.01, inputs['omega_pc'][0]], # two inputs denotes a range for a design variable
                            'k_float': [controller.Kp_float]}    # one input denotes a set value

            # Collect options
            rs_options = {}
            rs_options['linturb_options'] = rosco_init_options['linmodel_tuning']
            rs_options['ROSCO_options'] = {}
            rs_options['ROSCO_options']['controller_params'] = rosco_init_options
            rs_options['path_options']  = {'output_dir': os.path.join(self.options['opt_options']['general']['folder_output'],
                                                                     rosco_init_options['linmodel_tuning']['lintune_outpath']),
                                           'output_name': 'robust_scheduling'
                                          }
            rs_options['opt_options']   = scheduling_options

            # Add inputs to ROSCO_options to bypass need to generate turbine object in the ROSCO toolbox
            rs_options['ROSCO_options']['dict_inputs'] = inputs

            # Run robust scheduling
            self.sd = rsched_driver(rs_options)
            self.sd.setup()
            self.sd.execute()

            # Re-define ROSCO tuning parameters
            controller.omega_pc = self.sd.omegas
            print('Robust tuning omegas = {}'.format(controller.omega_pc))
            controller.zeta_pc = np.ones(len(self.sd.omegas)) * controller.zeta_pc

        # Tune Controller!
        controller.tune_controller(WISDEM_turbine)

        # DISCON Parameters
        ROSCO_input = DISCON_dict(WISDEM_turbine,controller)

        # Cp table info
        ROSCO_input['v_rated'] = float(inputs['v_rated'])
        ROSCO_input['Cp_pitch_initial_rad'] = self.pitch_vector
        ROSCO_input['Cp_TSR_initial'] = self.tsr_vector
        ROSCO_input['Cp_table'] = WISDEM_turbine.Cp_table
        ROSCO_input['Ct_table'] = WISDEM_turbine.Ct_table
        ROSCO_input['Cq_table'] = WISDEM_turbine.Cq_table
        ROSCO_input['Cp'] = WISDEM_turbine.Cp
        ROSCO_input['Ct'] = WISDEM_turbine.Ct
        ROSCO_input['Cq'] = WISDEM_turbine.Cq

        if (self.modeling_options['Level2']['flag'] or self.modeling_options['Level3']['flag']):
            self.modeling_options['General']['openfast_configuration']['fst_vt']['DISCON_in'] = ROSCO_input

        # Outputs
        if rosco_init_options['DAC_Mode'] >= 1:
            outputs['dactune_coeff1']   = 2*WISDEM_turbine.bld_flapwise_damp*WISDEM_turbine.bld_flapwise_freq + controller.kappa[-1]*WISDEM_turbine.bld_flapwise_freq**2*controller.Kp_flap[-1]
            outputs['dactune_coeff2']   = WISDEM_turbine.bld_flapwise_freq**2*(controller.Ki_flap[-1]*controller.kappa[-1] + 1)
        outputs['PC_Kp']   = controller.pc_gain_schedule.Kp[0]
        outputs['PC_Ki']   = controller.pc_gain_schedule.Ki[0]
        outputs['DAC_Kp']  = controller.Kp_flap[-1]
        outputs['DAC_Ki']  = controller.Ki_flap[-1]
        outputs['Fl_Kp']   = controller.Kp_float

        outputs['PC_GS_angles'] = controller.pitch_op_pc
        outputs['PC_GS_Kp']     = controller.pc_gain_schedule.Kp
        outputs['PC_GS_Ki']     = controller.pc_gain_schedule.Ki
        # outputs['VS_Rgn2K']     = controller.vs_rgn2K
        outputs['VS_Kp'] = controller.vs_gain_schedule.Kp[0]
        outputs['VS_Ki'] = controller.vs_gain_schedule.Ki[0]

class Cp_Ct_Cq_Tables(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        # self.options.declare('n_span')
        # self.options.declare('n_pitch', default=20)
        # self.options.declare('n_tsr', default=20)
        # self.options.declare('n_U', default=1)
        # self.options.declare('n_aoa')
        # self.options.declare('n_re')

    def setup(self):
        modeling_options = self.options['modeling_options']
        rotorse_options  = modeling_options['WISDEM']['RotorSE']
        self.n_span        = n_span    = rotorse_options['n_span']
        self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_pitch       = n_pitch   = rotorse_options['n_pitch_perf_surfaces']
        self.n_tsr         = n_tsr     = rotorse_options['n_tsr_perf_surfaces']
        self.n_U           = n_U       = rotorse_options['n_U_perf_surfaces']
        self.min_TSR       = rotorse_options['min_tsr_perf_surfaces']
        self.max_TSR       = rotorse_options['max_tsr_perf_surfaces']
        self.min_pitch     = rotorse_options['min_pitch_perf_surfaces']
        self.max_pitch     = rotorse_options['max_pitch_perf_surfaces']

        # parameters
        self.add_input('v_min',   val=0.0,             units='m/s',       desc='cut-in wind speed')
        self.add_input('v_max',  val=0.0,             units='m/s',       desc='cut-out wind speed')
        self.add_input('r',             val=np.zeros(n_span), units='m',         desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span), units='m',         desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span), units='deg',       desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,             units='m',         desc='hub radius')
        self.add_input('Rtip',          val=0.0,             units='m',         desc='tip radius')
        self.add_input('hub_height',    val=0.0,             units='m',         desc='hub height')
        self.add_input('precone',       val=0.0,             units='deg',       desc='precone angle')
        self.add_input('tilt',          val=0.0,             units='deg',       desc='shaft tilt')
        self.add_input('yaw',           val=0.0,                units='deg',       desc='yaw error')
        self.add_input('precurve',      val=np.zeros(n_span),   units='m',         desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m',         desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(n_span),   units='m',         desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                units='m',         desc='presweep at tip')
        self.add_input('rho',           val=1.225,              units='kg/m**3',    desc='density of air')
        self.add_input('mu',            val=1.81e-5,            units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',      val=0.0,                                desc='shear exponent')
        # self.add_discrete_input('airfoils',      val=[0]*n_span,                 desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
        self.add_discrete_input('nBlades',       val=0,                         desc='number of blades')
        self.add_discrete_input('nSector',       val=4,                         desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',       val=True,                      desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',       val=True,                      desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation',  val=True,                      desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',         val=True,                      desc='use drag coefficient in computing induction factors')
        self.add_input('pitch_vector_in',  val=np.zeros(n_pitch), units='deg',  desc='pitch vector specified by the user')
        self.add_input('tsr_vector_in',    val=np.zeros(n_tsr),                 desc='tsr vector specified by the user')
        self.add_input('U_vector_in',      val=np.zeros(n_U),     units='m/s',  desc='wind vector specified by the user')

        # outputs
        self.add_output('Cp',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero power coefficient')
        self.add_output('Ct',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero thrust coefficient')
        self.add_output('Cq',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero torque coefficient')
        self.add_output('pitch_vector',    val=np.zeros(n_pitch), units='deg',  desc='pitch vector used')
        self.add_output('tsr_vector',      val=np.zeros(n_tsr),                 desc='tsr vector used')
        self.add_output('U_vector',        val=np.zeros(n_U),     units='m/s',  desc='wind vector used')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None]*self.n_span
        for i in range(self.n_span):
            if self.n_tab > 1:
                ref_tab = int(np.floor(self.n_tab/2))
                af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,ref_tab], inputs['airfoils_cd'][i,:,:,ref_tab], inputs['airfoils_cm'][i,:,:,ref_tab])
            else:
                af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])

        n_pitch    = self.n_pitch
        n_tsr      = self.n_tsr
        n_U        = self.n_U
        min_TSR    = self.min_TSR
        max_TSR    = self.max_TSR
        min_pitch  = self.min_pitch
        max_pitch  = self.max_pitch
        U_vector   = inputs['U_vector_in']
        V_in       = inputs['v_min']
        V_out      = inputs['v_max']

        tsr_vector = inputs['tsr_vector_in']
        pitch_vector = inputs['pitch_vector_in']

        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])

        if max(U_vector) == 0.:
            U_vector    = np.linspace(V_in[0],V_out[0], n_U)
        if max(tsr_vector) == 0.:
            tsr_vector = np.linspace(min_TSR, max_TSR, n_tsr)
        if max(pitch_vector) == 0.:
            pitch_vector = np.linspace(min_pitch, max_pitch, n_pitch)

        outputs['pitch_vector'] = pitch_vector
        outputs['tsr_vector']   = tsr_vector
        outputs['U_vector']     = U_vector

        R = inputs['Rtip']
        k=0
        for i in range(n_U):
            for j in range(n_tsr):
                k +=1
                # if k/2. == int(k/2.) :
                print('Cp-Ct-Cq surfaces completed at ' + str(int(k/(n_U*n_tsr)*100.)) + ' %')
                U     =  U_vector[i] * np.ones(n_pitch)
                Omega = tsr_vector[j] *  U_vector[i] / R * 30. / np.pi * np.ones(n_pitch)
                myout, _  = self.ccblade.evaluate(U, Omega, pitch_vector, coefficients=True)
                outputs['Cp'][j,:,i], outputs['Ct'][j,:,i], outputs['Cq'][j,:,i] = [myout[key] for key in ['CP','CT','CQ']]

class ROSCO_Turbine(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']

        # Load yaml file

        parameter_filename = modeling_options['ROSCO']['tuning_yaml']
        if not os.path.isabs(parameter_filename):
            parameter_filename = os.path.join(weis_dir,parameter_filename)

        inps = load_rosco_yaml(parameter_filename)
        self.turbine_params         = inps['turbine_params']
        self.control_params         = inps['controller_params']

        FAST_InputFile = modeling_options['Level3']['openfast_file']    # FAST input file (ext=.fst)
        FAST_directory = modeling_options['Level3']['openfast_dir']   # Path to fst directory files
        if not os.path.isabs(FAST_directory):
            FAST_directory = os.path.join(weis_dir,FAST_directory)


        # Instantiate turbine, controller, and file processing classes
        self.turbine         = ROSCO_turbine.Turbine(self.turbine_params)

        # Load turbine data from OpenFAST and compute Cp surface here
        self.turbine.load_from_fast(FAST_InputFile, FAST_directory, dev_branch=True)

        self.add_output('rotor_inertia',     val=0.0,        units='kg*m**2',        desc='Rotor inertia')
        self.add_output('rho',               val=0.0,        units='kg/m**3',        desc='Air Density')
        self.add_output('R',                 val=0.0,        units='m',              desc='Rotor Radius')
        self.add_output('gear_ratio',        val=0.0,                                desc='Gearbox Ratio')
        self.add_output('rated_rotor_speed', val=0.0,        units='rad/s',          desc='Rated rotor speed')
        self.add_output('rated_power',       val=0.0,        units='W',              desc='Rated power')
        self.add_output('rated_torque',      val=0.0,        units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('v_rated',           val=0.0,        units='m/s',            desc='Rated wind speed')
        self.add_output('v_min',             val=0.0,        units='m/s',            desc='Minimum wind speed (cut-in)')
        self.add_output('v_max',             val=0.0,        units='m/s',            desc='Maximum wind speed (cut-out)')
        self.add_output('max_pitch_rate',    val=0.0,        units='rad/s',          desc='Maximum allowed blade pitch rate')
        self.add_output('max_torque_rate',   val=0.0,        units='N*m/s',          desc='Maximum allowed generator torque rate')
        self.add_output('tsr_operational',   val=0.0,                                desc='Operational tip-speed ratio')
        self.add_output('omega_min',         val=0.0,        units='rad/s',          desc='Minimum rotor speed')
        self.add_output('flap_freq',         val=0.0,        units='Hz',             desc='Blade flapwise first natural frequency')
        self.add_output('edge_freq',         val=0.0,        units='Hz',             desc='Blade edgewise first natural frequency')
        self.add_output('gearbox_efficiency',val=1.0,                                desc='Gearbox efficiency')
        self.add_output('generator_efficiency', val=1.0,                             desc='Generator efficiency')
        self.add_output('TowerHt',           val=1.0,        units='m',              desc='Tower height')
        self.add_output('hub_height',        val=1.0,        units='m',              desc='Hub height')
        self.add_output('twr_freq',          val=0.0,        units='Hz',                         desc='Tower natural frequency')

        #
        self.add_output('max_pitch',         val=0.0,        units='rad',            desc='')
        self.add_output('min_pitch',         val=0.0,        units='rad',            desc='')
        self.add_output('vs_minspd',         val=0.0,        units='rad/s',          desc='')

        # Rotor Performance
        n_pitch    = len(self.turbine.Cp.pitch_initial_rad)
        n_tsr      = len(self.turbine.Cp.TSR_initial)
        n_U        = 1
        modeling_options['ROSCO']['n_pitch']    = n_pitch
        modeling_options['ROSCO']['n_tsr']      = n_tsr
        modeling_options['ROSCO']['n_U']        = n_U
        self.add_output('Cp_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero power coefficient')
        self.add_output('Ct_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero thrust coefficient')
        self.add_output('Cq_table',          val=np.zeros((n_tsr, n_pitch)),                desc='table of aero torque coefficient')
        self.add_output('pitch_vector',      val=np.zeros(n_pitch),              units='rad',    desc='Pitch vector used')
        self.add_output('tsr_vector',        val=np.zeros(n_tsr),                                desc='TSR vector used')
        self.add_output('U_vector',          val=np.zeros(n_U),                  units='m/s',    desc='Wind speed vector used')


    def compute(self, inputs, outputs):

        outputs['rotor_inertia'          ] = self.turbine.rotor_inertia
        outputs['rho'                    ] = self.turbine.rho
        outputs['R'                      ] = self.turbine.rotor_radius
        outputs['gear_ratio'             ] = self.turbine.Ng
        outputs['gearbox_efficiency'     ] = self.turbine.GBoxEff / 100
        outputs['generator_efficiency'   ] = self.turbine.GenEff * outputs['gearbox_efficiency'     ] / 100
        outputs['rated_rotor_speed'      ] = self.turbine.rated_rotor_speed
        outputs['rated_power'            ] = self.turbine.rated_power
        outputs['rated_torque'           ] = self.turbine.rated_torque * self.turbine.Ng / outputs['gearbox_efficiency']  # change to match incoming rated_torque from WISDEM
        outputs['v_rated'                ] = self.turbine.v_rated
        outputs['v_min'                  ] = self.turbine.v_min
        outputs['v_max'                  ] = self.turbine.v_max
        outputs['max_pitch_rate'         ] = self.turbine.max_pitch_rate
        outputs['max_torque_rate'        ] = self.turbine.max_torque_rate
        outputs['tsr_operational'        ] = self.turbine.TSR_operational
        # outputs['omega_min'              ] = self.turbine.dummy
        outputs['flap_freq'              ] = self.turbine.bld_flapwise_freq / 2 / np.pi   # tuning yaml  in rad/s, WISDEM values in Hz: convert to Hz
        outputs['edge_freq'              ] = self.turbine.bld_edgewise_freq / 2 / np.pi   # tuning yaml  in rad/s, WISDEM values in Hz: convert to Hz
        outputs['TowerHt'                ] = self.turbine.TowerHt
        outputs['hub_height'             ] = self.turbine.hubHt
        outputs['twr_freq'               ] = self.control_params['twr_freq'] / 2 / np.pi  # tuning yaml  in rad/s, WISDEM values in Hz: convert to Hz

        # Rotor Performance
        outputs['Cp_table'               ] = self.turbine.Cp.performance_table
        outputs['Ct_table'               ] = self.turbine.Ct.performance_table
        outputs['Cq_table'               ] = self.turbine.Cq.performance_table
        outputs['pitch_vector'           ] = self.turbine.Cp.pitch_initial_rad
        outputs['tsr_vector'             ] = self.turbine.Cp.TSR_initial
        outputs['U_vector'               ] = np.array([5])
