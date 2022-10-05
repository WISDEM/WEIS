import numpy as np
import openmdao.api as om
from wisdem.glue_code.glue_code import WindPark as wisdemPark
#from wisdem.glue_code.gc_WT_DataStruc import WindTurbineOntologyOpenMDAO
#from wisdem.ccblade.ccblade_component import CCBladeTwist
#from wisdem.commonse.turbine_class import TurbineClass
from wisdem.drivetrainse.drivetrain import DrivetrainSE
from wisdem.commonse.cylinder_member import CylinderPostFrame
#from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015
#from wisdem.orbit.api.wisdem.fixed import Orbit
#from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints
from weis.aeroelasticse.openmdao_openfast import FASTLoadCases
from weis.control.dac import RunXFOIL
from wisdem.rotorse.rotor_power import NoStallConstraint
from weis.control.tune_rosco import ServoSE_ROSCO, ROSCO_Turbine
#from wisdem.rotorse.rotor_elasticity import RotorElasticity
from weis.aeroelasticse.utils import RotorLoadsDeflStrainsWEIS
from wisdem.glue_code.gc_RunTools import Convergence_Trends_Opt
from weis.glue_code.gc_RunTools import Outputs_2_Screen
from weis.frequency.raft_wrapper import RAFT_WEIS
from weis.control.tmd import TMD_group


class WindPark(om.Group):
    # Openmdao group to run the analysis of the wind turbine

    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        #self.linear_solver = lbgs = om.LinearBlockGS()
        #self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        #nlbgs.options['maxiter'] = 2
        #nlbgs.options['atol'] = nlbgs.options['atol'] = 1e-2

        dac_ivc = om.IndepVarComp()
        n_te_flaps = modeling_options['WISDEM']['RotorSE']['n_te_flaps']
        dac_ivc.add_output('te_flap_ext',   val = np.ones(n_te_flaps))
        dac_ivc.add_output('te_flap_start', val=np.zeros(n_te_flaps),               desc='1D array of the start positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('te_flap_end',   val=np.zeros(n_te_flaps),               desc='1D array of the end positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('chord_start',   val=np.zeros(n_te_flaps),               desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        dac_ivc.add_output('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')
        self.add_subsystem('dac_ivc',dac_ivc)

        tune_rosco_ivc = om.IndepVarComp()
        if modeling_options['ROSCO']['linmodel_tuning']['type'] == 'robust':
            n_PC = 1
        else:
            n_PC = len(modeling_options['ROSCO']['U_pc'])
        tune_rosco_ivc.add_output('omega_pc',         val=np.zeros(n_PC), units='rad/s',     desc='Pitch controller natural frequency')
        tune_rosco_ivc.add_output('zeta_pc',          val=np.zeros(n_PC),                    desc='Pitch controller damping ratio')
        tune_rosco_ivc.add_output('omega_vs',         val=0.0, units='rad/s',     desc='Generator torque controller natural frequency')
        tune_rosco_ivc.add_output('zeta_vs',          val=0.0,                    desc='Generator torque controller damping ratio')
        tune_rosco_ivc.add_output('flp_kp_norm',      val=0.0,                    desc='Flap controller normalized gain')
        tune_rosco_ivc.add_output('flp_tau',          val=0.0, units='s',         desc='Flap controller integral gain time constant')
        tune_rosco_ivc.add_output('IPC_Kp1p',         val=0.0, units='s',         desc='Individual pitch controller 1p proportional gain')
        tune_rosco_ivc.add_output('IPC_Ki1p',         val=0.0,                    desc='Individual pitch controller 1p integral gain')
        tune_rosco_ivc.add_output('stability_margin', val=0.0,                    desc='Stability margin for robust tuning')
        tune_rosco_ivc.add_output('omega_pc_max',     val=0.0,                    desc='Maximum allowable omega for robust tuning')
        # optional inputs - not connected right now!!
        tune_rosco_ivc.add_output('max_pitch',        val=0.0, units='rad',       desc='Maximum pitch angle , {default = 90 degrees}')
        tune_rosco_ivc.add_output('min_pitch',        val=0.0, units='rad',       desc='Minimum pitch angle [rad], {default = 0 degrees}')
        tune_rosco_ivc.add_output('vs_minspd',        val=0.0, units='rad/s',     desc='Minimum rotor speed [rad/s], {default = 0 rad/s}')
        tune_rosco_ivc.add_output('ss_cornerfreq',    val=0.0, units='rad/s',     desc='First order low-pass filter cornering frequency for setpoint smoother [rad/s]')
        tune_rosco_ivc.add_output('ss_vsgain',        val=0.0,                    desc='Torque controller setpoint smoother gain bias percentage [%, <= 1 ], {default = 100%}')
        tune_rosco_ivc.add_output('ss_pcgain',        val=0.0,                    desc='Pitch controller setpoint smoother gain bias percentage  [%, <= 1 ], {default = 0.1%}')
        tune_rosco_ivc.add_output('ps_percent',       val=0.0,                    desc='Percent peak shaving  [%, <= 1 ], {default = 80%}')
        tune_rosco_ivc.add_output('sd_maxpit',        val=0.0, units='rad',       desc='Maximum blade pitch angle to initiate shutdown [rad], {default = bld pitch at v_max}')
        tune_rosco_ivc.add_output('sd_cornerfreq',    val=0.0, units='rad/s',     desc='Cutoff Frequency for first order low-pass filter for blade pitch angle [rad/s], {default = 0.41888 ~ time constant of 15s}')
        tune_rosco_ivc.add_output('Kp_flap',          val=0.0, units='s',         desc='Proportional term of the PI controller for the trailing-edge flaps')
        tune_rosco_ivc.add_output('Ki_flap',          val=0.0,                    desc='Integral term of the PI controller for the trailing-edge flaps')
        tune_rosco_ivc.add_output('twr_freq',         val=3.2, units='rps',     desc='Tower natural frequency')
        tune_rosco_ivc.add_output('ptfm_freq',        val=0.2, units='rad/s',     desc='Platform natural frequency')
        tune_rosco_ivc.add_output('Kp_float',         val=0.0, units='s',         desc='Floating feedback gain')

        self.add_subsystem('tune_rosco_ivc',tune_rosco_ivc)

        # Analysis components
        self.add_subsystem('wisdem',   wisdemPark(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])

        # XFOIL
        self.add_subsystem('xf',        RunXFOIL(modeling_options = modeling_options, opt_options = opt_options)) # Recompute polars with xfoil (for flaps)
        # Connections to run xfoil for te flaps
        if not modeling_options['Level3']['from_openfast']:
            self.connect('blade.pa.chord_param',                  'xf.chord')
            self.connect('blade.outer_shape_bem.s',               'xf.s')
            self.connect('blade.interp_airfoils.coord_xy_interp', 'xf.coord_xy_interp')
            self.connect('airfoils.aoa',                          'xf.aoa')
            self.connect("blade.high_level_blade_props.r_blade",  "xf.r")
            self.connect('dac_ivc.te_flap_end',                   'xf.span_end')
            self.connect('dac_ivc.te_flap_ext',                   'xf.span_ext')
            self.connect('dac_ivc.chord_start',                   'xf.chord_start')
            self.connect('dac_ivc.delta_max_pos',                 'xf.delta_max_pos')
            self.connect('dac_ivc.delta_max_neg',                 'xf.delta_max_neg')
            self.connect('env.speed_sound_air',                   'xf.speed_sound_air')
            self.connect('env.rho_air',                           'xf.rho_air')
            self.connect('env.mu_air',                            'xf.mu_air')
            self.connect('control.rated_TSR',                     'xf.rated_TSR')
            if modeling_options['flags']['control']:
                self.connect('control.max_TS',                        'xf.max_TS')
            self.connect("af_3d.cl_corrected", "xf.cl_interp")
            self.connect("af_3d.cd_corrected", "xf.cd_interp")
            self.connect("af_3d.cm_corrected", "xf.cm_interp")

        # ROSCO can be used at all levels
        if modeling_options['ROSCO']['flag']:
            if modeling_options['Level3']['from_openfast']:  # not using WISDEM turbine info
                self.add_subsystem('rosco_turbine',          ROSCO_Turbine(modeling_options = modeling_options)) # ROSCO tuning

            self.add_subsystem('sse_tune',          ServoSE_ROSCO(modeling_options = modeling_options, opt_options = opt_options)) # ROSCO tuning

            if not modeling_options['Level3']['from_openfast']:     #from WISDEM models
                self.connect('rotorse.rp.powercurve.rated_V',         ['sse_tune.tune_rosco.v_rated'])
                #self.connect('rotorse.rp.gust.V_gust',                ['freq_rotor.aero_gust.V_load', 'freq_rotor.aero_hub_loads.V_load'])
                self.connect('rotorse.rp.powercurve.rated_Omega',     'sse_tune.tune_rosco.rated_rotor_speed')
                #self.connect('rotorse.rp.powercurve.rated_pitch',     ['freq_rotor.pitch_load', 'freq_rotor.tot_loads_gust.aeroloads_pitch'])
                self.connect('rotorse.rp.powercurve.rated_Q',          'sse_tune.tune_rosco.rated_torque')
                
                self.connect("blade.high_level_blade_props.r_blade",  "sse_tune.r")
                self.connect("blade.high_level_blade_props.rotor_radius", "sse_tune.Rtip")
                self.connect('hub.radius',                     'sse_tune.Rhub')
                self.connect("high_level_tower_props.hub_height", "sse_tune.hub_height")
                self.connect('hub.cone',                       'sse_tune.precone')
                self.connect('nacelle.uptilt',                 'sse_tune.tilt')
                self.connect('airfoils.aoa',                   'sse_tune.airfoils_aoa')
                self.connect('airfoils.Re',                    'sse_tune.airfoils_Re')
                self.connect('xf.cl_interp_flaps',             'sse_tune.airfoils_cl')
                self.connect('xf.cd_interp_flaps',             'sse_tune.airfoils_cd')
                self.connect('xf.cm_interp_flaps',             'sse_tune.airfoils_cm')
                self.connect('configuration.n_blades',         'sse_tune.nBlades')
                self.connect('env.rho_air',                    'sse_tune.rho')
                self.connect('env.mu_air',                     'sse_tune.mu')
                self.connect('blade.pa.chord_param',           'sse_tune.chord')
                self.connect('blade.pa.twist_param',           'sse_tune.theta')

                self.connect('control.V_in' ,                   'sse_tune.v_min')
                self.connect('control.V_out' ,                  'sse_tune.v_max')
                self.connect("blade.high_level_blade_props.prebend", 'sse_tune.precurve')
                self.connect("blade.high_level_blade_props.prebendTip", 'sse_tune.precurveTip')
                self.connect("blade.high_level_blade_props.presweep", 'sse_tune.presweep')
                self.connect("blade.high_level_blade_props.presweepTip", 'sse_tune.presweepTip')
                self.connect('xf.flap_angles',                  'sse_tune.airfoils_Ctrl')
                self.connect('control.minOmega',                'sse_tune.omega_min')
                self.connect('control.rated_TSR',               'sse_tune.tsr_operational')
                self.connect('configuration.rated_power',       'sse_tune.rated_power')

                self.connect('nacelle.gear_ratio',              'sse_tune.tune_rosco.gear_ratio')
                self.connect("blade.high_level_blade_props.rotor_radius", "sse_tune.tune_rosco.R")
                self.connect('rotorse.re.precomp.I_all_blades',    'sse_tune.tune_rosco.rotor_inertia', src_indices=[0])
                self.connect('rotorse.rs.frame.flap_mode_freqs','sse_tune.tune_rosco.flap_freq', src_indices=[0])
                self.connect('rotorse.rs.frame.edge_mode_freqs','sse_tune.tune_rosco.edge_freq', src_indices=[0])
                self.connect('rotorse.rp.powercurve.rated_efficiency', 'sse_tune.tune_rosco.generator_efficiency')
                self.connect('tower_grid.height',               'sse_tune.tune_rosco.TowerHt')
                self.connect('nacelle.gearbox_efficiency',      'sse_tune.tune_rosco.gearbox_efficiency')
                self.connect('control.max_pitch_rate' ,         'sse_tune.tune_rosco.max_pitch_rate')
                self.connect('control.max_torque_rate' ,        'sse_tune.tune_rosco.max_torque_rate')

            else:       # reading openfast model using ROSCO toolbox via rosco_turbine
                self.connect('rosco_turbine.v_rated'            ,   ['sse_tune.tune_rosco.v_rated'])
                self.connect('rosco_turbine.rated_rotor_speed'  ,   'sse_tune.tune_rosco.rated_rotor_speed')
                self.connect('rosco_turbine.rated_torque'       ,   'sse_tune.tune_rosco.rated_torque')
                self.connect('rosco_turbine.rotor_inertia',         'sse_tune.tune_rosco.rotor_inertia', src_indices=[0])
                self.connect('rosco_turbine.flap_freq',             'sse_tune.tune_rosco.flap_freq', src_indices=[0])
                self.connect('rosco_turbine.edge_freq',             'sse_tune.tune_rosco.edge_freq', src_indices=[0])
                self.connect('rosco_turbine.generator_efficiency',  'sse_tune.tune_rosco.generator_efficiency')                
                self.connect('rosco_turbine.rho',                   'sse_tune.rho')
                self.connect('rosco_turbine.R',                     'sse_tune.tune_rosco.R')
                self.connect('rosco_turbine.rated_power',           'sse_tune.rated_power')
                self.connect('rosco_turbine.v_min' ,                'sse_tune.v_min')
                self.connect('rosco_turbine.v_max' ,                'sse_tune.v_max')
                self.connect('rosco_turbine.max_pitch_rate' ,       'sse_tune.tune_rosco.max_pitch_rate')
                self.connect('rosco_turbine.max_torque_rate' ,      'sse_tune.tune_rosco.max_torque_rate')
                self.connect('rosco_turbine.omega_min',             'sse_tune.omega_min')
                self.connect('rosco_turbine.tsr_operational',       'sse_tune.tsr_operational')
            
                # Performance tables
                self.connect('rosco_turbine.Cp_table',        'sse_tune.tune_rosco.Cp_table')
                self.connect('rosco_turbine.Ct_table',        'sse_tune.tune_rosco.Ct_table')
                self.connect('rosco_turbine.Cq_table',        'sse_tune.tune_rosco.Cq_table')
                self.connect('rosco_turbine.pitch_vector',    'sse_tune.tune_rosco.pitch_vector')
                self.connect('rosco_turbine.tsr_vector',      'sse_tune.tune_rosco.tsr_vector')
                self.connect('rosco_turbine.U_vector',        'sse_tune.tune_rosco.U_vector')

                self.connect('rosco_turbine.gear_ratio',              'sse_tune.tune_rosco.gear_ratio')
                self.connect('rosco_turbine.gearbox_efficiency',      'sse_tune.tune_rosco.gearbox_efficiency')

                self.connect('rosco_turbine.TowerHt',               'sse_tune.tune_rosco.TowerHt')


            # ROSCO Independent Vars
            self.connect('tune_rosco_ivc.max_pitch',        'sse_tune.tune_rosco.max_pitch') 
            self.connect('tune_rosco_ivc.min_pitch',        'sse_tune.tune_rosco.min_pitch')
            self.connect('tune_rosco_ivc.vs_minspd',        'sse_tune.tune_rosco.vs_minspd') 
            self.connect('tune_rosco_ivc.ss_vsgain',        'sse_tune.tune_rosco.ss_vsgain') 
            self.connect('tune_rosco_ivc.ss_pcgain',        'sse_tune.tune_rosco.ss_pcgain') 
            self.connect('tune_rosco_ivc.ps_percent',       'sse_tune.tune_rosco.ps_percent') 
            self.connect('tune_rosco_ivc.omega_pc',         'sse_tune.tune_rosco.omega_pc')
            self.connect('tune_rosco_ivc.zeta_pc',          'sse_tune.tune_rosco.zeta_pc')
            self.connect('tune_rosco_ivc.omega_vs',         'sse_tune.tune_rosco.omega_vs')
            self.connect('tune_rosco_ivc.zeta_vs',          'sse_tune.tune_rosco.zeta_vs')
            self.connect('tune_rosco_ivc.IPC_Kp1p',         'sse_tune.tune_rosco.IPC_Kp1p')
            self.connect('tune_rosco_ivc.IPC_Ki1p',         'sse_tune.tune_rosco.IPC_Ki1p')
            self.connect('tune_rosco_ivc.stability_margin', 'sse_tune.tune_rosco.stability_margin')
            self.connect('tune_rosco_ivc.omega_pc_max', 'sse_tune.tune_rosco.omega_pc_max')

            # Someday, if we want to get ptfm_freq from Level 1, we'd switch that here
            self.connect('tune_rosco_ivc.ptfm_freq',        'sse_tune.tune_rosco.ptfm_freq')

            self.connect('tune_rosco_ivc.Kp_float',         'sse_tune.tune_rosco.Kp_float')
            self.connect('dac_ivc.delta_max_pos',           'sse_tune.tune_rosco.delta_max_pos')
            if modeling_options['ROSCO']['Flp_Mode'] > 0:
                self.connect('tune_rosco_ivc.flp_kp_norm',    'sse_tune.tune_rosco.flp_kp_norm')
                self.connect('tune_rosco_ivc.flp_tau',     'sse_tune.tune_rosco.flp_tau')

        if modeling_options['Level1']['flag']:
            self.add_subsystem('raft', RAFT_WEIS(modeling_options = modeling_options, analysis_options=opt_options))

            n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
            self.connect('configuration.turb_class',        'raft.turbulence_class')
            self.connect('configuration.ws_class' ,         'raft.turbine_class')
            self.connect('drivese.rna_mass', 'raft.turbine_mRNA')
            self.connect('drivese.rna_I_TT', 'raft.rna_I_TT')
            self.connect('drivese.rna_cm', 'raft.rna_cm')
            self.connect("nacelle.overhang", "raft.turbine_overhang")
            self.connect("nacelle.distance_tt_hub", "raft.drive_height")
            self.connect('drivese.base_F', 'raft.turbine_Fthrust', src_indices=[0], flat_src_indices=True) # TODO: Multiple DLCs
            self.connect("high_level_tower_props.hub_height", "raft.turbine_hHub")
            self.connect("tower.layer_thickness", "raft.tower_layer_thickness")
            self.connect("tower_grid.s", "raft.turbine_tower_stations")
            self.connect('tower.diameter', 'raft.turbine_tower_d')
            self.connect('env.water_depth', 'raft.mooring_water_depth')
            self.connect('env.rho_water', 'raft.rho_water')
            self.connect('env.rho_air', 'raft.rho_air')
            self.connect('env.mu_air', 'raft.mu_air')
            self.connect('env.shear_exp', 'raft.shear_exp')
            self.connect('sse_tune.tune_rosco.PC_GS_angles',    'raft.rotor_PC_GS_angles')
            self.connect('sse_tune.tune_rosco.PC_GS_Kp',        'raft.rotor_PC_GS_Kp')
            self.connect('sse_tune.tune_rosco.PC_GS_Ki',        'raft.rotor_PC_GS_Ki')
            self.connect('sse_tune.tune_rosco.Fl_Kp',           'raft.Fl_Kp')
            self.connect('sse_tune.tune_rosco.VS_Kp',           'raft.rotor_TC_VS_Kp')
            self.connect('sse_tune.tune_rosco.VS_Ki',           'raft.rotor_TC_VS_Ki')
            self.connect('rotorse.re.precomp.I_all_blades',     'raft.rotor_inertia', src_indices=[0])
            self.connect('rotorse.rp.powercurve.rated_V',       'raft.Vrated')
            self.connect('control.V_in',                    'raft.V_cutin')
            self.connect('control.V_out',                   'raft.V_cutout')
            self.connect('rotorse.rp.powercurve.rated_Omega',     'raft.rated_rotor_speed')

            if modeling_options["flags"]["blade"]:
                self.connect("configuration.n_blades", "raft.nBlades")
                self.connect("hub.cone", "raft.precone")
                self.connect("nacelle.uptilt", "raft.tilt")
                self.connect("nacelle.gear_ratio", "raft.gear_ratio")
                self.connect("blade.high_level_blade_props.r_blade",  "raft.blade_r")
                self.connect("blade.high_level_blade_props.rotor_radius", "raft.blade_Rtip")
                self.connect("hub.radius", "raft.hub_radius")
                self.connect("blade.pa.chord_param", "raft.blade_chord")
                self.connect("blade.pa.twist_param", "raft.blade_theta")
                self.connect("blade.high_level_blade_props.prebend", "raft.blade_precurve")
                self.connect("blade.high_level_blade_props.prebendTip", "raft.blade_precurveTip")
                self.connect("blade.high_level_blade_props.presweep", "raft.blade_presweep")
                self.connect("blade.high_level_blade_props.presweepTip", "raft.blade_presweepTip")
                self.connect("airfoils.name", "raft.airfoils_name")
                self.connect("airfoils.r_thick", "raft.airfoils_r_thick")
                self.connect("blade.opt_var.af_position", "raft.airfoils_position")
                self.connect("airfoils.aoa", "raft.airfoils_aoa")
                self.connect("airfoils.cl", "raft.airfoils_cl")
                self.connect("airfoils.cd", "raft.airfoils_cd")
                self.connect("airfoils.cm", "raft.airfoils_cm")
                self.connect("high_level_tower_props.hub_height", "raft.wind_reference_height")
                self.connect("rotorse.rp.powercurve.V", "raft.rotor_powercurve_v")
                self.connect("rotorse.rp.powercurve.Omega", "raft.rotor_powercurve_omega_rpm")
                self.connect("rotorse.rp.powercurve.pitch", "raft.rotor_powercurve_pitch")

            if modeling_options["flags"]["tower"]:
                self.connect('towerse.member.rho', 'raft.tower_rho')
                self.connect('towerse.tower_section_height', 'raft.tower_section_height')
                self.connect('towerse.member.tor_stff', 'raft.tower_torsional_stiffness')
                self.connect('towerse.z_param',        'raft.wind.z')
                self.connect("rotorse.rp.gust.V_gust", "raft.Uref")
                self.connect("high_level_tower_props.hub_height", "raft.zref")
                self.connect("high_level_tower_props.tower_ref_axis", "raft.turbine_tower_rA", src_indices=om.slicer[0,:])
                self.connect("high_level_tower_props.tower_ref_axis", "raft.turbine_tower_rB", src_indices=om.slicer[-1,:])
                
            if modeling_options["flags"]["floating"]:
                self.connect("floatingse.member_variable_height", "raft.member_variable_height")

                for k, kname in enumerate(modeling_options["floating"]["members"]["name"]):
                    idx = modeling_options["floating"]["members"]["name2idx"][kname]
                    self.connect(f"floating.memgrid{idx}.outer_diameter", f"raft.platform_member{k+1}_d")
                    self.connect(f"floating.memgrid{idx}.layer_thickness", f"raft.member{k}:layer_thickness")
                    self.connect(f"floatingse.member{k}.height", f"raft.member{k}:height")
                    self.connect(f"floatingse.member{k}.rho", f"raft.member{k}:rho")
                    self.connect(f"floating.memgrp{idx}.s", f"raft.platform_member{k+1}_stations")
                    self.connect(f"floating.memgrp{idx}.ring_stiffener_web_height", f"raft.member{k}:ring_stiffener_web_height")
                    self.connect(f"floating.memgrp{idx}.ring_stiffener_web_thickness", f"raft.member{k}:ring_stiffener_web_thickness")
                    self.connect(f"floating.memgrp{idx}.ring_stiffener_flange_width", f"raft.member{k}:ring_stiffener_flange_width")
                    self.connect(f"floating.memgrp{idx}.ring_stiffener_flange_thickness", f"raft.member{k}:ring_stiffener_flange_thickness")
                    self.connect(f"floating.memgrp{idx}.ring_stiffener_spacing", f"raft.member{k}:ring_stiffener_spacing")
                    self.connect(f"floating.memgrp{idx}.bulkhead_grid", f"raft.platform_member{k+1}_cap_stations")
                    self.connect(f"floating.memgrp{idx}.bulkhead_thickness", f"raft.platform_member{k+1}_cap_t")
                    self.connect(f"floating.member_{kname}:joint1", f"raft.platform_member{k+1}_rA")
                    self.connect(f"floating.member_{kname}:joint2", f"raft.platform_member{k+1}_rB")
                    self.connect(f"floating.member_{kname}:s_ghost1", f"raft.platform_member{k+1}_s_ghostA")
                    self.connect(f"floating.member_{kname}:s_ghost2", f"raft.platform_member{k+1}_s_ghostB")
                    self.connect(f"floating.memgrp{idx}.ballast_grid", f"raft.member{k}:ballast_grid")
                    self.connect(f"floatingse.member{k}.ballast_height", f"raft.member{k}:ballast_height")
                    self.connect(f"floatingse.member{k}.ballast_density", f"raft.member{k}:ballast_density")

                self.connect("mooring.mooring_nodes", 'raft.mooring_nodes')
                self.connect("mooring.unstretched_length", 'raft.unstretched_length')
                for var in ['diameter','mass_density','stiffness','breaking_load','cost_rate',
                            'transverse_added_mass','tangential_added_mass','transverse_drag','tangential_drag']:
                    self.connect(f'mooring.line_{var}', f'raft.line_{var}')

        # TMD connections to openmdao_openfast
        if modeling_options['flags']['TMDs']:
            self.add_subsystem('TMDs',  TMD_group(modeling_options = modeling_options, opt_options = opt_options))

            self.connect('TMDs.mass',               'aeroelastic.TMD_mass')
            self.connect('TMDs.stiffness',          'aeroelastic.TMD_stiffness')
            self.connect('TMDs.damping',            'aeroelastic.TMD_damping')

        if modeling_options['Level3']['flag'] or modeling_options['Level2']['flag']:
            self.add_subsystem('aeroelastic',       FASTLoadCases(modeling_options = modeling_options, opt_options = opt_options))
            self.add_subsystem('stall_check_of',    NoStallConstraint(modeling_options = modeling_options))
            
            if modeling_options['WISDEM']['RotorSE']['flag']: 
                self.add_subsystem('rlds_post',      RotorLoadsDeflStrainsWEIS(modeling_options = modeling_options, opt_options = opt_options))

                # Connections from blade struct parametrization to rotor load anlysis
                self.connect('blade.opt_var.s_opt_spar_cap_ss',   'rlds_post.constr.s_opt_spar_cap_ss')
                self.connect('blade.opt_var.s_opt_spar_cap_ps',   'rlds_post.constr.s_opt_spar_cap_ps')

                # Connections to the stall check 
                self.connect('blade.outer_shape_bem.s',        'stall_check_of.s')
                self.connect('airfoils.aoa',                   'stall_check_of.airfoils_aoa')
                self.connect('xf.cl_interp_flaps',             'stall_check_of.airfoils_cl')
                self.connect('xf.cd_interp_flaps',             'stall_check_of.airfoils_cd')
                self.connect('xf.cm_interp_flaps',             'stall_check_of.airfoils_cm')
                self.connect('aeroelastic.max_aoa',            'stall_check_of.aoa_along_span')
        
            if  modeling_options["flags"]["nacelle"]:
                self.add_subsystem('drivese_post',   DrivetrainSE(modeling_options=modeling_options))

            # TODO: FIX NDLC HERE
            if modeling_options["flags"]["tower"]:
                self.add_subsystem('towerse_post',   CylinderPostFrame(modeling_options=modeling_options["WISDEM"]["TowerSE"], n_dlc=1))
                
            if modeling_options["flags"]["monopile"]:
                self.add_subsystem('fixedse_post',   CylinderPostFrame(modeling_options=modeling_options["WISDEM"]["FixedBottomSE"], n_dlc=1))
                
            if not modeling_options['Level3']['from_openfast']:
                self.add_subsystem('tcons_post',     TurbineConstraints(modeling_options = modeling_options))
                self.add_subsystem('financese_post', PlantFinance(verbosity=modeling_options['General']['verbosity']))
            
            # Post-processing
            self.add_subsystem('outputs_2_screen_weis',  Outputs_2_Screen(modeling_options = modeling_options, opt_options = opt_options))
            if opt_options['opt_flag']:
                self.add_subsystem('conv_plots_weis',    Convergence_Trends_Opt(opt_options = opt_options))

            if modeling_options['ROSCO']['Flp_Mode']:
                # Connections to blade 
                self.connect('dac_ivc.te_flap_end',             'blade.outer_shape_bem.span_end')
                self.connect('dac_ivc.te_flap_ext',             'blade.outer_shape_bem.span_ext')



            if modeling_options['ROSCO']['flag']==False:
                raise Exception("ERROR: WISDEM does not support openfast without the tuning of ROSCO")

            # Configuration parameters needed if model comes from openfast
            self.connect('control.V_in',                    'aeroelastic.V_cutin')
            self.connect('control.V_out',                   'aeroelastic.V_cutout')
            self.connect('env.shear_exp',                   'aeroelastic.shearExp')
            
            # Connections to aeroelasticse
            self.connect('configuration.turb_class',        'aeroelastic.turbulence_class')
            self.connect('configuration.ws_class' ,         'aeroelastic.turbine_class')

            if not modeling_options['Level3']['from_openfast']:
                self.connect('blade.outer_shape_bem.ref_axis',  'aeroelastic.ref_axis_blade')
                self.connect('configuration.rotor_orientation', 'aeroelastic.rotor_orientation')
                self.connect("blade.high_level_blade_props.r_blade",  "aeroelastic.r")
                self.connect('blade.outer_shape_bem.pitch_axis','aeroelastic.le_location')
                self.connect('blade.pa.chord_param',            'aeroelastic.chord')
                self.connect('blade.pa.twist_param',            'aeroelastic.theta')
                self.connect('blade.interp_airfoils.coord_xy_interp', 'aeroelastic.coord_xy_interp')
                self.connect('env.rho_air',                     'aeroelastic.rho')
                self.connect('env.speed_sound_air',             'aeroelastic.speed_sound_air')
                self.connect('env.mu_air',                      'aeroelastic.mu')
                self.connect('env.water_depth',                 'aeroelastic.water_depth')
                self.connect('env.rho_water',                   'aeroelastic.rho_water')
                self.connect('env.mu_water',                    'aeroelastic.mu_water')
                self.connect('env.Hsig_wave',                    'aeroelastic.Hsig_wave')     # TODO: these depend on wind speed
                self.connect('env.Tsig_wave',                    'aeroelastic.Tsig_wave')
                #self.connect('env.beta_wave',                    'aeroelastic.beta_wave') # TODO: NEED ONTOLOGY INPUT HERE
                self.connect("blade.high_level_blade_props.rotor_radius", "aeroelastic.Rtip")
                self.connect('hub.radius',                      'aeroelastic.Rhub')
                self.connect('hub.cone',                        'aeroelastic.cone')
                self.connect('drivese.hub_system_mass',         'aeroelastic.hub_system_mass')
                self.connect('drivese.hub_system_I',            'aeroelastic.hub_system_I')
                # TODO: Create these outputs in DriveSE: hub_system_cm needs 3-dim, not s-coord.  Need adder for rna-yaw_mass?
                #self.connect('drivese_post.hub_system_cm',                    'aeroelastic.hub_system_cm')
                self.connect('drivese.above_yaw_mass',          'aeroelastic.above_yaw_mass')
                self.connect('drivese.yaw_mass',                'aeroelastic.yaw_mass')
                self.connect('drivese.rna_I_TT',                'aeroelastic.rna_I_TT')
                self.connect('drivese.above_yaw_I_TT',          'aeroelastic.nacelle_I_TT')
                self.connect('drivese.above_yaw_cm',            'aeroelastic.nacelle_cm')
                self.connect('drivese.generator_rotor_I',       'aeroelastic.GenIner', src_indices=[0])
                self.connect('nacelle.gear_ratio',              'aeroelastic.gearbox_ratio')
                self.connect('rotorse.rp.powercurve.rated_efficiency',  'aeroelastic.generator_efficiency')
                self.connect('control.max_pitch_rate' ,         'aeroelastic.max_pitch_rate')
                self.connect('nacelle.gearbox_efficiency',      'aeroelastic.gearbox_efficiency')
                self.connect('nacelle.uptilt',                  'aeroelastic.tilt')
                self.connect('nacelle.overhang',                'aeroelastic.overhang')
                self.connect('nacelle.distance_tt_hub',         'aeroelastic.distance_tt_hub')
                self.connect('drivese.drivetrain_spring_constant', 'aeroelastic.drivetrain_spring_constant')
                self.connect('drivese.drivetrain_damping_coefficient', 'aeroelastic.drivetrain_damping_coefficient')

                self.connect("high_level_tower_props.hub_height", "aeroelastic.hub_height")
                if modeling_options["flags"]["tower"]:
                    self.connect('towerse.member.mass_den',                'aeroelastic.mass_den')
                    self.connect('towerse.member.foreaft_stff',            'aeroelastic.foreaft_stff')
                    self.connect('towerse.member.sideside_stff',           'aeroelastic.sideside_stff')
                    self.connect('towerse.member.tor_stff',                'aeroelastic.tor_stff')
                    self.connect('towerse.tower_outer_diameter',    'aeroelastic.tower_outer_diameter')
                    self.connect('towerse.z_param',                 'aeroelastic.tower_z')
                    self.connect('towerse.z_full',                  'aeroelastic.tower_z_full')
                    self.connect('tower.cd',                        'aeroelastic.tower_cd')
                    self.connect('tower_grid.height',               'aeroelastic.tower_height')
                    self.connect('tower_grid.foundation_height',    'aeroelastic.tower_base_height')
                    self.connect('towerse.tower_I_base',            'aeroelastic.tower_I_base')
                    if modeling_options["flags"]["monopile"] or modeling_options["flags"]["jacket"]:
                        self.connect('fixedse.torsion_freqs',      'aeroelastic.tor_freq', src_indices=[0])
                        self.connect('fixedse.tower_fore_aft_modes',     'aeroelastic.fore_aft_modes')
                        self.connect('fixedse.tower_side_side_modes',    'aeroelastic.side_side_modes')
                        self.connect('fixedse.f1',         'sse_tune.tune_rosco.twr_freq')

                    elif modeling_options["flags"]["floating"]:
                        self.connect('floatingse.torsion_freqs',      'aeroelastic.tor_freq', src_indices=[0])
                        self.connect('floatingse.fore_aft_modes',     'aeroelastic.fore_aft_modes')
                        self.connect('floatingse.side_side_modes',    'aeroelastic.side_side_modes')
                        self.connect('floatingse.f1',         'sse_tune.tune_rosco.twr_freq')
                    else:
                        self.connect('towerse.tower.torsion_freqs',      'aeroelastic.tor_freq', src_indices=[0])
                        self.connect('towerse.tower.fore_aft_modes',     'aeroelastic.fore_aft_modes')
                        self.connect('towerse.tower.side_side_modes',    'aeroelastic.side_side_modes')
                        self.connect('towerse.tower.f1',         'sse_tune.tune_rosco.twr_freq')
                        
                if modeling_options['flags']['monopile']:
                    self.connect('monopile.transition_piece_mass',  'aeroelastic.transition_piece_mass')
                    self.connect('fixedse.transition_piece_I',      'aeroelastic.transition_piece_I', src_indices=[0,1,2])
                    self.connect('monopile.gravity_foundation_mass', 'aeroelastic.gravity_foundation_mass')
                    self.connect('fixedse.gravity_foundation_I',    'aeroelastic.gravity_foundation_I', src_indices=[0,1,2])
                    self.connect('fixedse.z_param',                 'aeroelastic.monopile_z')
                    self.connect('fixedse.z_full',                  'aeroelastic.monopile_z_full')
                    self.connect('fixedse.monopile_outer_diameter',    'aeroelastic.monopile_outer_diameter')
                    self.connect('fixedse.monopile_wall_thickness',    'aeroelastic.monopile_wall_thickness')
                    self.connect('fixedse.member.E',                       'aeroelastic.monopile_E')
                    self.connect('fixedse.member.G',                       'aeroelastic.monopile_G')
                    self.connect('fixedse.member.rho',                     'aeroelastic.monopile_rho')
                        
                elif modeling_options['flags']['floating']:
                    self.connect("floatingse.platform_nodes", "aeroelastic.platform_nodes")
                    self.connect("floatingse.platform_elem_n1", "aeroelastic.platform_elem_n1")
                    self.connect("floatingse.platform_elem_n2", "aeroelastic.platform_elem_n2")
                    self.connect("floatingse.platform_elem_D", "aeroelastic.platform_elem_D")
                    self.connect("floatingse.platform_elem_t", "aeroelastic.platform_elem_t")
                    self.connect("floatingse.platform_elem_rho", "aeroelastic.platform_elem_rho")
                    self.connect("floatingse.platform_elem_E", "aeroelastic.platform_elem_E")
                    self.connect("floatingse.platform_elem_G", "aeroelastic.platform_elem_G")
                    self.connect("floatingse.platform_elem_memid", "aeroelastic.platform_elem_memid")
                    if modeling_options['Level1']['flag']:
                        ptfm_data_source = 'raft'
                    else:
                        ptfm_data_source = 'floatingse'
                    self.connect(f"{ptfm_data_source}.platform_mass", "aeroelastic.platform_mass")
                    self.connect(f"{ptfm_data_source}.platform_total_center_of_mass", "aeroelastic.platform_total_center_of_mass")
                    self.connect(f"{ptfm_data_source}.platform_I_total", "aeroelastic.platform_I_total")
                    self.connect(f"{ptfm_data_source}.platform_displacement", "aeroelastic.platform_displacement")
                    self.connect("floating.transition_node", "aeroelastic.transition_node")

                    for k, kname in enumerate(modeling_options["floating"]["members"]["name"]):
                        idx = modeling_options["floating"]["members"]["name2idx"][kname]
                        #self.connect(f"floating.memgrp{idx}.outer_diameter", f"floatingse.member{k}.outer_diameter_in")
                        self.connect(f"floating.memgrp{idx}.s", f"aeroelastic.member{k}:s")
                        self.connect(f"floatingse.member{k}.outer_diameter", f"aeroelastic.member{k}:outer_diameter")
                        self.connect(f"floatingse.member{k}.wall_thickness", f"aeroelastic.member{k}:wall_thickness")

                        for var in ["joint1", "joint2", "s_ghost1", "s_ghost2"]:
                            self.connect(f"floating.member_{kname}:{var}", f"aeroelastic.member{k}:{var}")
                    
                    if modeling_options["flags"]["tower"]:
                        self.connect('floating.transition_piece_mass',  'aeroelastic.transition_piece_mass')
                        self.connect('floatingse.transition_piece_I',      'aeroelastic.transition_piece_I', src_indices=[0,1,2])
                        
                self.connect('airfoils.aoa',                    'aeroelastic.airfoils_aoa')
                self.connect('airfoils.Re',                     'aeroelastic.airfoils_Re')
                self.connect('xf.cl_interp_flaps',              'aeroelastic.airfoils_cl')
                self.connect('xf.cd_interp_flaps',              'aeroelastic.airfoils_cd')
                self.connect('xf.cm_interp_flaps',              'aeroelastic.airfoils_cm')
                self.connect('blade.interp_airfoils.r_thick_interp', 'aeroelastic.rthick')
                self.connect('blade.interp_airfoils.ac_interp', 'aeroelastic.ac')
                self.connect('rotorse.rhoA',                    'aeroelastic.beam:rhoA')
                self.connect('rotorse.EIxx',                    'aeroelastic.beam:EIxx')
                self.connect('rotorse.EIyy',                    'aeroelastic.beam:EIyy')
                self.connect('rotorse.re.Tw_iner',                 'aeroelastic.beam:Tw_iner')
                self.connect('rotorse.rs.frame.flap_mode_shapes',       'aeroelastic.flap_mode_shapes')
                self.connect('rotorse.rs.frame.edge_mode_shapes',       'aeroelastic.edge_mode_shapes')
                self.connect('rotorse.rp.powercurve.V',                'aeroelastic.U')
                self.connect('rotorse.rp.powercurve.Omega',            'aeroelastic.Omega')
                self.connect('rotorse.rp.powercurve.pitch',            'aeroelastic.pitch')
                self.connect('rotorse.rp.powercurve.V_R25',            'aeroelastic.V_R25')
                self.connect('rotorse.rp.powercurve.rated_V',          'aeroelastic.Vrated')
                self.connect('rotorse.rp.gust.V_gust',                 'aeroelastic.Vgust')
                self.connect('rotorse.wt_class.V_extreme1',             'aeroelastic.V_extreme1')
                self.connect('rotorse.wt_class.V_extreme50',            'aeroelastic.V_extreme50')
                self.connect('rotorse.wt_class.V_mean',                 'aeroelastic.V_mean_iec')
                self.connect('configuration.rated_power',       'aeroelastic.control_ratedPower')
                self.connect('control.max_TS',                  'aeroelastic.control_maxTS')
                self.connect('control.maxOmega',                'aeroelastic.control_maxOmega')
                self.connect('sse_tune.aeroperf_tables.pitch_vector','aeroelastic.pitch_vector')
                self.connect('sse_tune.aeroperf_tables.tsr_vector', 'aeroelastic.tsr_vector')
                self.connect('sse_tune.aeroperf_tables.U_vector', 'aeroelastic.U_vector')
                self.connect('sse_tune.aeroperf_tables.Cp',     'aeroelastic.Cp_aero_table')
                self.connect('sse_tune.aeroperf_tables.Ct',     'aeroelastic.Ct_aero_table')
                self.connect('sse_tune.aeroperf_tables.Cq',     'aeroelastic.Cq_aero_table')
                self.connect('xf.flap_angles',                  'aeroelastic.airfoils_Ctrl')

                if modeling_options['flags']['mooring']:
                    self.connect("mooring.line_diameter", "aeroelastic.line_diameter")
                    self.connect("mooring.line_mass_density", "aeroelastic.line_mass_density")
                    self.connect("mooring.line_stiffness", "aeroelastic.line_stiffness")
                    self.connect("mooring.line_transverse_added_mass", "aeroelastic.line_transverse_added_mass")
                    self.connect("mooring.line_tangential_added_mass", "aeroelastic.line_tangential_added_mass")
                    self.connect("mooring.line_transverse_drag", "aeroelastic.line_transverse_drag")
                    self.connect("mooring.line_tangential_drag", "aeroelastic.line_tangential_drag")
                    self.connect("mooring.mooring_nodes", "aeroelastic.nodes_location_full")
                    self.connect("mooring.nodes_mass", "aeroelastic.nodes_mass")
                    self.connect("mooring.nodes_volume", "aeroelastic.nodes_volume")
                    self.connect("mooring.nodes_added_mass", "aeroelastic.nodes_added_mass")
                    self.connect("mooring.nodes_drag_area", "aeroelastic.nodes_drag_area")
                    self.connect("mooring.unstretched_length", "aeroelastic.unstretched_length")
                    self.connect("mooring.node_names", "aeroelastic.node_names")
            
                # For fatigue
                self.connect('configuration.lifetime', 'aeroelastic.lifetime')
                self.connect('blade.fatigue.sparU_sigma_ult', 'aeroelastic.blade_sparU_ultstress')
                self.connect('blade.fatigue.sparU_wohlerexp', 'aeroelastic.blade_sparU_wohlerexp')
                self.connect('blade.fatigue.sparU_wohlerA', 'aeroelastic.blade_sparU_wohlerA')
                self.connect('blade.fatigue.sparL_sigma_ult', 'aeroelastic.blade_sparL_ultstress')
                self.connect('blade.fatigue.sparL_wohlerexp', 'aeroelastic.blade_sparL_wohlerexp')
                self.connect('blade.fatigue.sparL_wohlerA', 'aeroelastic.blade_sparL_wohlerA')
                self.connect('blade.fatigue.teU_sigma_ult', 'aeroelastic.blade_teU_ultstress')
                self.connect('blade.fatigue.teU_wohlerexp', 'aeroelastic.blade_teU_wohlerexp')
                self.connect('blade.fatigue.teU_wohlerA', 'aeroelastic.blade_teU_wohlerA')
                self.connect('blade.fatigue.teL_sigma_ult', 'aeroelastic.blade_teL_ultstress')
                self.connect('blade.fatigue.teL_wohlerexp', 'aeroelastic.blade_teL_wohlerexp')
                self.connect('blade.fatigue.teL_wohlerA', 'aeroelastic.blade_teL_wohlerA')
                self.connect('rotorse.rs.strains.axial_root_sparU_load2stress', 'aeroelastic.blade_root_sparU_load2stress')
                self.connect('rotorse.rs.strains.axial_root_sparL_load2stress', 'aeroelastic.blade_root_sparL_load2stress')
                self.connect('rotorse.rs.strains.axial_maxc_teU_load2stress', 'aeroelastic.blade_maxc_teU_load2stress')
                self.connect('rotorse.rs.strains.axial_maxc_teL_load2stress', 'aeroelastic.blade_maxc_teL_load2stress')
                self.connect('drivese.lss_wohler_exp', 'aeroelastic.lss_wohlerexp')
                self.connect('drivese.lss_wohler_A', 'aeroelastic.lss_wohlerA')
                self.connect('drivese.lss_Xt', 'aeroelastic.lss_ultstress')
                self.connect('drivese.lss_axial_load2stress', 'aeroelastic.lss_axial_load2stress')
                self.connect('drivese.lss_shear_load2stress', 'aeroelastic.lss_shear_load2stress')
                if modeling_options["flags"]["tower"]:
                    self.connect('towerse.member.wohler_exp', 'aeroelastic.tower_wohlerexp')
                    self.connect('towerse.member.wohler_A', 'aeroelastic.tower_wohlerA')
                    self.connect('towerse.member.sigma_ult', 'aeroelastic.tower_ultstress')
                    self.connect('towerse.member.axial_load2stress', 'aeroelastic.tower_axial_load2stress')
                    self.connect('towerse.member.shear_load2stress', 'aeroelastic.tower_shear_load2stress')
                if modeling_options["flags"]["monopile"]:
                    self.connect('fixedse.member.wohler_exp', 'aeroelastic.monopile_wohlerexp')
                    self.connect('fixedse.member.wohler_A', 'aeroelastic.monopile_wohlerA')
                    self.connect('fixedse.member.sigma_ult', 'aeroelastic.monopile_ultstress')
                    self.connect('fixedse.member.axial_load2stress', 'aeroelastic.monopile_axial_load2stress')
                    self.connect('fixedse.member.shear_load2stress', 'aeroelastic.monopile_shear_load2stress')

                # Connections to rotor load analysis
                self.connect('aeroelastic.blade_maxTD_Mx', 'rlds_post.m2pa.Mx')
                self.connect('aeroelastic.blade_maxTD_My', 'rlds_post.m2pa.My')
                self.connect('aeroelastic.blade_maxTD_Fz', 'rlds_post.strains.F3')

                self.connect("rotorse.rs.frame.alpha", "rlds_post.alpha")
                self.connect('rotorse.EA', 'rlds_post.strains.EA')
                self.connect('rotorse.A', 'rlds_post.strains.A')
                self.connect('blade.pa.chord_param',  'rlds_post.strains.chord')
                self.connect('rotorse.rs.frame.EI11', 'rlds_post.strains.EI11')
                self.connect('rotorse.rs.frame.EI22', 'rlds_post.strains.EI22')
                self.connect('rotorse.xu_spar', 'rlds_post.strains.xu_spar')
                self.connect('rotorse.xl_spar', 'rlds_post.strains.xl_spar')
                self.connect('rotorse.yu_spar', 'rlds_post.strains.yu_spar')
                self.connect('rotorse.yl_spar', 'rlds_post.strains.yl_spar')
                self.connect('rotorse.xu_te', 'rlds_post.strains.xu_te')
                self.connect('rotorse.xl_te', 'rlds_post.strains.xl_te')
                self.connect('rotorse.yu_te', 'rlds_post.strains.yu_te')
                self.connect('rotorse.yl_te', 'rlds_post.strains.yl_te')
                self.connect('blade.outer_shape_bem.s','rlds_post.constr.s')

                # Connections to DriveSE
                if modeling_options['WISDEM']['DriveSE']['flag']:
                    self.connect('hub.diameter'                    , 'drivese_post.hub_diameter')
                    self.connect('hub.hub_in2out_circ'             , 'drivese_post.hub_in2out_circ')
                    self.connect('hub.flange_t2shell_t'            , 'drivese_post.flange_t2shell_t')
                    self.connect('hub.flange_OD2hub_D'             , 'drivese_post.flange_OD2hub_D')
                    self.connect('hub.flange_ID2flange_OD'         , 'drivese_post.flange_ID2flange_OD')
                    self.connect('hub.hub_stress_concentration'    , 'drivese_post.hub_stress_concentration')
                    self.connect('hub.n_front_brackets'            , 'drivese_post.n_front_brackets')
                    self.connect('hub.n_rear_brackets'             , 'drivese_post.n_rear_brackets')
                    self.connect('hub.clearance_hub_spinner'       , 'drivese_post.clearance_hub_spinner')
                    self.connect('hub.spin_hole_incr'              , 'drivese_post.spin_hole_incr')
                    self.connect('hub.pitch_system_scaling_factor' , 'drivese_post.pitch_system_scaling_factor')
                    self.connect('hub.spinner_gust_ws'             , 'drivese_post.spinner_gust_ws')
                    self.connect('configuration.n_blades',          'drivese_post.n_blades')
                    self.connect("blade.high_level_blade_props.rotor_diameter", "drivese_post.rotor_diameter")
                    self.connect('configuration.upwind',       'drivese_post.upwind')
                    self.connect('control.minOmega' ,          'drivese_post.minimum_rpm')
                    self.connect('rotorse.rp.powercurve.rated_Omega',  'drivese_post.rated_rpm')
                    self.connect('rotorse.rp.powercurve.rated_Q',      'drivese_post.rated_torque')
                    self.connect('configuration.rated_power',  'drivese_post.machine_rating')    
                    self.connect('tower.diameter',             'drivese_post.D_top', src_indices=[-1])
                    self.connect('aeroelastic.hub_Fxyz',       'drivese_post.F_hub')
                    self.connect('aeroelastic.hub_Mxyz',       'drivese_post.M_hub')
                    self.connect('aeroelastic.max_RootMyb',     'drivese_post.pitch_system.BRFM')
                    self.connect('blade.pa.chord_param',         'drivese_post.blade_root_diameter', src_indices=[0])
                    self.connect('rotorse.re.precomp.blade_mass',        'drivese_post.blade_mass')
                    self.connect('rotorse.re.precomp.mass_all_blades',   'drivese_post.blades_mass')
                    self.connect('rotorse.re.precomp.I_all_blades',      'drivese_post.blades_I')

                    self.connect('nacelle.distance_hub2mb',           'drivese_post.L_h1')
                    self.connect('nacelle.distance_mb2mb',            'drivese_post.L_12')
                    self.connect('nacelle.L_generator',               'drivese_post.L_generator')
                    self.connect('nacelle.overhang',                  'drivese_post.overhang')
                    self.connect('nacelle.distance_tt_hub',           'drivese_post.drive_height')
                    self.connect('nacelle.uptilt',                    'drivese_post.tilt')
                    self.connect('nacelle.gear_ratio',                'drivese_post.gear_ratio')
                    self.connect('nacelle.mb1Type',                   'drivese_post.bear1.bearing_type')
                    self.connect('nacelle.mb2Type',                   'drivese_post.bear2.bearing_type')
                    self.connect('nacelle.lss_diameter',              'drivese_post.lss_diameter')
                    self.connect('nacelle.lss_wall_thickness',        'drivese_post.lss_wall_thickness')
                    if modeling_options['WISDEM']['DriveSE']['direct']:
                        self.connect('nacelle.nose_diameter',              'drivese_post.bear1.D_shaft', src_indices=[0])
                        self.connect('nacelle.nose_diameter',              'drivese_post.bear2.D_shaft', src_indices=[-1])
                    else:
                        self.connect('nacelle.lss_diameter',              'drivese_post.bear1.D_shaft', src_indices=[0])
                        self.connect('nacelle.lss_diameter',              'drivese_post.bear2.D_shaft', src_indices=[-1])
                    self.connect('nacelle.uptower',                   'drivese_post.uptower')
                    self.connect('nacelle.brake_mass_user',           'drivese_post.brake_mass_user')
                    self.connect('nacelle.hvac_mass_coeff',           'drivese_post.hvac_mass_coeff')
                    self.connect('nacelle.converter_mass_user',       'drivese_post.converter_mass_user')
                    self.connect('nacelle.transformer_mass_user',     'drivese_post.transformer_mass_user')

                    if modeling_options['WISDEM']['DriveSE']['direct']:
                        self.connect('nacelle.nose_diameter',             'drivese_post.nose_diameter') # only used in direct
                        self.connect('nacelle.nose_wall_thickness',       'drivese_post.nose_wall_thickness') # only used in direct
                        self.connect('nacelle.bedplate_wall_thickness',   'drivese_post.bedplate_wall_thickness') # only used in direct
                    else:
                        self.connect('nacelle.hss_length',                'drivese_post.L_hss') # only used in geared
                        self.connect('nacelle.hss_diameter',              'drivese_post.hss_diameter') # only used in geared
                        self.connect('nacelle.hss_wall_thickness',        'drivese_post.hss_wall_thickness') # only used in geared
                        self.connect('nacelle.hss_material',              'drivese_post.hss_material')
                        self.connect('nacelle.planet_numbers',            'drivese_post.planet_numbers') # only used in geared
                        self.connect('nacelle.gear_configuration',        'drivese_post.gear_configuration') # only used in geared
                        self.connect('nacelle.bedplate_flange_width',     'drivese_post.bedplate_flange_width') # only used in geared
                        self.connect('nacelle.bedplate_flange_thickness', 'drivese_post.bedplate_flange_thickness') # only used in geared
                        self.connect('nacelle.bedplate_web_thickness',    'drivese_post.bedplate_web_thickness') # only used in geared
                            
                    self.connect('hub.hub_material',                  'drivese_post.hub_material')
                    self.connect('hub.spinner_material',              'drivese_post.spinner_material')
                    self.connect('nacelle.lss_material',              'drivese_post.lss_material')
                    self.connect('nacelle.bedplate_material',         'drivese_post.bedplate_material')
                    self.connect('materials.name',                    'drivese_post.material_names')
                    self.connect('materials.E',                       'drivese_post.E_mat')
                    self.connect('materials.G',                       'drivese_post.G_mat')
                    self.connect('materials.rho',                     'drivese_post.rho_mat')
                    self.connect('materials.sigma_y',                 'drivese_post.Xy_mat')
                    self.connect("materials.Xt",                      "drivese_post.Xt_mat")
                    self.connect("materials.wohler_exp",              "drivese_post.wohler_exp_mat")
                    self.connect("materials.wohler_intercept",        "drivese_post.wohler_A_mat")
                    self.connect('materials.unit_cost',               'drivese_post.unit_cost_mat')

                    if modeling_options['flags']['generator']:

                        self.connect('generator.B_r'          , 'drivese_post.generator.B_r')
                        self.connect('generator.P_Fe0e'       , 'drivese_post.generator.P_Fe0e')
                        self.connect('generator.P_Fe0h'       , 'drivese_post.generator.P_Fe0h')
                        self.connect('generator.S_N'          , 'drivese_post.generator.S_N')
                        self.connect('generator.alpha_p'      , 'drivese_post.generator.alpha_p')
                        self.connect('generator.b_r_tau_r'    , 'drivese_post.generator.b_r_tau_r')
                        self.connect('generator.b_ro'         , 'drivese_post.generator.b_ro')
                        self.connect('generator.b_s_tau_s'    , 'drivese_post.generator.b_s_tau_s')
                        self.connect('generator.b_so'         , 'drivese_post.generator.b_so')
                        self.connect('generator.cofi'         , 'drivese_post.generator.cofi')
                        self.connect('generator.freq'         , 'drivese_post.generator.freq')
                        self.connect('generator.h_i'          , 'drivese_post.generator.h_i')
                        self.connect('generator.h_sy0'        , 'drivese_post.generator.h_sy0')
                        self.connect('generator.h_w'          , 'drivese_post.generator.h_w')
                        self.connect('generator.k_fes'        , 'drivese_post.generator.k_fes')
                        self.connect('generator.k_fillr'      , 'drivese_post.generator.k_fillr')
                        self.connect('generator.k_fills'      , 'drivese_post.generator.k_fills')
                        self.connect('generator.k_s'          , 'drivese_post.generator.k_s')
                        self.connect('generator.m'            , 'drivese_post.generator.m')
                        self.connect('generator.mu_0'         , 'drivese_post.generator.mu_0')
                        self.connect('generator.mu_r'         , 'drivese_post.generator.mu_r')
                        self.connect('generator.p'            , 'drivese_post.generator.p')
                        self.connect('generator.phi'          , 'drivese_post.generator.phi')
                        self.connect('generator.q1'           , 'drivese_post.generator.q1')
                        self.connect('generator.q2'           , 'drivese_post.generator.q2')
                        self.connect('generator.ratio_mw2pp'  , 'drivese_post.generator.ratio_mw2pp')
                        self.connect('generator.resist_Cu'    , 'drivese_post.generator.resist_Cu')
                        self.connect('generator.sigma'        , 'drivese_post.generator.sigma')
                        self.connect('generator.y_tau_p'      , 'drivese_post.generator.y_tau_p')
                        self.connect('generator.y_tau_pr'     , 'drivese_post.generator.y_tau_pr')

                        self.connect('generator.I_0'          , 'drivese_post.generator.I_0')
                        self.connect('generator.d_r'          , 'drivese_post.generator.d_r')
                        self.connect('generator.h_m'          , 'drivese_post.generator.h_m')
                        self.connect('generator.h_0'          , 'drivese_post.generator.h_0')
                        self.connect('generator.h_s'          , 'drivese_post.generator.h_s')
                        self.connect('generator.len_s'        , 'drivese_post.generator.len_s')
                        self.connect('generator.n_r'          , 'drivese_post.generator.n_r')
                        self.connect('generator.rad_ag'       , 'drivese_post.generator.rad_ag')
                        self.connect('generator.t_wr'         , 'drivese_post.generator.t_wr')

                        self.connect('generator.n_s'          , 'drivese_post.generator.n_s')
                        self.connect('generator.b_st'         , 'drivese_post.generator.b_st')
                        self.connect('generator.d_s'          , 'drivese_post.generator.d_s')
                        self.connect('generator.t_ws'         , 'drivese_post.generator.t_ws')

                        self.connect('generator.rho_Copper'   , 'drivese_post.generator.rho_Copper')
                        self.connect('generator.rho_Fe'       , 'drivese_post.generator.rho_Fe')
                        self.connect('generator.rho_Fes'      , 'drivese_post.generator.rho_Fes')
                        self.connect('generator.rho_PM'       , 'drivese_post.generator.rho_PM')

                        self.connect('generator.C_Cu'         , 'drivese_post.generator.C_Cu')
                        self.connect('generator.C_Fe'         , 'drivese_post.generator.C_Fe')
                        self.connect('generator.C_Fes'        , 'drivese_post.generator.C_Fes')
                        self.connect('generator.C_PM'         , 'drivese_post.generator.C_PM')

                        if modeling_options['WISDEM']['GeneratorSE']['type'] in ['pmsg_outer']:
                            self.connect('generator.N_c'          , 'drivese_post.generator.N_c')
                            self.connect('generator.b'            , 'drivese_post.generator.b')
                            self.connect('generator.c'            , 'drivese_post.generator.c')
                            self.connect('generator.E_p'          , 'drivese_post.generator.E_p')
                            self.connect('generator.h_yr'         , 'drivese_post.generator.h_yr')
                            self.connect('generator.h_ys'         , 'drivese_post.generator.h_ys')
                            self.connect('generator.h_sr'         , 'drivese_post.generator.h_sr')
                            self.connect('generator.h_ss'         , 'drivese_post.generator.h_ss')
                            self.connect('generator.t_r'          , 'drivese_post.generator.t_r')
                            self.connect('generator.t_s'          , 'drivese_post.generator.t_s')

                            self.connect('generator.u_allow_pcent', 'drivese_post.generator.u_allow_pcent')
                            self.connect('generator.y_allow_pcent', 'drivese_post.generator.y_allow_pcent')
                            self.connect('generator.z_allow_deg'  , 'drivese_post.generator.z_allow_deg')
                            self.connect('generator.B_tmax'       , 'drivese_post.generator.B_tmax')
                            self.connect('rotorse.rp.powercurve.rated_mech', 'drivese_post.generator.P_mech')

                        if modeling_options['WISDEM']['GeneratorSE']['type'] in ['eesg','pmsg_arms','pmsg_disc']:
                            self.connect('generator.tau_p'        , 'drivese_post.generator.tau_p')
                            self.connect('generator.h_ys'         , 'drivese_post.generator.h_ys')
                            self.connect('generator.h_yr'         , 'drivese_post.generator.h_yr')
                            self.connect('generator.b_arm'        , 'drivese_post.generator.b_arm')

                        elif modeling_options['WISDEM']['GeneratorSE']['type'] in ['scig','dfig']:
                            self.connect('generator.B_symax'      , 'drivese_post.generator.B_symax')
                            self.connect('generator.S_Nmax'      , 'drivese_post.generator.S_Nmax')

                        if modeling_options['WISDEM']['DriveSE']['direct']:
                            self.connect('nacelle.nose_diameter',             'drivese_post.generator.D_nose', src_indices=[-1])
                            self.connect('nacelle.lss_diameter',              'drivese_post.generator.D_shaft', src_indices=[0])
                        else:
                            self.connect('nacelle.hss_diameter',              'drivese_post.generator.D_shaft', src_indices=[-1])

                    else:
                        self.connect('generator.generator_mass_user', 'drivese_post.generator_mass_user')
                        self.connect('generator.generator_efficiency_user', 'drivese_post.generator_efficiency_user')

                # Connections to TowerSE
                if modeling_options["flags"]["tower"]:
                    tow_params = ["z_full","d_full","t_full",
                                  "E_full","G_full","rho_full","sigma_y_full"]
                    for k in tow_params:
                        self.connect(f'towerse.{k}', f'towerse_post.{k}')
                    self.connect("towerse.env.qdyn", "towerse_post.qdyn")
                    self.connect("tower_grid.height", "towerse_post.bending_height")

                    self.connect("aeroelastic.tower_maxMy_Fz", "towerse_post.cylinder_Fz")
                    self.connect("aeroelastic.tower_maxMy_Fx", "towerse_post.cylinder_Vx")
                    self.connect("aeroelastic.tower_maxMy_Fy", "towerse_post.cylinder_Vy")
                    self.connect("aeroelastic.tower_maxMy_Mx", "towerse_post.cylinder_Mxx")
                    self.connect("aeroelastic.tower_maxMy_My", "towerse_post.cylinder_Myy")
                    self.connect("aeroelastic.tower_maxMy_Mz", "towerse_post.cylinder_Mzz")

                if modeling_options["flags"]["monopile"]:
                    mono_params = ["z_full","d_full","t_full",
                                  "E_full","G_full","rho_full","sigma_y_full"]
                    for k in mono_params:
                        self.connect(f'fixedse.{k}', f'fixedse_post.{k}')
                    self.connect("fixedse.env.qdyn", "fixedse_post.qdyn")
                    self.connect("monopile.height", "fixedse_post.bending_height")

                    self.connect("aeroelastic.monopile_maxMy_Fz", "fixedse_post.cylinder_Fz")
                    self.connect("aeroelastic.monopile_maxMy_Fx", "fixedse_post.cylinder_Vx")
                    self.connect("aeroelastic.monopile_maxMy_Fy", "fixedse_post.cylinder_Vy")
                    self.connect("aeroelastic.monopile_maxMy_Mx", "fixedse_post.cylinder_Mxx")
                    self.connect("aeroelastic.monopile_maxMy_My", "fixedse_post.cylinder_Myy")
                    self.connect("aeroelastic.monopile_maxMy_Mz", "fixedse_post.cylinder_Mzz")

                #self.connect('yield_stress',            'tow.sigma_y') # TODO- materials
                #self.connect('max_taper_ratio',         'max_taper') # TODO- 
                #self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
					
                # Connections to turbine constraints
                self.connect('configuration.rotor_orientation', 'tcons_post.rotor_orientation')
                self.connect('aeroelastic.max_TipDxc',          'tcons_post.tip_deflection')
                self.connect("blade.high_level_blade_props.rotor_radius", "tcons_post.Rtip")
                self.connect('blade.outer_shape_bem.ref_axis',  'tcons_post.ref_axis_blade')
                self.connect('hub.cone',                        'tcons_post.precone')
                self.connect('nacelle.uptilt',                  'tcons_post.tilt')
                self.connect('nacelle.overhang',                'tcons_post.overhang')
                self.connect('tower.ref_axis',                  'tcons_post.ref_axis_tower')
                self.connect('tower.diameter',                  'tcons_post.d_full')
                
            else:  # connections from outside WISDEM
                self.connect('rosco_turbine.v_rated',               'aeroelastic.Vrated')
                self.connect('rosco_turbine.R',                     'aeroelastic.Rtip')
                self.connect('rosco_turbine.hub_height',            'aeroelastic.hub_height')
                self.connect('rosco_turbine.twr_freq',              'sse_tune.tune_rosco.twr_freq')
            
            # Inputs to plantfinancese from wt group
            if not modeling_options['Level3']['from_openfast']:

                # Connect computed AEP only if DLC 1.1 is used, otherwise use rotorse
                if modeling_options['DLC_driver']['n_ws_dlc11'] > 0:
                    self.connect('aeroelastic.AEP', 'financese_post.turbine_aep')
                else:
                    self.connect('rotorse.rp.AEP', 'financese_post.turbine_aep')

                self.connect('tcc.turbine_cost_kW',     'financese_post.tcc_per_kW')
                if modeling_options["flags"]["bos"]:
                    if modeling_options['flags']['offshore']:
                        self.connect('orbit.total_capex_kW',    'financese_post.bos_per_kW')
                    else:
                        self.connect('landbosse.bos_capex_kW',  'financese_post.bos_per_kW')
                else:
                    self.connect("costs.bos_per_kW", "financese_post.bos_per_kW")

            # Inputs to plantfinancese from input yaml
            if modeling_options['flags']['control'] and not modeling_options['Level3']['from_openfast']:
                self.connect('configuration.rated_power',     'financese_post.machine_rating')

            if not modeling_options['Level3']['from_openfast']:    
                self.connect('costs.turbine_number',    'financese_post.turbine_number')
                self.connect('costs.opex_per_kW',       'financese_post.opex_per_kW')
                self.connect('costs.offset_tcc_per_kW', 'financese_post.offset_tcc_per_kW')
                self.connect('costs.wake_loss_factor',  'financese_post.wake_loss_factor')
                self.connect('costs.fixed_charge_rate', 'financese_post.fixed_charge_rate')

            if modeling_options['DLC_driver']['n_ws_dlc11'] > 0:
                self.connect('aeroelastic.AEP',     'outputs_2_screen_weis.aep')

            # Connections to outputs to screen
            if not modeling_options['Level3']['from_openfast']:
                self.connect('financese_post.lcoe',          'outputs_2_screen_weis.lcoe')

                self.connect('rotorse.re.precomp.blade_mass',  'outputs_2_screen_weis.blade_mass')
                self.connect('aeroelastic.max_TipDxc', 'outputs_2_screen_weis.tip_deflection')

            if modeling_options['General']['openfast_configuration']['model_only'] == False:
                self.connect('aeroelastic.DEL_RootMyb',        'outputs_2_screen_weis.DEL_RootMyb')
                self.connect('aeroelastic.DEL_TwrBsMyt',       'outputs_2_screen_weis.DEL_TwrBsMyt')
                self.connect('aeroelastic.rotor_overspeed',    'outputs_2_screen_weis.rotor_overspeed')
                self.connect('aeroelastic.Std_PtfmPitch',      'outputs_2_screen_weis.Std_PtfmPitch')
                self.connect('aeroelastic.Max_PtfmPitch',      'outputs_2_screen_weis.Max_PtfmPitch')
                self.connect('tune_rosco_ivc.omega_pc',        'outputs_2_screen_weis.omega_pc')
                self.connect('tune_rosco_ivc.zeta_pc',         'outputs_2_screen_weis.zeta_pc')
                self.connect('tune_rosco_ivc.omega_vs',        'outputs_2_screen_weis.omega_vs')
                self.connect('tune_rosco_ivc.zeta_vs',         'outputs_2_screen_weis.zeta_vs')
                self.connect('tune_rosco_ivc.Kp_float',        'outputs_2_screen_weis.Kp_float')
                self.connect('tune_rosco_ivc.ptfm_freq',       'outputs_2_screen_weis.ptfm_freq')
                self.connect('tune_rosco_ivc.flp_kp_norm',       'outputs_2_screen_weis.flp_kp_norm')
                self.connect('tune_rosco_ivc.flp_tau',        'outputs_2_screen_weis.flp_tau')
                self.connect('tune_rosco_ivc.IPC_Kp1p',        'outputs_2_screen_weis.IPC_Kp1p')
                self.connect('tune_rosco_ivc.IPC_Ki1p',        'outputs_2_screen_weis.IPC_Ki1p')
                self.connect('dac_ivc.te_flap_end',            'outputs_2_screen_weis.te_flap_end')
                if modeling_options['OL2CL']['flag']:
                    self.connect('aeroelastic.OL2CL_pitch',      'outputs_2_screen_weis.OL2CL_pitch')
