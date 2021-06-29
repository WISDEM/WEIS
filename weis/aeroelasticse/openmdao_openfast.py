import numpy as np
import os, shutil, sys, platform
import copy
from scipy.interpolate                              import PchipInterpolator
from openmdao.api                                   import ExplicitComponent
from wisdem.commonse.mpi_tools                      import MPI
import wisdem.commonse.utilities                    as util
from wisdem.rotorse.rotor_power                     import eval_unsteady
from weis.aeroelasticse.FAST_writer                 import InputWriter_OpenFAST
from wisdem.towerse                                 import NFREQ, get_nfull
from weis.aeroelasticse.runFAST_pywrapper           import runFAST_pywrapper_batch
from weis.aeroelasticse.FAST_post                   import FAST_IO_timeseries
from weis.aeroelasticse.CaseGen_IEC                 import CaseGen_General, CaseGen_IEC
from wisdem.floatingse.floating_frame               import NULL, NNODES_MAX, NELEM_MAX
from weis.aeroelasticse.LinearFAST                  import LinearFAST
from weis.control.LinearModel                       import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse                             import FileTools
from weis.aeroelasticse.Turbsim_mdao.turbsim_file   import TurbSimFile
from ROSCO_toolbox                                  import control_interface as ROSCO_ci
from pCrunch.io                                     import OpenFASTOutput, OpenFASTBinary, OpenFASTAscii
from pCrunch                                        import LoadsAnalysis


# It probably should be here, so the channels can match what we set to be output in openfast
magnitude_channels = {
    'LSShftF': ["RotThrust", "LSShftFys", "LSShftFzs"], 
    'LSShftM': ["RotTorq", "LSSTipMys", "LSSTipMzs"],
    'RootMc1': ["RootMxc1", "RootMyc1", "RootMzc1"],
    'RootMc2': ["RootMxc2", "RootMyc2", "RootMzc2"],
    'RootMc3': ["RootMxc3", "RootMyc3", "RootMzc3"],
    # 'TipDc1': ['TipDxc1', 'TipDyc1', 'TipDzc1'],
    # 'TipDc2': ['TipDxc2', 'TipDyc2', 'TipDzc2'],
    # 'TipDc3': ['TipDxc3', 'TipDyc3', 'TipDzc3'],
}

fatigue_channels = {
    'RootMc1': 10,
    'RootMc2': 10,
    'RootMc3': 10,
    'RootMyb1': 10,
    'RootMyb2': 10,
    'RootMyb3': 10,
    'TwrBsMyt': 10
}


la = LoadsAnalysis(
    outputs=[],
    magnitude_channels=magnitude_channels,
    fatigue_channels=fatigue_channels,
)


from pCrunch import LoadsAnalysis, PowerProduction
import fatpack
import pickle

if MPI:
    from mpi4py   import MPI

if platform.system() == 'Windows':
    lib_ext = '.dll'
elif platform.system() == 'Darwin':
    lib_ext = '.dylib'
else:
    lib_ext = '.so'

class FASTLoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        rotorse_options = self.options['modeling_options']['WISDEM']['RotorSE']
        openfast_init_options = self.options['modeling_options']['openfast']
        mat_init_options     = self.options['modeling_options']['materials']

        self.n_blades      = self.options['modeling_options']['assembly']['number_of_blades']
        self.n_span        = n_span    = rotorse_options['n_span']
        self.n_pc          = n_pc      = rotorse_options['n_pc']
        n_OF     = len(openfast_init_options['dlc_settings']['Power_Curve']['U'])
        if n_OF == 0 and openfast_init_options['dlc_settings']['run_power_curve']:
            for i in range(len(openfast_init_options['dlc_settings']['IEC'])):
                if openfast_init_options['dlc_settings']['IEC'][i]['DLC'] == 1.1:
                    n_OF = len(openfast_init_options['dlc_settings']['IEC'][i]['U'])
            if n_OF == 0:
                raise ValueError('There is a problem with the initialization of the DLCs to compute the powercurve. Please check modeling_options.yaml')

        self.n_pitch       = n_pitch   = rotorse_options['n_pitch_perf_surfaces']
        self.n_tsr         = n_tsr     = rotorse_options['n_tsr_perf_surfaces']
        self.n_U           = n_U       = rotorse_options['n_U_perf_surfaces']
        self.n_mat         = n_mat    = mat_init_options['n_mat']
        self.n_layers      = n_layers = rotorse_options['n_layers']

        self.n_xy          = n_xy      = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        
        self.te_ss_var       = rotorse_options['te_ss']
        self.te_ps_var       = rotorse_options['te_ps']
        self.spar_cap_ss_var = rotorse_options['spar_cap_ss']
        self.spar_cap_ps_var = rotorse_options['spar_cap_ps']

        n_height_tow = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_tower']
        n_height_mon = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_monopile']
        n_height     = self.options['modeling_options']['WISDEM']['TowerSE']['n_height']
        n_full_tow   = get_nfull(n_height_tow)
        n_full_mon   = get_nfull(n_height_mon)
        n_full       = get_nfull(n_height)
        n_freq_tower = int(NFREQ/2)
        n_freq_blade = int(rotorse_options['n_freq']/2)
        n_pc         = int(rotorse_options['n_pc'])

        FASTpref = self.options['modeling_options']['openfast']
        # self.FatigueFile   = self.options['modeling_options']['rotorse']['FatigueFile']
        
        # ElastoDyn Inputs
        # Assuming the blade modal damping to be unchanged. Cannot directly solve from the Rayleigh Damping without making assumptions. J.Jonkman recommends 2-3% https://wind.nrel.gov/forum/wind/viewtopic.php?t=522
        self.add_input('r',                     val=np.zeros(n_span), units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_input('le_location',           val=np.zeros(n_span), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_input('beam:Tw_iner',          val=np.zeros(n_span), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('beam:rhoA',             val=np.zeros(n_span), units='kg/m', desc='mass per unit length')
        self.add_input('beam:EIyy',             val=np.zeros(n_span), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('beam:EIxx',             val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('x_tc',                  val=np.zeros(n_span), units='m',      desc='x-distance to the neutral axis (torsion center)')
        self.add_input('y_tc',                  val=np.zeros(n_span), units='m',      desc='y-distance to the neutral axis (torsion center)')
        self.add_input('flap_mode_shapes',      val=np.zeros((n_freq_blade,5)), desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)')
        self.add_input('edge_mode_shapes',      val=np.zeros((n_freq_blade,5)), desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)')
        self.add_input('gearbox_efficiency',    val=1.0,               desc='Gearbox efficiency')
        self.add_input('gearbox_ratio',         val=1.0,               desc='Gearbox ratio')
        self.add_input('platform_displacement', val=1.0,               desc='Volumetric platform displacement', units='m**3')

        # ServoDyn Inputs
        self.add_input('generator_efficiency',   val=1.0,              desc='Generator efficiency')
        self.add_input('max_pitch_rate',         val=0.0,        units='deg/s',          desc='Maximum allowed blade pitch rate')

        # tower properties
        self.add_input('fore_aft_modes',   val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)')
        self.add_input('side_side_modes',  val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)')
        self.add_input('mass_den',         val=np.zeros(n_height-1),         units='kg/m',   desc='sectional mass per unit length')
        self.add_input('foreaft_stff',     val=np.zeros(n_height-1),         units='N*m**2', desc='sectional fore-aft bending stiffness per unit length about the Y_E elastic axis')
        self.add_input('sideside_stff',    val=np.zeros(n_height-1),         units='N*m**2', desc='sectional side-side bending stiffness per unit length about the Y_E elastic axis')
        self.add_input('tor_stff',    val=np.zeros(n_height-1),         units='N*m**2', desc='torsional stiffness per unit length about the Y_E elastic axis')
        self.add_input('tor_freq',    val=0.0,         units='Hz', desc='First tower torsional frequency')
        self.add_input('tower_section_height', val=np.zeros(n_height-1), units='m',      desc='parameterized section heights along cylinder')
        self.add_input('tower_outer_diameter', val=np.zeros(n_height),   units='m',      desc='cylinder diameter at corresponding locations')
        self.add_input('tower_monopile_z', val=np.zeros(n_height),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
        self.add_input('tower_monopile_z_full', val=np.zeros(n_full),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
        self.add_input('tower_height',              val=0.0, units='m', desc='tower height from the tower base')
        self.add_input('tower_base_height',         val=0.0, units='m', desc='tower base height from the ground or mean sea level')
        self.add_input('tower_cd',         val=np.zeros(n_height_tow),                   desc='drag coefficients along tower height at corresponding locations')
        
        # These next ones are needed for SubDyn
        self.add_input('tower_wall_thickness', val=np.zeros(n_height-1), units='m')
        self.add_input('tower_E', val=np.zeros(n_height-1), units='Pa')
        self.add_input('tower_G', val=np.zeros(n_height-1), units='Pa')
        self.add_input('tower_rho', val=np.zeros(n_height-1), units='kg/m**3')
        self.add_input('transition_piece_mass', val=0.0, units='kg')
        self.add_input('transition_piece_I', val=np.zeros(3), units='kg*m**2')
        self.add_input('gravity_foundation_mass', val=0.0, units='kg')
        self.add_input('gravity_foundation_I', val=np.zeros(3), units='kg*m**2')

        # DriveSE quantities
        self.add_input('hub_system_cm',   val=np.zeros(3),             units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_input('hub_system_I',    val=np.zeros(6),             units='kg*m**2', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_input('hub_system_mass', val=0.0,                     units='kg', desc='mass of hub system')
        self.add_input('above_yaw_mass',  val=0.0, units='kg', desc='Mass of the nacelle above the yaw system')
        self.add_input('yaw_mass',        val=0.0, units='kg', desc='Mass of yaw system')
        self.add_input('rna_I_TT',       val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the rna [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the tower top')
        self.add_input('nacelle_cm',      val=np.zeros(3), units='m', desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_input('nacelle_I_TT',       val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the nacelle [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the tower top')
        self.add_input('distance_tt_hub', val=0.0,         units='m',   desc='Vertical distance from tower top plane to hub flange')
        self.add_input('twr2shaft',       val=0.0,         units='m',   desc='Vertical distance from tower top plane to shaft start')
        self.add_input('GenIner',         val=0.0,         units='kg*m**2',   desc='Moments of inertia for the generator about high speed shaft')
        self.add_input('drivetrain_spring_constant',         val=0.0,         units='N*m/rad',   desc='Moments of inertia for the generator about high speed shaft')
        self.add_input("drivetrain_damping_coefficient", 0.0, units="N*m*s/rad", desc='Equivalent damping coefficient for the drivetrain system')

        # AeroDyn Inputs
        self.add_input('ref_axis_blade',    val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        self.add_input('chord',             val=np.zeros(n_span), units='m', desc='chord at airfoil locations')
        self.add_input('theta',             val=np.zeros(n_span), units='deg', desc='twist at airfoil locations')
        self.add_input('rthick',            val=np.zeros(n_span), desc='relative thickness of airfoil distribution')
        self.add_input('ac',                val=np.zeros(n_span), desc='aerodynamic center of airfoil distribution')
        self.add_input('pitch_axis',        val=np.zeros(n_span), desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('Rhub',              val=0.0, units='m', desc='dimensional radius of hub')
        self.add_input('Rtip',              val=0.0, units='m', desc='dimensional radius of tip')
        self.add_input('airfoils_cl',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',      val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',       val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
        self.add_input('airfoils_Ctrl',     val=np.zeros((n_span, n_Re, n_tab)), units='deg',desc='Airfoil control paremeter (i.e. flap angle)')
        
        # Airfoil coordinates
        self.add_input('coord_xy_interp',   val=np.zeros((n_span, n_xy, 2)),              desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.')

        # Floating platform inputs
        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_I_total", np.zeros(6), units="kg*m**2")
        
        # Turbine level inputs
        self.add_discrete_input('rotor_orientation',val='upwind', desc='Rotor orientation, either upwind or downwind.')
        self.add_input('hub_height',                val=0.0, units='m', desc='hub height')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbulence class')
        self.add_input('control_ratedPower',        val=0.,  units='W',    desc='machine power rating')
        self.add_input('control_maxOmega',          val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_input('control_maxTS',             val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
        self.add_input('cone',             val=0.0, units='deg',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
        self.add_input('tilt',             val=0.0, units='deg',   desc='Nacelle uptilt angle. A standard machine has positive values.')
        self.add_input('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')

        # Initial conditions
        self.add_input('U_init',        val=np.zeros(n_pc), units='m/s', desc='wind speeds')
        self.add_input('Omega_init',    val=np.zeros(n_pc), units='rpm', desc='rotation speeds to run')
        self.add_input('pitch_init',    val=np.zeros(n_pc), units='deg', desc='pitch angles to run')
        self.add_input('V',             val=np.zeros(n_pc), units='m/s',  desc='wind vector')

        # Cp-Ct-Cq surfaces
        self.add_input('Cp_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero power coefficient')
        self.add_input('Ct_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero thrust coefficient')
        self.add_input('Cq_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero torque coefficient')
        self.add_input('pitch_vector',  val=np.zeros(n_pitch), units='deg',  desc='Pitch vector used')
        self.add_input('tsr_vector',    val=np.zeros(n_tsr),                 desc='TSR vector used')
        self.add_input('U_vector',      val=np.zeros(n_U),     units='m/s',  desc='Wind speed vector used')

        # Environmental conditions 
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_input('V_R25',       val=0.0, units='m/s',      desc='region 2.5 transition wind speed')
        self.add_input('Vgust',       val=0.0, units='m/s',      desc='gust wind speed')
        self.add_input('V_extreme1',  val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 1-year retunr period')
        self.add_input('V_extreme50', val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 50-year retunr period')
        self.add_input('V_mean_iec',  val=0.0, units='m/s',      desc='IEC mean wind for turbulence class')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed (cut-out)')
        self.add_input('rho',         val=0.0, units='kg/m**3',  desc='density of air')
        self.add_input('mu',          val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_input('shearExp',    val=0.0,                   desc='shear exponent')
        self.add_input('speed_sound_air',  val=340.,    units='m/s',        desc='Speed of sound in air.')
        self.add_input(
                "water_depth", val=0.0, units="m", desc="Water depth for analysis.  Values > 0 mean offshore"
            )
        self.add_input('rho_water',   val=0.0, units='kg/m**3',  desc='density of water')
        self.add_input('mu_water',    val=0.0, units='kg/(m*s)', desc='dynamic viscosity of water')
        self.add_input('beta_wave',    val=0.0, units='deg', desc='Incident wave propagation heading direction')
        self.add_input('Hsig_wave',    val=0.0, units='m', desc='Significant wave height of incident waves')
        self.add_input('Tsig_wave',    val=0.0, units='s', desc='Peak-spectral period of incident waves')

        # Blade composite material properties (used for fatigue analysis)
        self.add_input('gamma_f',      val=1.35,                             desc='safety factor on loads')
        self.add_input('gamma_m',      val=1.1,                              desc='safety factor on materials')
        self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_input('Xt',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        self.add_input('Xc',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        self.add_input('m',            val=np.zeros([n_mat]),                desc='2D array of the S-N fatigue slope exponent for the materials') 

        # Blade composit layup info (used for fatigue analysis)
        self.add_input('sc_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('sc_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('te_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('te_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_discrete_input('definition_layer', val=np.zeros(n_layers),  desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        # self.add_discrete_input('layer_name',       val=n_layers * [''],     desc='1D array of the names of the layers modeled in the blade structure.')
        # self.add_discrete_input('layer_web',        val=n_layers * [''],     desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
        # self.add_discrete_input('layer_mat',        val=n_layers * [''],     desc='1D array of the names of the materials of each layer modeled in the blade structure.')
        self.layer_name = rotorse_options['layer_name']
        
        # MoorDyn inputs
        mooropt = self.options['modeling_options']["mooring"]
        if self.options["modeling_options"]["flags"]["mooring"]:
            n_nodes = mooropt["n_nodes"]
            n_lines = mooropt["n_lines"]
            self.add_input("line_diameter", val=np.zeros(n_lines), units="m")
            self.add_input("line_mass_density", val=np.zeros(n_lines), units="kg/m")
            self.add_input("line_stiffness", val=np.zeros(n_lines), units="N")
            self.add_input("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
            self.add_input("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
            self.add_input("line_transverse_drag", val=np.zeros(n_lines))
            self.add_input("line_tangential_drag", val=np.zeros(n_lines))
            self.add_input("nodes_location_full", val=np.zeros((n_nodes, 3)), units="m")
            self.add_input("nodes_mass", val=np.zeros(n_nodes), units="kg")
            self.add_input("nodes_volume", val=np.zeros(n_nodes), units="m**3")
            self.add_input("nodes_added_mass", val=np.zeros(n_nodes))
            self.add_input("nodes_drag_area", val=np.zeros(n_nodes), units="m**2")
            self.add_input("unstretched_length", val=np.zeros(n_lines), units="m")
            self.add_discrete_input("node_names", val=[""] * n_nodes)
        
        # FAST run preferences
        self.FASTpref            = FASTpref 
        self.Analysis_Level      = FASTpref['analysis_settings']['Analysis_Level']
        self.debug_level         = FASTpref['analysis_settings']['debug_level']
        if FASTpref['file_management']['FAST_exe'] != 'none':
            if os.path.isabs(FASTpref['file_management']['FAST_exe']):
                self.FAST_exe = FASTpref['file_management']['FAST_exe']
            else:
                self.FAST_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             FASTpref['file_management']['FAST_exe'])

        if FASTpref['file_management']['FAST_lib'] != 'none':
            if os.path.isabs(FASTpref['file_management']['FAST_lib']):
                self.FAST_lib = FASTpref['file_management']['FAST_lib']
            else:
                self.FAST_lib = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             FASTpref['file_management']['FAST_lib'])

        if os.path.isabs(FASTpref['file_management']['FAST_directory']):
            self.FAST_directory = FASTpref['file_management']['FAST_directory']
        else:
            self.FAST_directory = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                               FASTpref['file_management']['FAST_directory'])
        
        if FASTpref['file_management']['Turbsim_exe'] != 'none':
            if os.path.isabs(FASTpref['file_management']['Turbsim_exe']):
                self.Turbsim_exe = FASTpref['file_management']['Turbsim_exe']
            else:
                self.Turbsim_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                                FASTpref['file_management']['Turbsim_exe'])
                
        self.FAST_InputFile      = FASTpref['file_management']['FAST_InputFile']
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            self.FAST_runDirectory = os.path.join(FASTpref['file_management']['FAST_runDirectory'],'rank_%000d'%int(rank))
            self.FAST_namingOut  = FASTpref['file_management']['FAST_namingOut']+'_%000d'%int(rank)
        else:
            self.FAST_runDirectory = FASTpref['file_management']['FAST_runDirectory']
            self.FAST_namingOut  = FASTpref['file_management']['FAST_namingOut']
        self.cores               = FASTpref['analysis_settings']['cores']
        self.case                = {}
        self.channels            = {}

        self.clean_FAST_directory = False
        if 'clean_FAST_directory' in FASTpref.keys():
            self.clean_FAST_directory = FASTpref['clean_FAST_directory']

        self.mpi_run             = False
        if 'mpi_run' in FASTpref['analysis_settings'].keys():
            self.mpi_run         = FASTpref['analysis_settings']['mpi_run']
            if self.mpi_run:
                self.mpi_comm_map_down   = FASTpref['analysis_settings']['mpi_comm_map_down']
        

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('V_out',       val=np.zeros(n_OF), units='m/s',  desc='wind vector')
        self.add_output('P_out',       val=np.zeros(n_OF), units='W',    desc='rotor electrical power')
        self.add_output('P_aero_out',       val=np.zeros(n_OF), units='W',    desc='rotor aerodynamic power')
        self.add_output('Cp_out',      val=np.zeros(n_OF),               desc='rotor aero power coefficient')
        self.add_output('Omega_out',   val=np.zeros(n_OF), units='rpm',  desc='rotation speeds to run')
        self.add_output('pitch_out',   val=np.zeros(n_OF), units='deg',  desc='pitch angles to run')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')
        self.add_output('AEP',         val=0.0,            units='kW*h', desc='annual energy production')

        self.add_output('loads_r',      val=np.zeros(n_span), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega',  val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch',  val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')
        
        # Control outputs
        self.add_output('rotor_overspeed', val=0.0, desc='Maximum percent overspeed of the rotor during an OpenFAST simulation')  # is this over a set of sims?
        
        # Blade outputs
        self.add_output('max_TipDxc', val=0.0, units='m', desc='Maximum of channel TipDxc, i.e. out of plane tip deflection. For upwind rotors, the max value is tower the tower')
        self.add_output('max_RootMyb', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root flapwise moment')
        self.add_output('max_RootMyc', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root out of plane moment')
        self.add_output('max_RootMzb', val=0.0, units='kN*m', desc='Maximum of the signals RootMzb1, RootMzb2, ... across all n blades representing the maximum blade root torsional moment')
        self.add_output('DEL_RootMyb', val=0.0, units='kN*m', desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('max_aoa', val=np.zeros(n_span), units='deg', desc='maxima of the angles of attack distributed along blade span')
        self.add_output('std_aoa', val=np.zeros(n_span), units='deg', desc='standard deviation of the angles of attack distributed along blade span')
        self.add_output('mean_aoa', val=np.zeros(n_span), units='deg', desc='mean of the angles of attack distributed along blade span')
        # Blade loads corresponding to maximum blade tip deflection
        self.add_output('blade_maxTD_Mx', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned x-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_My', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned y-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_Fz', val=np.zeros(n_span), units='kN', desc='distributed force in blade-aligned z-direction corresponding to maximum blade tip deflection')
        
        # Hub outputs
        self.add_output('hub_Fxyz', val=np.zeros(3), units='kN', desc = 'Maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz', val=np.zeros(3), units='kN*m', desc = 'Maximum hub moments in the non rotating frame')

        # Tower outputs
        self.add_output('max_TwrBsMyt',val=0.0, units='kN*m', desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt',val=0.0, units='kN*m', desc='damage equivalent load of tower base bending moment in fore-aft direction')
        self.add_output('tower_maxMy_Fx', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned x-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fy', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned y-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fz', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned z-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mx', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_My', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mz', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')

        # Monopile outputs
        self.add_output('max_M1N1MKye',val=0.0, units='kN*m', desc='maximum of My moment of member 1 at node 1 (base of the monopile)')
        monlen = max(0, n_full_mon-1)
        self.add_output('monopile_maxMy_Fx', val=np.zeros(monlen), units='kN', desc='distributed force in monopile-aligned x-direction corresponding to max_M1N1MKye')
        self.add_output('monopile_maxMy_Fy', val=np.zeros(monlen), units='kN', desc='distributed force in monopile-aligned y-direction corresponding to max_M1N1MKye')
        self.add_output('monopile_maxMy_Fz', val=np.zeros(monlen), units='kN', desc='distributed force in monopile-aligned z-direction corresponding to max_M1N1MKye')
        self.add_output('monopile_maxMy_Mx', val=np.zeros(monlen), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
        self.add_output('monopile_maxMy_My', val=np.zeros(monlen), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
        self.add_output('monopile_maxMy_Mz', val=np.zeros(monlen), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')

        self.add_output('tower_monopile_maxMy_Fx', val=np.zeros(n_full-1), units='kN', desc='distributed force in monopile-aligned x-direction corresponding to max_M1N1MKye')
        self.add_output('tower_monopile_maxMy_Fy', val=np.zeros(n_full-1), units='kN', desc='distributed force in monopile-aligned y-direction corresponding to max_M1N1MKye')
        self.add_output('tower_monopile_maxMy_Fz', val=np.zeros(n_full-1), units='kN', desc='distributed force in monopile-aligned z-direction corresponding to max_M1N1MKye')
        self.add_output('tower_monopile_maxMy_Mx', val=np.zeros(n_full-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
        self.add_output('tower_monopile_maxMy_My', val=np.zeros(n_full-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
        self.add_output('tower_monopile_maxMy_Mz', val=np.zeros(n_full-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
        
        # Floating outputs
        self.add_output('Max_PtfmPitch', val=0.0, desc='Maximum platform pitch angle over a set of OpenFAST simulations')
        self.add_output('Std_PtfmPitch', val=0.0, units='deg', desc='standard deviation of platform pitch angle')

        # self.add_output('C_miners_SC_SS', val=np.zeros((n_span, n_mat, 2)), desc="Miner's rule cummulative damage to Spar Cap, suction side")
        # self.add_output('C_miners_SC_PS', val=np.zeros((n_span, n_mat, 2)), desc="Miner's rule cummulative damage to Spar Cap, pressure side")
        # self.add_output('C_miners_TE_SS', val=np.zeros((n_span, n_mat, 2)), desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, suction side")
        # self.add_output('C_miners_TE_PS', val=np.zeros((n_span, n_mat, 2)), desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, pressure side")

        self.add_discrete_output('fst_vt_out', val={})

        # Iteration counter for openfast calls. Initialize at -1 so 0 after first call
        self.of_inumber = -1
        self.sim_idx = -1
        
        if self.options['modeling_options']['Level2']['flag']:
            self.lin_pkl_file_name = os.path.join(self.options['opt_options']['general']['folder_output'], 'ABCD_matrices.pkl')
            ABCD_list = []
            
            with open(self.lin_pkl_file_name, 'wb') as handle:
                pickle.dump(ABCD_list, handle)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        #print(impl.world_comm().rank, 'Rotor_fast','start')
        sys.stdout.flush()

        if self.options['modeling_options']['Level2']['flag']:
            self.sim_idx += 1
            ABCD = {
                'sim_idx' : self.sim_idx,
                'A' : None,
                'B' : None,
                'C' : None,
                'D' : None,
                'omega_rpm' : None,
                'DescCntrlInpt' : None,
                'DescStates' : None,
                'DescOutput' : None,
                'StateDerivOrder' : None,
                'ind_fast_inps' : None,
                'ind_fast_outs' : None,
                }
            with open(self.lin_pkl_file_name, 'rb') as handle:
                ABCD_list = pickle.load(handle)
                
            ABCD_list.append(ABCD)
            
            with open(self.lin_pkl_file_name, 'wb') as handle:
                pickle.dump(ABCD_list, handle)
        
        fst_vt = self.init_FAST_model()
        fst_vt = self.update_FAST_model(fst_vt, inputs, discrete_inputs)
        
        if self.Analysis_Level == 2:
            # Run FAST with ElastoDyn

            summary_stats, extreme_table, DELs, case_list, case_name_list, dlc_list  = self.run_FAST(inputs, discrete_inputs, fst_vt)
           
            if self.options['modeling_options']['Level2']['flag']:
                LinearTurbine = LinearTurbineModel(
                self.FAST_runDirectory,
                case_name_list,
                nlin=self.options['modeling_options']['Level2']['linearization']['NLinTimes']
                )

                # DZ->JJ: the info you seek is in LinearTurbine
                # LinearTurbine.omega_rpm has the rotor speed at each linearization point
                # LinearTurbine.Desc* has a description of all the inputs, states, outputs
                # DZ TODO: post process operating points, do Level2 simulation, etc.
                print('Saving ABCD matrices!')
                ABCD = {
                    'sim_idx' : self.sim_idx,
                    'A' : LinearTurbine.A_ops,
                    'B' : LinearTurbine.B_ops,
                    'C' : LinearTurbine.C_ops,
                    'D' : LinearTurbine.D_ops,
                    'omega_rpm' : LinearTurbine.omega_rpm,
                    'DescCntrlInpt' : LinearTurbine.DescCntrlInpt,
                    'DescStates' : LinearTurbine.DescStates,
                    'DescOutput' : LinearTurbine.DescOutput,
                    'StateDerivOrder' : LinearTurbine.StateDerivOrder,
                    'ind_fast_inps' : LinearTurbine.ind_fast_inps,
                    'ind_fast_outs' : LinearTurbine.ind_fast_outs,
                    }
                with open(self.lin_pkl_file_name, 'rb') as handle:
                    ABCD_list = pickle.load(handle)
                    
                ABCD_list[self.sim_idx] = ABCD
                
                with open(self.lin_pkl_file_name, 'wb') as handle:
                    pickle.dump(ABCD_list, handle)

                print('Saving Operating Points...')
                
                # Shorten output names from linearization output to one like level3 openfast output
                # This depends on how openfast sets up the linearization output names and may break if that is changed
                OutList     = [out_name.split()[1][:-1] for out_name in LinearTurbine.DescOutput]
                OutOps      = {}
                for i_out, out in enumerate(OutList):
                    OutOps[out] = LinearTurbine.y_ops[i_out,:]

                # save to yaml, might want in analysis outputs
                FileTools.save_yaml(
                    self.options['modeling_options']['openfast']['file_management']['FAST_runDirectory'],
                    'OutOps.yaml',OutOps)

                # Run linear simulation:

                # Get case list, wind inputs should have already been generated
                if self.FASTpref['dlc_settings']['run_IEC'] or self.FASTpref['dlc_settings']['run_blade_fatigue']:
                    case_list, case_name_list, dlc_list = self.DLC_creation_IEC(inputs, discrete_inputs, fst_vt)

                    # Extract disturbance(s)
                    level2_disturbance = []
                    for case in case_list:
                        ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
                        ts_file.compute_rot_avg(fst_vt['ElastoDyn']['TipRad'])
                        u_h         = ts_file['rot_avg'][0,:]
                        tt          = ts_file['t']
                        level2_disturbance.append({'Time':tt, 'Wind': u_h})

                    # This is going to use the last discon_in file of the linearization set as the simulation file
                    # Currently fine because openfast is executed (or not executed if overwrite=False) after the file writing
                    discon_in_file = os.path.join(self.FAST_runDirectory, self.fst_vt['ServoDyn']['DLL_InFile'])

                    lib_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../local/lib/libdiscon'+lib_ext)
        
                    ss = {}
                    et = {}
                    dl = {}
                    ct = []
                    for i_dist, dist in enumerate(level2_disturbance):
                        sim_name = 'l2_sim_{}'.format(i_dist)
                        controller_int = ROSCO_ci.ControllerInterface(
                            lib_name,
                            param_filename=discon_in_file,
                            DT=1/80,        # modelling input?
                            sim_name = os.path.join(self.FAST_runDirectory,sim_name)
                            )

                        l2_out, _, P_op = LinearTurbine.solve(dist,Plot=False,controller=controller_int)

                        output = OpenFASTOutput.from_dict(l2_out, sim_name, magnitude_channels=magnitude_channels)
                    
                        _name, _ss, _et, _dl = la._process_output(output)
                        ss[_name] = _ss
                        et[_name] = _et
                        dl[_name] = _dl
                        ct.append(l2_out)

                        output.df.to_pickle(os.path.join(self.FAST_runDirectory,sim_name+'.p'))

                        summary_stats, extreme_table, DELs = la.post_process(ss, et, dl)
                    
            self.post_process(summary_stats, extreme_table, DELs, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs)
            
            # list_cases, list_casenames, required_channels, case_keys = self.DLC_creation(inputs, discrete_inputs, fst_vt)
            # FAST_Output = self.run_FAST(fst_vt, list_cases, list_casenames, required_channels)

        elif self.Analysis_Level == 1:
            # Write FAST files, do not run
            self.write_FAST(fst_vt, discrete_outputs)

        # discrete_outputs['fst_vt_out'] = fst_vt

        # delete run directory. not recommended for most cases, use for large parallelization problems where disk storage will otherwise fill up
        if self.clean_FAST_directory:
            try:
                shutil.rmtree(self.FAST_runDirectory)
            except:
                print('Failed to delete directory: %s'%self.FAST_runDirectory)

    def init_FAST_model(self):

        fst_vt = self.options['modeling_options']['openfast']['fst_vt']
        modeling_options = self.options['modeling_options']
        
        # Main .fst file`
        fst_vt['Fst']               = {}
        fst_vt['ElastoDyn']         = {}
        fst_vt['ElastoDynBlade']    = {}
        fst_vt['ElastoDynTower']    = {}
        fst_vt['AeroDyn15']         = {}
        fst_vt['AeroDynBlade']      = {}
        fst_vt['ServoDyn']          = {}
        fst_vt['InflowWind']        = {}
        fst_vt['SubDyn']            = {}
        fst_vt['HydroDyn']          = {}
        fst_vt['MoorDyn']           = {}
        fst_vt['MAP']               = {}

        for key in modeling_options['Level3']['simulation']:
            fst_vt['Fst'][key] = modeling_options['Level3']['simulation'][key]
            
        for key in modeling_options['Level3']['ElastoDyn']:
            fst_vt['ElastoDyn'][key] = modeling_options['Level3']['ElastoDyn'][key]
            
        for key in modeling_options['Level3']['ElastoDynBlade']:
            fst_vt['ElastoDynBlade'][key] = modeling_options['Level3']['ElastoDynBlade'][key]
            
        for key in modeling_options['Level3']['ElastoDynTower']:
            fst_vt['ElastoDynTower'][key] = modeling_options['Level3']['ElastoDynTower'][key]
            
        for key in modeling_options['Level3']['AeroDyn']:
            fst_vt['AeroDyn15'][key] = copy.copy(modeling_options['Level3']['AeroDyn'][key])
            
        for key in modeling_options['Level3']['InflowWind']:
            fst_vt['InflowWind'][key] = modeling_options['Level3']['InflowWind'][key]
            
        for key in modeling_options['Level3']['ServoDyn']:
            fst_vt['ServoDyn'][key] = modeling_options['Level3']['ServoDyn'][key]
            
        for key in modeling_options['Level3']['SubDyn']:
            fst_vt['SubDyn'][key] = modeling_options['Level3']['SubDyn'][key]
            
        for key in modeling_options['Level3']['HydroDyn']:
            fst_vt['HydroDyn'][key] = modeling_options['Level3']['HydroDyn'][key]
            
        for key in modeling_options['Level3']['MoorDyn']:
            fst_vt['MoorDyn'][key] = modeling_options['Level3']['MoorDyn'][key]
        
        for key1 in modeling_options['Level3']['outlist']:
                for key2 in modeling_options['Level3']['outlist'][key1]:
                    fst_vt['outlist'][key1][key2] = modeling_options['Level3']['outlist'][key1][key2]

        fst_vt['ServoDyn']['DLL_FileName'] = modeling_options['openfast']['file_management']['path2dll']

        if fst_vt['AeroDyn15']['IndToler'] == 0.:
            fst_vt['AeroDyn15']['IndToler'] = 'default'
        if fst_vt['AeroDyn15']['DTAero'] == 0.:
            fst_vt['AeroDyn15']['DTAero'] = 'default'
        if fst_vt['AeroDyn15']['OLAF']['DTfvw'] == 0.:
            fst_vt['AeroDyn15']['OLAF']['DTfvw'] = 'default'
        if fst_vt['ElastoDyn']['DT'] == 0.:
            fst_vt['ElastoDyn']['DT'] = 'default'

        return fst_vt

    def update_FAST_model(self, fst_vt, inputs, discrete_inputs):
        
        modeling_options = self.options['modeling_options']

        # Update fst_vt nested dictionary with data coming from WISDEM

        # Update ElastoDyn
        fst_vt['ElastoDyn']['NumBl']  = self.n_blades
        fst_vt['ElastoDyn']['TipRad'] = inputs['Rtip'][0]
        fst_vt['ElastoDyn']['HubRad'] = inputs['Rhub'][0]
        if discrete_inputs['rotor_orientation'] == 'upwind':
            k = -1.
        else:
            k = 1
        fst_vt['ElastoDyn']['PreCone(1)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(2)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(3)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['ShftTilt']   = k*inputs['tilt'][0]
        fst_vt['ElastoDyn']['OverHang']   = k*inputs['overhang'][0]
        fst_vt['ElastoDyn']['GBoxEff']    = inputs['gearbox_efficiency'][0] * 100.
        fst_vt['ElastoDyn']['GBRatio']    = inputs['gearbox_ratio'][0]

        # Update ServoDyn
        fst_vt['ServoDyn']['GenEff']       = float(inputs['generator_efficiency']/inputs['gearbox_efficiency']) * 100.
        fst_vt['ServoDyn']['PitManRat(1)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(2)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(3)'] = float(inputs['max_pitch_rate'])

        # Masses and inertias from DriveSE
        fst_vt['ElastoDyn']['HubMass']   = inputs['hub_system_mass'][0]
        fst_vt['ElastoDyn']['HubIner']   = inputs['hub_system_I'][0]
        fst_vt['ElastoDyn']['HubCM']     = inputs['hub_system_cm'][0] # k*inputs['overhang'][0] - inputs['hub_system_cm'][0], but we need to solve the circular dependency in DriveSE first
        fst_vt['ElastoDyn']['NacMass']   = inputs['above_yaw_mass'][0]
        fst_vt['ElastoDyn']['YawBrMass'] = inputs['yaw_mass'][0]
        fst_vt['ElastoDyn']['NacYIner']  = inputs['nacelle_I_TT'][2]
        fst_vt['ElastoDyn']['NacCMxn']   = -k*inputs['nacelle_cm'][0]
        fst_vt['ElastoDyn']['NacCMyn']   = inputs['nacelle_cm'][1]
        fst_vt['ElastoDyn']['NacCMzn']   = inputs['nacelle_cm'][2]
        fst_vt['ElastoDyn']['Twr2Shft']  = float(inputs['twr2shaft'])
        fst_vt['ElastoDyn']['GenIner']   = float(inputs['GenIner'])

        # Mass and inertia inputs
        fst_vt['ElastoDyn']['TipMass(1)'] = 0.
        fst_vt['ElastoDyn']['TipMass(2)'] = 0.
        fst_vt['ElastoDyn']['TipMass(3)'] = 0.

        tower_base_height = max(float(inputs['tower_base_height']), float(inputs["platform_center_of_mass"][2]))
        fst_vt['ElastoDyn']['TowerBsHt'] = tower_base_height # Height of tower base above ground level [onshore] or MSL [offshore] (meters)
        fst_vt['ElastoDyn']['TowerHt']   = float(inputs['hub_height']) - float(inputs['distance_tt_hub']) # Height of tower above ground level [onshore] or MSL [offshore] (meters)

        # TODO: There is some confusion on PtfmRefzt
        # DZ: based on the openfast r-tests:
        #   if this is floating, the z ref. point is 0.  Is this the reference that platform_center_of_mass is relative to?
        #   if fixed bottom, it's the tower base height.  
        if modeling_options['flags']['floating']:
            fst_vt['ElastoDyn']['PtfmMass'] = float(inputs["platform_mass"])
            fst_vt['ElastoDyn']['PtfmRIner'] = float(inputs["platform_I_total"][0])
            fst_vt['ElastoDyn']['PtfmPIner'] = float(inputs["platform_I_total"][1])
            fst_vt['ElastoDyn']['PtfmYIner'] = float(inputs["platform_I_total"][2])
            fst_vt['ElastoDyn']['PtfmCMxt'] = float(inputs["platform_center_of_mass"][0])
            fst_vt['ElastoDyn']['PtfmCMyt'] = float(inputs["platform_center_of_mass"][1])
            fst_vt['ElastoDyn']['PtfmCMzt'] = float(inputs["platform_center_of_mass"][2])
            fst_vt['ElastoDyn']['PtfmRefzt'] = 0. # Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)

        else:
            # Ptfm* can capture the transition piece for fixed-bottom, but we are doing that in subdyn, so only worry about getting height right
            fst_vt['ElastoDyn']['PtfmMass'] = 0.
            fst_vt['ElastoDyn']['PtfmRIner'] = 0.
            fst_vt['ElastoDyn']['PtfmPIner'] = 0.
            fst_vt['ElastoDyn']['PtfmYIner'] = 0.
            fst_vt['ElastoDyn']['PtfmCMxt'] = 0.
            fst_vt['ElastoDyn']['PtfmCMyt'] = 0.
            fst_vt['ElastoDyn']['PtfmCMzt'] = float(inputs['tower_base_height'])
            fst_vt['ElastoDyn']['PtfmRefzt'] = tower_base_height # Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)

        # Drivetrain inputs
        fst_vt['ElastoDyn']['DTTorSpr'] = float(inputs['drivetrain_spring_constant'])
        fst_vt['ElastoDyn']['DTTorDmp'] = float(inputs['drivetrain_damping_coefficient'])

        # Update Inflowwind
        fst_vt['InflowWind']['RefHt'] = float(inputs['hub_height'])
        fst_vt['InflowWind']['RefHt_Uni'] = float(inputs['hub_height'])
        fst_vt['InflowWind']['PLexp'] = float(inputs['shearExp'])
        if fst_vt['InflowWind']['NWindVel'] == 1:
            fst_vt['InflowWind']['WindVxiList'] = 0.
            fst_vt['InflowWind']['WindVyiList'] = 0.
            fst_vt['InflowWind']['WindVziList'] = float(inputs['hub_height'])
        else:
            raise Exception('The code only supports InflowWind NWindVel == 1')

        # Update ElastoDyn Tower Input File
        twr_elev  = inputs['tower_monopile_z']
        twr_index = np.argmin(abs(twr_elev - np.maximum(1.0, tower_base_height)))
        cd_index  = 0
        if twr_elev[twr_index] <= 1.:
            twr_index += 1
            cd_index  += 1
        fst_vt['AeroDyn15']['NumTwrNds'] = len(twr_elev[twr_index:])
        fst_vt['AeroDyn15']['TwrElev']   = twr_elev[twr_index:]
        fst_vt['AeroDyn15']['TwrDiam']   = inputs['tower_outer_diameter'][twr_index:]
        fst_vt['AeroDyn15']['TwrCd']     = inputs['tower_cd'][cd_index:]
        fst_vt['AeroDyn15']['TwrTI']     = np.ones(len(twr_elev[twr_index:])) * fst_vt['AeroDyn15']['TwrTI']

        z_tow = twr_elev[twr_index:]
        z_sec, _ = util.nodal2sectional(z_tow)
        sec_loc = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])
        fst_vt['ElastoDynTower']['NTwInpSt'] = len(sec_loc)
        fst_vt['ElastoDynTower']['HtFract']  = sec_loc
        fst_vt['ElastoDynTower']['TMassDen'] = inputs['mass_den'][twr_index:]
        fst_vt['ElastoDynTower']['TwFAStif'] = inputs['foreaft_stff'][twr_index:]
        fst_vt['ElastoDynTower']['TwSSStif'] = inputs['sideside_stff'][twr_index:]
        fst_vt['ElastoDynTower']['TwFAM1Sh'] = inputs['fore_aft_modes'][0, :]  / sum(inputs['fore_aft_modes'][0, :])
        fst_vt['ElastoDynTower']['TwFAM2Sh'] = inputs['fore_aft_modes'][1, :]  / sum(inputs['fore_aft_modes'][1, :])
        fst_vt['ElastoDynTower']['TwSSM1Sh'] = inputs['side_side_modes'][0, :] / sum(inputs['side_side_modes'][0, :])
        fst_vt['ElastoDynTower']['TwSSM2Sh'] = inputs['side_side_modes'][1, :] / sum(inputs['side_side_modes'][1, :])

        # Calculate yaw stiffness of tower (springs in series) and use in servodyn as yaw spring constant
        n_height_mon = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_monopile']
        k_tow_tor = inputs['tor_stff'][n_height_mon:] / np.diff(inputs['tower_monopile_z'][n_height_mon:])
        k_tow_tor = 1.0/np.sum(1.0/k_tow_tor)
        # R. Bergua's suggestion to set the stiffness to the tower torsional stiffness and the
        # damping to the frequency of the first tower torsional mode- easier than getting the yaw inertia right
        damp_ratio = 0.01
        f_torsion = float(inputs['tor_freq'])
        fst_vt['ServoDyn']['YawSpr'] = k_tow_tor
        if f_torsion > 0.0:
            fst_vt['ServoDyn']['YawDamp'] = damp_ratio * k_tow_tor / np.pi / f_torsion
        else:
            fst_vt['ServoDyn']['YawDamp'] = 2 * damp_ratio * np.sqrt(k_tow_tor * inputs['rna_I_TT'][2])
        
        # Update ElastoDyn Blade Input File
        fst_vt['ElastoDynBlade']['NBlInpSt']   = len(inputs['r'])
        fst_vt['ElastoDynBlade']['BlFract']    = (inputs['r']-inputs['Rhub'])/(inputs['Rtip']-inputs['Rhub'])
        fst_vt['ElastoDynBlade']['BlFract'][0] = 0.
        fst_vt['ElastoDynBlade']['BlFract'][-1]= 1.
        fst_vt['ElastoDynBlade']['PitchAxis']  = inputs['le_location']
        fst_vt['ElastoDynBlade']['StrcTwst']   = inputs['theta'] # to do: structural twist is not nessessarily (nor likely to be) the same as aero twist
        fst_vt['ElastoDynBlade']['BMassDen']   = inputs['beam:rhoA']
        fst_vt['ElastoDynBlade']['FlpStff']    = inputs['beam:EIyy']
        fst_vt['ElastoDynBlade']['EdgStff']    = inputs['beam:EIxx']
        fst_vt['ElastoDynBlade']['BldFl1Sh']   = np.zeros(5)
        fst_vt['ElastoDynBlade']['BldFl2Sh']   = np.zeros(5)
        fst_vt['ElastoDynBlade']['BldEdgSh']   = np.zeros(5)
        for i in range(5):
            fst_vt['ElastoDynBlade']['BldFl1Sh'][i] = inputs['flap_mode_shapes'][0,i] / sum(inputs['flap_mode_shapes'][0,:])
            fst_vt['ElastoDynBlade']['BldFl2Sh'][i] = inputs['flap_mode_shapes'][1,i] / sum(inputs['flap_mode_shapes'][1,:])
            fst_vt['ElastoDynBlade']['BldEdgSh'][i] = inputs['edge_mode_shapes'][0,i] / sum(inputs['edge_mode_shapes'][0,:])
        
        # Update AeroDyn15
        fst_vt['AeroDyn15']['AirDens']   = float(inputs['rho'])
        fst_vt['AeroDyn15']['KinVisc']   = inputs['mu'][0] / inputs['rho'][0]
        fst_vt['AeroDyn15']['SpdSound']  = float(inputs['speed_sound_air'])

        # Update OLAF
        if fst_vt['AeroDyn15']['WakeMod'] == 3:
            _, _, nNWPanel, nFWPanel, nFWPanelFree = OLAFParams(fst_vt['ElastoDyn']['RotSpeed'])
            fst_vt['AeroDyn15']['OLAF']['nNWPanel'] = nNWPanel
            fst_vt['AeroDyn15']['OLAF']['nFWPanel'] = nFWPanel
            fst_vt['AeroDyn15']['OLAF']['nFWPanelFree'] = nFWPanelFree

        # Update AeroDyn15 Blade Input File
        r = (inputs['r']-inputs['Rhub'])
        r[0]  = 0.
        r[-1] = inputs['Rtip']-inputs['Rhub']
        fst_vt['AeroDynBlade']['NumBlNds'] = self.n_span
        fst_vt['AeroDynBlade']['BlSpn']    = r
        fst_vt['AeroDynBlade']['BlCrvAC']  = inputs['ref_axis_blade'][:,0]
        fst_vt['AeroDynBlade']['BlSwpAC']  = inputs['ref_axis_blade'][:,1]
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(inputs['ref_axis_blade'][:,0])/np.gradient(r)))
        fst_vt['AeroDynBlade']['BlTwist']  = inputs['theta']
        fst_vt['AeroDynBlade']['BlChord']  = inputs['chord']
        fst_vt['AeroDynBlade']['BlAFID']   = np.asarray(range(1,self.n_span+1))

        # Update AeroDyn15 Airfoile Input Files
        # airfoils = inputs['airfoils']
        fst_vt['AeroDyn15']['NumAFfiles'] = self.n_span
        # fst_vt['AeroDyn15']['af_data'] = [{}]*len(airfoils)
        fst_vt['AeroDyn15']['af_data'] = []

        # Set the AD15 flag AFTabMod, deciding whether we use more Re per airfoil or user-defined tables (used for example in distributed aerodynamic control)
        if fst_vt['AeroDyn15']['AFTabMod'] == 1:
            # If AFTabMod is the default coming form the schema, check the value from WISDEM, which might be set to 2 if more Re per airfoil are defined in the geometry yaml 
            fst_vt['AeroDyn15']['AFTabMod'] = modeling_options["WISDEM"]["RotorSE"]["AFTabMod"]
        if self.n_tab > 1 and fst_vt['AeroDyn15']['AFTabMod'] == 1:
            fst_vt['AeroDyn15']['AFTabMod'] = 3
        elif self.n_tab > 1 and fst_vt['AeroDyn15']['AFTabMod'] == 2:
            raise Exception('OpenFAST does not support both multiple Re and multiple user defined tabs. Please remove DAC devices or Re polars')

        for i in range(self.n_span): # No of blade radial stations
        
            fst_vt['AeroDyn15']['af_data'].append([])
            
            if fst_vt['AeroDyn15']['AFTabMod'] == 1:
                loop_index = 1
            elif fst_vt['AeroDyn15']['AFTabMod'] == 2:
                loop_index = self.n_Re
            else:
                loop_index = self.n_tab

            for j in range(loop_index): # Number of tabs or Re
                if fst_vt['AeroDyn15']['AFTabMod'] == 1:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,0], inputs['airfoils_cd'][i,:,0,0], inputs['airfoils_cm'][i,:,0,0])
                elif fst_vt['AeroDyn15']['AFTabMod'] == 2:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,j,0], inputs['airfoils_cd'][i,:,j,0], inputs['airfoils_cm'][i,:,j,0])
                else:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,j], inputs['airfoils_cd'][i,:,0,j], inputs['airfoils_cm'][i,:,0,j])
                
                fst_vt['AeroDyn15']['af_data'][i].append({})


                fst_vt['AeroDyn15']['af_data'][i][j]['InterpOrd'] = "DEFAULT"
                fst_vt['AeroDyn15']['af_data'][i][j]['NonDimArea']= 1
                if modeling_options['openfast']['analysis_settings']['generate_af_coords']:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = '@"AF{:02d}_Coords.txt"'.format(i)
                else:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = 0
                
                fst_vt['AeroDyn15']['af_data'][i][j]['NumTabs']   = loop_index
                if fst_vt['AeroDyn15']['AFTabMod'] == 3:
                    fst_vt['AeroDyn15']['af_data'][i][j]['Ctrl'] = inputs['airfoils_Ctrl'][i,0,j]  # unsteady['Ctrl'] # added to unsteady function for variable flap controls at airfoils
                    fst_vt['AeroDyn15']['af_data'][i][j]['Re']   = inputs['airfoils_Re'][0] # If AFTabMod==3 the Re is neglected, but it still must be the same across tables
                else:
                    fst_vt['AeroDyn15']['af_data'][i][j]['Re']   = inputs['airfoils_Re'][j]
                    fst_vt['AeroDyn15']['af_data'][i][j]['Ctrl'] = 0.
                fst_vt['AeroDyn15']['af_data'][i][j]['InclUAdata']= "True"
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha0']    = unsteady['alpha0']
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha1']    = unsteady['alpha1']
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha2']    = unsteady['alpha2']
                fst_vt['AeroDyn15']['af_data'][i][j]['eta_e']     = unsteady['eta_e']
                fst_vt['AeroDyn15']['af_data'][i][j]['C_nalpha']  = unsteady['C_nalpha']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_f0']      = unsteady['T_f0']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_V0']      = unsteady['T_V0']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_p']       = unsteady['T_p']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_VL']      = unsteady['T_VL']
                fst_vt['AeroDyn15']['af_data'][i][j]['b1']        = unsteady['b1']
                fst_vt['AeroDyn15']['af_data'][i][j]['b2']        = unsteady['b2']
                fst_vt['AeroDyn15']['af_data'][i][j]['b5']        = unsteady['b5']
                fst_vt['AeroDyn15']['af_data'][i][j]['A1']        = unsteady['A1']
                fst_vt['AeroDyn15']['af_data'][i][j]['A2']        = unsteady['A2']
                fst_vt['AeroDyn15']['af_data'][i][j]['A5']        = unsteady['A5']
                fst_vt['AeroDyn15']['af_data'][i][j]['S1']        = unsteady['S1']
                fst_vt['AeroDyn15']['af_data'][i][j]['S2']        = unsteady['S2']
                fst_vt['AeroDyn15']['af_data'][i][j]['S3']        = unsteady['S3']
                fst_vt['AeroDyn15']['af_data'][i][j]['S4']        = unsteady['S4']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cn1']       = unsteady['Cn1']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cn2']       = unsteady['Cn2']
                fst_vt['AeroDyn15']['af_data'][i][j]['St_sh']     = unsteady['St_sh']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cd0']       = unsteady['Cd0']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cm0']       = unsteady['Cm0']
                fst_vt['AeroDyn15']['af_data'][i][j]['k0']        = unsteady['k0']
                fst_vt['AeroDyn15']['af_data'][i][j]['k1']        = unsteady['k1']
                fst_vt['AeroDyn15']['af_data'][i][j]['k2']        = unsteady['k2']
                fst_vt['AeroDyn15']['af_data'][i][j]['k3']        = unsteady['k3']
                fst_vt['AeroDyn15']['af_data'][i][j]['k1_hat']    = unsteady['k1_hat']
                fst_vt['AeroDyn15']['af_data'][i][j]['x_cp_bar']  = unsteady['x_cp_bar']
                fst_vt['AeroDyn15']['af_data'][i][j]['UACutout']  = unsteady['UACutout']
                fst_vt['AeroDyn15']['af_data'][i][j]['filtCutOff']= unsteady['filtCutOff']
                fst_vt['AeroDyn15']['af_data'][i][j]['NumAlf']    = len(unsteady['Alpha'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Alpha']     = np.array(unsteady['Alpha'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cl']        = np.array(unsteady['Cl'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cd']        = np.array(unsteady['Cd'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cm']        = np.array(unsteady['Cm'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cpmin']     = np.zeros_like(unsteady['Cm'])        
        
        fst_vt['AeroDyn15']['af_coord'] = []
        fst_vt['AeroDyn15']['rthick']   = np.zeros(self.n_span)
        fst_vt['AeroDyn15']['ac']   = np.zeros(self.n_span)
        for i in range(self.n_span):
            fst_vt['AeroDyn15']['af_coord'].append({})
            fst_vt['AeroDyn15']['af_coord'][i]['x']  = inputs['coord_xy_interp'][i,:,0]
            fst_vt['AeroDyn15']['af_coord'][i]['y']  = inputs['coord_xy_interp'][i,:,1]
            fst_vt['AeroDyn15']['rthick'][i]         = inputs['rthick'][i]
            fst_vt['AeroDyn15']['ac'][i]             = inputs['ac'][i]
                
        # # AeroDyn blade spanwise output positions
        r_out_target  = [0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        r = r/r[-1]
        idx_out       = [np.argmin(abs(r-ri)) for ri in r_out_target]
        self.R_out_AD = [fst_vt['AeroDynBlade']['BlSpn'][i] for i in idx_out]
        if len(self.R_out_AD) != len(np.unique(self.R_out_AD)):
            raise Exception('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        fst_vt['AeroDyn15']['BlOutNd']  = [str(idx+1) for idx in idx_out]
        fst_vt['AeroDyn15']['NBlOuts']  = len(idx_out)

        # ElastoDyn blade spanwise output positions
        nBldNodes     = fst_vt['ElastoDyn']['BldNodes']
        bld_fract     = np.arange(1./nBldNodes/2., 1, 1./nBldNodes)
        idx_out       = [np.argmin(abs(bld_fract-ri)) for ri in r_out_target]
        r_nodes       = bld_fract*(fst_vt['ElastoDyn']['TipRad']-fst_vt['ElastoDyn']['HubRad']) + fst_vt['ElastoDyn']['HubRad']
        self.R_out_ED_bl = np.hstack((fst_vt['ElastoDyn']['HubRad'], [r_nodes[i] for i in idx_out]))
        if len(self.R_out_ED_bl) != len(np.unique(self.R_out_ED_bl)):
            raise Exception('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        fst_vt['ElastoDyn']['BldGagNd'] = [idx+1 for idx in idx_out]
        fst_vt['ElastoDyn']['NBlGages'] = len(idx_out)

        # ElastoDyn tower output positions along height
        fst_vt['ElastoDyn']['NTwGages'] = 9
        nTwrNodes = fst_vt['ElastoDyn']['TwrNodes']
        twr_fract = np.arange(1./nTwrNodes/2., 1, 1./nTwrNodes)
        idx_out = [np.argmin(abs(twr_fract-ri)) for ri in r_out_target]
        fst_vt['ElastoDyn']['TwrGagNd'] = [idx+1 for idx in idx_out]
        fst_vt['AeroDyn15']['NTwOuts'] = 0
        self.Z_out_ED_twr = np.hstack((0., [twr_fract[i] for i in idx_out], 1.))

        # SubDyn inputs- monopile and floating
        if modeling_options['flags']['monopile']:
            mono_index = twr_index+1 # Duplicate intersection point
            n_joints = len(inputs['tower_outer_diameter'][1:mono_index]) # Omit submerged pile
            n_members = n_joints - 1
            itrans = n_joints - 1
            fst_vt['SubDyn']['JointXss'] = np.zeros( n_joints )
            fst_vt['SubDyn']['JointYss'] = np.zeros( n_joints )
            fst_vt['SubDyn']['JointZss'] = twr_elev[1:mono_index]
            fst_vt['SubDyn']['NReact'] = 1
            fst_vt['SubDyn']['RJointID'] = [1]
            fst_vt['SubDyn']['RctTDXss'] = fst_vt['SubDyn']['RctTDYss'] = fst_vt['SubDyn']['RctTDZss'] = [1]
            fst_vt['SubDyn']['RctRDXss'] = fst_vt['SubDyn']['RctRDYss'] = fst_vt['SubDyn']['RctRDZss'] = [1]
            fst_vt['SubDyn']['IJointID'] = [n_joints]
            fst_vt['SubDyn']['MJointID1'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['MJointID2'] = np.arange( n_members, dtype=np.int_ ) + 2
            fst_vt['SubDyn']['YoungE1'] = inputs['tower_E'][1:mono_index]
            fst_vt['SubDyn']['ShearG1'] = inputs['tower_G'][1:mono_index]
            fst_vt['SubDyn']['MatDens1'] = inputs['tower_rho'][1:mono_index]
            fst_vt['SubDyn']['XsecD'] = util.nodal2sectional(inputs['tower_outer_diameter'][1:mono_index])[0]
            fst_vt['SubDyn']['XsecT'] = inputs['tower_wall_thickness'][1:mono_index]
            
            # Find the members where the 9 channels of SubDyn should be placed
            grid_joints_monopile = (fst_vt['SubDyn']['JointZss'] - fst_vt['SubDyn']['JointZss'][0]) / (fst_vt['SubDyn']['JointZss'][-1] - fst_vt['SubDyn']['JointZss'][0])
            n_channels = 9
            grid_target = np.linspace(0., 1., n_channels)
            # Take the first node for every member, except for last one
            idx_out = [np.argmin(abs(grid_joints_monopile-grid_i)) for grid_i in grid_target]
            fst_vt['SubDyn']['NMOutputs'] = n_channels
            fst_vt['SubDyn']['MemberID_out'] = [idx+1 for idx in idx_out]
            fst_vt['SubDyn']['MemberID_out'][-1] -= 1 
            fst_vt['SubDyn']['NOutCnt'] = np.ones_like(fst_vt['SubDyn']['MemberID_out'])
            fst_vt['SubDyn']['NodeCnt'] = np.ones_like(fst_vt['SubDyn']['MemberID_out'])
            fst_vt['SubDyn']['NodeCnt'][-1] = 2
            self.Z_out_SD_mpl = [grid_joints_monopile[i] for i in idx_out]
                
        elif modeling_options['flags']['floating']:
            joints_xyz = inputs["platform_nodes"]
            n_joints = np.where(joints_xyz[:, 0] == NULL)[0][0]
            joints_xyz = joints_xyz[:n_joints, :]
            itrans = util.closest_node(joints_xyz, inputs["transition_node"])

            N1 = np.int_(inputs["platform_elem_n1"])
            n_members = np.where(N1 == NULL)[0][0]
            N1 = N1[:n_members]
            N2 = np.int_(inputs["platform_elem_n2"][:n_members])
                
            fst_vt['SubDyn']['JointXss'] = joints_xyz[:,0]
            fst_vt['SubDyn']['JointYss'] = joints_xyz[:,1]
            fst_vt['SubDyn']['JointZss'] = joints_xyz[:,2]
            fst_vt['SubDyn']['NReact'] = 0
            fst_vt['SubDyn']['RJointID'] = []
            fst_vt['SubDyn']['RctTDXss'] = fst_vt['SubDyn']['RctTDYss'] = fst_vt['SubDyn']['RctTDZss'] = []
            fst_vt['SubDyn']['RctRDXss'] = fst_vt['SubDyn']['RctRDYss'] = fst_vt['SubDyn']['RctRDZss'] = []
            if modeling_options['floating']['transition_joint'] is None:
                fst_vt['SubDyn']['IJointID'] = []
            else:
                fst_vt['SubDyn']['IJointID'] = [itrans+1]
            fst_vt['SubDyn']['MJointID1'] = N1+1
            fst_vt['SubDyn']['MJointID2'] = N2+1
            
            fst_vt['SubDyn']['YoungE1'] = inputs["platform_elem_E"][:n_members]
            fst_vt['SubDyn']['ShearG1'] = inputs["platform_elem_G"][:n_members]
            fst_vt['SubDyn']['MatDens1'] = inputs["platform_elem_rho"][:n_members]
            fst_vt['SubDyn']['XsecD'] = inputs["platform_elem_D"][:n_members]
            fst_vt['SubDyn']['XsecT'] = inputs["platform_elem_t"][:n_members]
            
        # SubDyn inputs- offshore generic
        if modeling_options['flags']['offshore']:
            if fst_vt['SubDyn']['SDdeltaT']<=-999.0: fst_vt['SubDyn']['SDdeltaT'] = "DEFAULT"
            fst_vt['SubDyn']['JDampings'] = [str(m) for m in fst_vt['SubDyn']['JDampings']]
            fst_vt['SubDyn']['Rct_SoilFile'] = ['']
            fst_vt['SubDyn']['NJoints'] = n_joints
            fst_vt['SubDyn']['JointID'] = np.arange( n_joints, dtype=np.int_) + 1
            fst_vt['SubDyn']['ItfTDXss'] = fst_vt['SubDyn']['ItfTDYss'] = fst_vt['SubDyn']['ItfTDZss'] = [1]
            fst_vt['SubDyn']['ItfRDXss'] = fst_vt['SubDyn']['ItfRDYss'] = fst_vt['SubDyn']['ItfRDZss'] = [1]
            fst_vt['SubDyn']['NMembers'] = n_members
            fst_vt['SubDyn']['MemberID'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['MPropSetID1'] = fst_vt['SubDyn']['MPropSetID2'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['NPropSets'] = n_members
            fst_vt['SubDyn']['PropSetID1'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['NCOSMs'] = 0
            fst_vt['SubDyn']['NXPropSets'] = 0
            fst_vt['SubDyn']['NCmass'] = 2 if inputs['gravity_foundation_mass'] > 0.0 else 1
            fst_vt['SubDyn']['CMJointID'] = [itrans+1]
            fst_vt['SubDyn']['JMass'] = [float(inputs['transition_piece_mass'])]
            fst_vt['SubDyn']['JMXX'] = [inputs['transition_piece_I'][0]]
            fst_vt['SubDyn']['JMYY'] = [inputs['transition_piece_I'][1]]
            fst_vt['SubDyn']['JMZZ'] = [inputs['transition_piece_I'][2]]
            if inputs['gravity_foundation_mass'] > 0.0:
                fst_vt['SubDyn']['CMJointID'] += [1]
                fst_vt['SubDyn']['JMass'] += [float(inputs['gravity_foundation_mass'])]
                fst_vt['SubDyn']['JMXX'] += [inputs['gravity_foundation_I'][0]]
                fst_vt['SubDyn']['JMYY'] += [inputs['gravity_foundation_I'][1]]
                fst_vt['SubDyn']['JMZZ'] += [inputs['gravity_foundation_I'][2]]
        

        # HydroDyn inputs
        if modeling_options['flags']['offshore']:
            fst_vt['HydroDyn']['WtrDens'] = float(inputs['rho_water'])
            fst_vt['HydroDyn']['WtrDpth'] = float(inputs['water_depth'])
            fst_vt['HydroDyn']['MSL2SWL'] = 0
            fst_vt['HydroDyn']['WaveHs'] = float(inputs['Hsig_wave'])
            fst_vt['HydroDyn']['WaveTp'] = float(inputs['Tsig_wave'])
            if fst_vt['HydroDyn']['WavePkShp']<=-999.0: fst_vt['HydroDyn']['WavePkShp'] = "DEFAULT"
            fst_vt['HydroDyn']['WaveDir'] = float(inputs['beta_wave'])
            fst_vt['HydroDyn']['WaveDirRange'] = np.rad2deg(fst_vt['HydroDyn']['WaveDirRange'])
            fst_vt['HydroDyn']['WaveElevxi'] = [str(m) for m in fst_vt['HydroDyn']['WaveElevxi']]
            fst_vt['HydroDyn']['WaveElevyi'] = [str(m) for m in fst_vt['HydroDyn']['WaveElevyi']]
            fst_vt['HydroDyn']['CurrSSDir'] = "DEFAULT" if fst_vt['HydroDyn']['CurrSSDir']<=-999.0 else np.rad2deg(fst_vt['HydroDyn']['CurrSSDir']) 
            fst_vt['HydroDyn']['AddF0'] = np.array( fst_vt['HydroDyn']['AddF0'] )
            fst_vt['HydroDyn']['AddCLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddCLin'+str(m+1)] for m in range(6)]) )
            fst_vt['HydroDyn']['AddBLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddBLin'+str(m+1)] for m in range(6)]) )
            fst_vt['HydroDyn']['AddBQuad'] = np.vstack( tuple([fst_vt['HydroDyn']['AddBQuad'+str(m+1)] for m in range(6)]) )
            fst_vt['HydroDyn']['NAxCoef'] = 1
            fst_vt['HydroDyn']['AxCoefID'] = 1 + np.arange( fst_vt['HydroDyn']['NAxCoef'], dtype=np.int_)
            fst_vt['HydroDyn']['AxCd'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
            fst_vt['HydroDyn']['AxCa'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
            fst_vt['HydroDyn']['AxCp'] = np.ones( fst_vt['HydroDyn']['NAxCoef'] )
            fst_vt['HydroDyn']['NJoints'] = fst_vt['SubDyn']['NJoints']
            fst_vt['HydroDyn']['JointID'] = fst_vt['SubDyn']['JointID']
            fst_vt['HydroDyn']['Jointxi'] = fst_vt['SubDyn']['JointXss']
            fst_vt['HydroDyn']['Jointyi'] = fst_vt['SubDyn']['JointYss']
            fst_vt['HydroDyn']['Jointzi'] = fst_vt['SubDyn']['JointZss']
            fst_vt['HydroDyn']['JointAxID'] = np.ones( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['JointOvrlp'] = np.zeros( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['NPropSets'] = fst_vt['SubDyn']['NPropSets']
            fst_vt['HydroDyn']['PropSetID'] = fst_vt['SubDyn']['PropSetID1']
            fst_vt['HydroDyn']['PropD'] = fst_vt['SubDyn']['XsecD']
            fst_vt['HydroDyn']['PropThck'] = fst_vt['SubDyn']['XsecT']

            fst_vt['HydroDyn']['SimplCd'] = 1.0
            fst_vt['HydroDyn']['SimplCdMG'] = 1.0
            fst_vt['HydroDyn']['SimplCa'] = 1.0
            fst_vt['HydroDyn']['SimplCaMG'] = 1.0
            fst_vt['HydroDyn']['SimplCp'] = 1.0
            fst_vt['HydroDyn']['SimplCpMG'] = 1.0
            fst_vt['HydroDyn']['SimplAxCa'] = 1.0
            fst_vt['HydroDyn']['SimplAxCaMG'] = 1.0
            fst_vt['HydroDyn']['SimplAxCp'] = 1.0
            fst_vt['HydroDyn']['SimplAxCpMG'] = 1.0
            fst_vt['HydroDyn']['NCoefDpth'] = 0
            fst_vt['HydroDyn']['NCoefMembers'] = 0
            fst_vt['HydroDyn']['NMembers'] = fst_vt['SubDyn']['NMembers']
            fst_vt['HydroDyn']['MemberID'] = fst_vt['SubDyn']['MemberID']
            fst_vt['HydroDyn']['MJointID1'] = fst_vt['SubDyn']['MJointID1']
            fst_vt['HydroDyn']['MJointID2'] = fst_vt['SubDyn']['MJointID2']
            fst_vt['HydroDyn']['MPropSetID1'] = fst_vt['SubDyn']['MPropSetID1']
            fst_vt['HydroDyn']['MPropSetID2'] = fst_vt['SubDyn']['MPropSetID2']
            fst_vt['HydroDyn']['MDivSize'] = 0.5*np.ones( fst_vt['HydroDyn']['NMembers'] )
            fst_vt['HydroDyn']['MCoefMod'] = np.ones( fst_vt['HydroDyn']['NMembers'], dtype=np.int_)
            fst_vt['HydroDyn']['PropPot'] = ['FALSE']* fst_vt['HydroDyn']['NMembers']
            fst_vt['HydroDyn']['NFillGroups'] = 0
            fst_vt['HydroDyn']['NMGDepths'] = 0

            # HARDCODED for IEA-15MW  # TODO : resolve these hardcodings
            fst_vt['HydroDyn']['NJoints'] = 2
            fst_vt['HydroDyn']['JointID'] = np.array([1,2])
            fst_vt['HydroDyn']['Jointxi'] = np.array([0.0,0.0])
            fst_vt['HydroDyn']['Jointyi'] = np.array([0.0,0.0])
            fst_vt['HydroDyn']['Jointzi'] = np.array([-13.18,-14.18])

            fst_vt['HydroDyn']['NPropSets'] = 1
            fst_vt['HydroDyn']['PropSetID'] = np.array([1])
            fst_vt['HydroDyn']['PropD']     = np.array([1e-5])
            fst_vt['HydroDyn']['PropThck']  = np.array([1e-6])

            fst_vt['HydroDyn']['NMembers']      = 1
            fst_vt['HydroDyn']['MemberID']      = np.array([1])
            fst_vt['HydroDyn']['MJointID1']     = np.array([1])
            fst_vt['HydroDyn']['MJointID2']     = np.array([2])
            fst_vt['HydroDyn']['MPropSetID1']   = np.array([1])
            fst_vt['HydroDyn']['MPropSetID2']   = np.array([1])
            fst_vt['HydroDyn']['MDivSize']      = np.array([1.0])
            fst_vt['HydroDyn']['MCoefMod']      = np.ones( fst_vt['HydroDyn']['NMembers'], dtype=np.int_)
            fst_vt['HydroDyn']['PropPot']       = ['True']* fst_vt['HydroDyn']['NMembers']

            # set PotFile directory relative to WEIS
            # we're probably going to have to copy these files in aeroelasticse when we start changing them each iteration
            weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if fst_vt['HydroDyn']['PotFile']:
                fst_vt['HydroDyn']['PotFile'] = os.path.join(weis_dir, fst_vt['HydroDyn']['PotFile'])

            fst_vt['HydroDyn']['PtfmVol0'] = float(inputs['platform_displacement'])
            
        # Moordyn inputs
        if modeling_options["flags"]["mooring"]:
            mooropt = modeling_options["mooring"]
            # Creating a line type for each line, regardless of whether it is unique or not
            n_lines = mooropt["n_lines"]
            line_names = ['line'+str(m) for m in range(n_lines)]
            fst_vt['MoorDyn']['NTypes'] = n_lines
            fst_vt['MoorDyn']['Name'] = fst_vt['MAP']['LineType'] = line_names
            fst_vt['MoorDyn']['Diam'] = fst_vt['MAP']['Diam'] = inputs["line_diameter"]
            fst_vt['MoorDyn']['MassDen'] = fst_vt['MAP']['MassDenInAir'] = inputs["line_mass_density"]
            fst_vt['MoorDyn']['EA'] = inputs["line_stiffness"]
            fst_vt['MoorDyn']['BA_zeta'] = -1*np.ones(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['Can'] = inputs["line_transverse_added_mass"]
            fst_vt['MoorDyn']['Cat'] = inputs["line_tangential_added_mass"]
            fst_vt['MoorDyn']['Cdn'] = inputs["line_transverse_drag"]
            fst_vt['MoorDyn']['Cdt'] = inputs["line_tangential_drag"]

            n_nodes = mooropt["n_nodes"]
            fst_vt['MoorDyn']['NConnects'] = n_nodes
            fst_vt['MoorDyn']['Node'] = np.arange(n_nodes)+1
            fst_vt['MoorDyn']['Type'] = mooropt["node_type"][:]
            fst_vt['MoorDyn']['X'] = inputs['nodes_location_full'][:,0]
            fst_vt['MoorDyn']['Y'] = inputs['nodes_location_full'][:,1]
            fst_vt['MoorDyn']['Z'] = inputs['nodes_location_full'][:,2]
            fst_vt['MoorDyn']['M'] = inputs['nodes_mass']
            fst_vt['MoorDyn']['V'] = inputs['nodes_volume']
            fst_vt['MoorDyn']['FX'] = np.zeros( n_nodes )
            fst_vt['MoorDyn']['FY'] = np.zeros( n_nodes )
            fst_vt['MoorDyn']['FZ'] = np.zeros( n_nodes )
            fst_vt['MoorDyn']['CdA'] = inputs['nodes_drag_area']
            fst_vt['MoorDyn']['CA'] = inputs['nodes_added_mass']

            fst_vt['MoorDyn']['NLines'] = n_lines
            fst_vt['MoorDyn']['Line'] = np.arange(n_lines)+1
            fst_vt['MoorDyn']['LineType'] = line_names
            fst_vt['MoorDyn']['UnstrLen'] = inputs['unstretched_length']
            fst_vt['MoorDyn']['NumSegs'] = 50*np.ones(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['NodeAnch'] = np.zeros(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['NodeFair'] = np.zeros(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['Outputs'] = ['-'] * n_lines
            fst_vt['MoorDyn']['CtrlChan'] = np.zeros(n_lines, dtype=np.int64)
            
            for k in range(n_lines):
                id1 = discrete_inputs['node_names'].index( mooropt["node1"][k] )
                id2 = discrete_inputs['node_names'].index( mooropt["node2"][k] )
                if (fst_vt['MoorDyn']['Type'][id1].lower() == 'vessel' and
                    'fix' in fst_vt['MoorDyn']['Type'][id2].lower()):
                    fst_vt['MoorDyn']['NodeFair'][k] = id1+1
                    fst_vt['MoorDyn']['NodeAnch'][k] = id2+1
                if (fst_vt['MoorDyn']['Type'][id2].lower() == 'vessel' and
                    'fix' in fst_vt['MoorDyn']['Type'][id1].lower()):
                    fst_vt['MoorDyn']['NodeFair'][k] = id2+1
                    fst_vt['MoorDyn']['NodeAnch'][k] = id1+1
                if (fst_vt['MoorDyn']['Type'][id2].lower() == 'vessel' and
                    'fix' in fst_vt['MoorDyn']['Type'][id1].lower()):
                    fst_vt['MoorDyn']['NodeFair'][k] = id2+1
                    fst_vt['MoorDyn']['NodeAnch'][k] = id1+1
                else:
                    print(discrete_inputs['node_names'])
                    print(mooropt["node1"][k], mooropt["node2"][k])
                    print(fst_vt['MoorDyn']['Type'][id1], fst_vt['MoorDyn']['Type'][id2])
                    raise ValueError('Mooring line seems to be between unknown endpoint types.')
                    
            for key in fst_vt['MoorDyn']:
                fst_vt['MAP'][key] = copy.copy(fst_vt['MoorDyn'][key])
                
            fst_vt['MAP']['LineType'] = ['main', 'main', 'main']
            
            for idx, node_type in enumerate(fst_vt['MAP']['Type']):
                if node_type == 'fixed':
                    fst_vt['MAP']['Type'][idx] = 'fix'
            
            # TODO: FIXME: these values are hardcoded for the IEA15MW linearization studies
            fst_vt['MAP']['CB'] = np.ones(n_lines)
            fst_vt['MAP']['CIntDamp'] = np.zeros(n_lines)
            fst_vt['MAP']['Ca'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdn'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdt'] = np.zeros(n_lines)
            fst_vt['MAP']['B'] = np.zeros( n_nodes )
            fst_vt['MAP']['Option'] = ["outer_tol 1e-5"]
            
        return fst_vt

    def run_FAST(self, inputs, discrete_inputs, fst_vt):

        case_list      = []
        case_name_list = []
        dlc_list       = []

        if self.FASTpref['dlc_settings']['run_IEC'] or self.FASTpref['dlc_settings']['run_blade_fatigue']:
            case_list_IEC, case_name_list_IEC, dlc_list_IEC = self.DLC_creation_IEC(inputs, discrete_inputs, fst_vt)
            case_list      += case_list_IEC
            case_name_list += case_name_list_IEC
            dlc_list       += dlc_list_IEC

        if self.FASTpref['dlc_settings']['run_power_curve']:
            case_list_pc, case_name_list_pc, dlc_list_pc = self.DLC_creation_powercurve(inputs, discrete_inputs, fst_vt)
            case_list      += case_list_pc
            case_name_list += case_name_list_pc
            dlc_list       += dlc_list_pc

        # Mandatory output channels to include 
        # TODO: what else is needed here?
        channels_out  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2"]
        channels_out += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2"]
        channels_out += ["TipDxb1", "TipDyb1", "TipDzb1", "TipDxb2", "TipDyb2", "TipDzb2"]
        channels_out += ["RootMxb1", "RootMyb1", "RootMzb1", "RootMxb2", "RootMyb2", "RootMzb2"]
        channels_out += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2"]
        channels_out += ["RootFxb1", "RootFyb1", "RootFzb1", "RootFxb2", "RootFyb2", "RootFzb2"]
        channels_out += ["Spn1FLzb1", "Spn2FLzb1", "Spn3FLzb1", "Spn4FLzb1", "Spn5FLzb1", "Spn6FLzb1", "Spn7FLzb1", "Spn8FLzb1", "Spn9FLzb1"]
        channels_out += ["Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1"]
        channels_out += ["Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
        channels_out += ["Spn1FLzb2", "Spn2FLzb2", "Spn3FLzb2", "Spn4FLzb2", "Spn5FLzb2", "Spn6FLzb2", "Spn7FLzb2", "Spn8FLzb2", "Spn9FLzb2"]
        channels_out += ["Spn1MLxb2", "Spn2MLxb2", "Spn3MLxb2", "Spn4MLxb2", "Spn5MLxb2", "Spn6MLxb2", "Spn7MLxb2", "Spn8MLxb2", "Spn9MLxb2"]
        channels_out += ["Spn1MLyb2", "Spn2MLyb2", "Spn3MLyb2", "Spn4MLyb2", "Spn5MLyb2", "Spn6MLyb2", "Spn7MLyb2", "Spn8MLyb2", "Spn9MLyb2"]
        channels_out += ["Spn1FLzb3", "Spn2FLzb3", "Spn3FLzb3", "Spn4FLzb3", "Spn5FLzb3", "Spn6FLzb3", "Spn7FLzb3", "Spn8FLzb3", "Spn9FLzb3"]
        channels_out += ["Spn1MLxb3", "Spn2MLxb3", "Spn3MLxb3", "Spn4MLxb3", "Spn5MLxb3", "Spn6MLxb3", "Spn7MLxb3", "Spn8MLxb3", "Spn9MLxb3"]
        channels_out += ["Spn1MLyb3", "Spn2MLyb3", "Spn3MLyb3", "Spn4MLyb3", "Spn5MLyb3", "Spn6MLyb3", "Spn7MLyb3", "Spn8MLyb3", "Spn9MLyb3"]
        channels_out += ["RtAeroCp", "RtAeroCt"]
        channels_out += ["RotSpeed", "GenSpeed", "NacYaw", "Azimuth"]
        channels_out += ["GenPwr", "GenTq", "BldPitch1", "BldPitch2", "BldPitch3"]
        channels_out += ["Wind1VelX", "Wind1VelY", "Wind1VelZ"]
        channels_out += ["RtVAvgxh", "RtVAvgyh", "RtVAvgzh"]
        channels_out += ["TwrBsFxt",  "TwrBsFyt", "TwrBsFzt", "TwrBsMxt",  "TwrBsMyt", "TwrBsMzt"]
        channels_out += ["YawBrFxp", "YawBrFyp", "YawBrFzp", "YawBrMxp", "YawBrMyp", "YawBrMzp"]
        channels_out += ["TwHt1FLxt", "TwHt2FLxt", "TwHt3FLxt", "TwHt4FLxt", "TwHt5FLxt", "TwHt6FLxt", "TwHt7FLxt", "TwHt8FLxt", "TwHt9FLxt"]
        channels_out += ["TwHt1FLyt", "TwHt2FLyt", "TwHt3FLyt", "TwHt4FLyt", "TwHt5FLyt", "TwHt6FLyt", "TwHt7FLyt", "TwHt8FLyt", "TwHt9FLyt"]
        channels_out += ["TwHt1FLzt", "TwHt2FLzt", "TwHt3FLzt", "TwHt4FLzt", "TwHt5FLzt", "TwHt6FLzt", "TwHt7FLzt", "TwHt8FLzt", "TwHt9FLzt"]
        channels_out += ["TwHt1MLxt", "TwHt2MLxt", "TwHt3MLxt", "TwHt4MLxt", "TwHt5MLxt", "TwHt6MLxt", "TwHt7MLxt", "TwHt8MLxt", "TwHt9MLxt"]
        channels_out += ["TwHt1MLyt", "TwHt2MLyt", "TwHt3MLyt", "TwHt4MLyt", "TwHt5MLyt", "TwHt6MLyt", "TwHt7MLyt", "TwHt8MLyt", "TwHt9MLyt"]
        channels_out += ["TwHt1MLzt", "TwHt2MLzt", "TwHt3MLzt", "TwHt4MLzt", "TwHt5MLzt", "TwHt6MLzt", "TwHt7MLzt", "TwHt8MLzt", "TwHt9MLzt"]
        channels_out += ["RtAeroFxh", "RtAeroFyh", "RtAeroFzh"]
        channels_out += ["RotThrust", "LSShftFys", "LSShftFzs", "RotTorq", "LSSTipMys", "LSSTipMzs"]
        channels_out += ["B1N1Alpha", "B1N2Alpha", "B1N3Alpha", "B1N4Alpha", "B1N5Alpha", "B1N6Alpha", "B1N7Alpha", "B1N8Alpha", "B1N9Alpha", "B2N1Alpha", "B2N2Alpha", "B2N3Alpha", "B2N4Alpha", "B2N5Alpha", "B2N6Alpha", "B2N7Alpha", "B2N8Alpha","B2N9Alpha"]
        channels_out += ["PtfmSurge", "PtfmSway", "PtfmHeave", "PtfmRoll", "PtfmPitch", "PtfmYaw","NcIMURAys"]
        if self.n_blades == 3:
            channels_out += ["TipDxc3", "TipDyc3", "TipDzc3", "RootMxc3", "RootMyc3", "RootMzc3", "TipDxb3", "TipDyb3", "TipDzb3", "RootMxb3",
                             "RootMyb3", "RootMzb3", "RootFxc3", "RootFyc3", "RootFzc3", "RootFxb3", "RootFyb3", "RootFzb3", "BldPitch3"]
            channels_out += ["B3N1Alpha", "B3N2Alpha", "B3N3Alpha", "B3N4Alpha", "B3N5Alpha", "B3N6Alpha", "B3N7Alpha", "B3N8Alpha", "B3N9Alpha"]

        # Channels for distributed aerodynamic control
        if self.n_tab > 1:
            channels_out += ['BLFLAP1', 'BLFLAP2', 'BLFLAP3']

        # Channels for monopile-based structure
        if self.options['modeling_options']['flags']['monopile']:
            k=1
            for i in range(len(self.Z_out_SD_mpl)):
                if k==9:
                    Node=2
                else:
                    Node=1
                channels_out += ["M" + str(k) + "N" + str(Node) + "FKxe"]
                channels_out += ["M" + str(k) + "N" + str(Node) + "FKye"]
                channels_out += ["M" + str(k) + "N" + str(Node) + "FKze"]
                channels_out += ["M" + str(k) + "N" + str(Node) + "MKxe"]
                channels_out += ["M" + str(k) + "N" + str(Node) + "MKye"]
                channels_out += ["M" + str(k) + "N" + str(Node) + "MKze"]
                channels_out += ['ReactFXss', 'ReactFYss', 'ReactFZss', 'ReactMXss', 'ReactMYss', 'ReactMZss']
                k+=1

        # Floating output channels
        if self.options['modeling_options']['flags']['floating']:
            channels_out += ["PtfmPitch", "PtfmRoll", "PtfmYaw", "PtfmSurge", "PtfmSway", "PtfmHeave"]
            
        channels = {}
        for var in channels_out:
            channels[var] = True

        # FAST wrapper setup
        # JJ->DZ: here is the first point in logic for linearization 
        if self.options['modeling_options']['Level2']['flag']:
            linearization_options               = self.options['modeling_options']['Level2']['linearization']
            
            # Use openfast binary until library works
            fastBatch                           = LinearFAST(**linearization_options)
            fastBatch.FAST_lib                  = None      # linearization not working with library
            fastBatch.FAST_exe                  = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../local/bin/openfast')
            fastBatch.fst_vt                    = fst_vt
            fastBatch.cores                     = self.cores

            case_list, case_name_list = fastBatch.gen_linear_cases(inputs)
        else:
            fastBatch = runFAST_pywrapper_batch()
        fastBatch.channels = channels

        # JJ->DZ: we need to add the options and settings from `gen_linear_model` here
        if self.FASTpref['file_management']['FAST_exe'] != 'none':
            fastBatch.FAST_exe          = self.FAST_exe
            fastBatch.FAST_lib          = self.FAST_lib
        fastBatch.FAST_runDirectory = self.FAST_runDirectory
        fastBatch.FAST_InputFile    = self.FAST_InputFile
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.debug_level       = self.debug_level
        fastBatch.fst_vt            = fst_vt
        fastBatch.keep_time         = False
        fastBatch.post              = FAST_IO_timeseries

        fastBatch.case_list         = case_list
        fastBatch.case_name_list    = case_name_list
        fastBatch.channels          = channels

        fastBatch.overwrite_outfiles = False  #<--- Debugging only, set to False to prevent OpenFAST from running if the .outb already exists

        # Run FAST
        if self.mpi_run and self.options['opt_options']['driver']['optimization']['flag']:
            summary_stats, extreme_table, DELs, chan_time = fastBatch.run_mpi(self.mpi_comm_map_down)
        else:
            if self.cores == 1:
                summary_stats, extreme_table, DELs, chan_time = fastBatch.run_serial()
            else:
                summary_stats, extreme_table, DELs, chan_time = fastBatch.run_multi(self.cores)

        self.fst_vt = fst_vt
        self.of_inumber = self.of_inumber + 1
        sys.stdout.flush()
        return summary_stats, extreme_table, DELs, case_list, case_name_list, dlc_list

    def DLC_creation_IEC(self, inputs, discrete_inputs, fst_vt, powercurve=False):

        iec = CaseGen_IEC()

        # Turbine Data
        iec.Turbine_Class    = discrete_inputs['turbine_class']
        iec.Turbulence_Class = discrete_inputs['turbulence_class']
        iec.D                = fst_vt['ElastoDyn']['TipRad']*2. #np.min([fst_vt['InflowWind']['RefHt']*1.9 , fst_vt['ElastoDyn']['TipRad']*2.5])
        iec.z_hub            = fst_vt['InflowWind']['RefHt']

        # Turbine initial conditions
        iec.init_cond = {} # can leave as {} if data not available
        iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':inputs['U_init']}
        iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = inputs['Omega_init']
        iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':inputs['U_init']}
        iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = inputs['pitch_init']
        iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
        iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]

        # If running OLAF...
        if fst_vt['AeroDyn15']['WakeMod'] == 3:
            # Set DT according to OLAF guidelines
            dt_wanted, _, _, _, _ = OLAFParams(inputs['Omega_init'])
            iec.init_cond[("Fst","DT")]        = {'U':inputs['U_init']}
            iec.init_cond[("Fst","DT")]['val'] = dt_wanted
            # Raise the center of the grid 50% above hub height because the wake will expand
            iec.grid_center_over_hh = 1.5

        # Todo: need a way to handle Metocean conditions for Offshore
        # if offshore:
        #     iec.init_cond[("HydroDyn","WaveHs")]        = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        #     iec.init_cond[("HydroDyn","WaveHs")]['val'] = [1.101917033, 1.101917033, 1.179052649, 1.315715154, 1.536867124, 1.835816514, 2.187994638, 2.598127096, 3.061304068, 3.617035443, 4.027470219, 4.51580671, 4.51580671, 6.98, 10.7]
        #     iec.init_cond[("HydroDyn","WaveTp")]        = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        #     iec.init_cond[("HydroDyn","WaveTp")]['val'] = [8.515382435, 8.515382435, 8.310063688, 8.006300889, 7.6514231, 7.440581338, 7.460834063, 7.643300307, 8.046899942, 8.521314105, 8.987021024, 9.451641026, 9.451641026, 11.7, 14.2]

        # Setup dlc settings
        iec.dlc_inputs = {}
        iec.dlc_inputs['DLC']   = []
        iec.dlc_inputs['U']     = []
        iec.dlc_inputs['Seeds'] = []
        iec.dlc_inputs['Yaw']   = []
        iec.uniqueSeeds         = self.FASTpref['dlc_settings']['unique_wind_seeds']
        iec.uniqueWaveSeeds     = self.FASTpref['dlc_settings']['unique_wave_seeds']
        
        if powercurve:
            # running turbulent power curve
            iec.dlc_inputs['DLC'].append(1.1)
            iec.dlc_inputs['U'].append(self.FASTpref['dlc_settings']['Power_Curve']['U'])
            iec.dlc_inputs['Seeds'].append(self.FASTpref['dlc_settings']['Power_Curve']['Seeds'])
            iec.dlc_inputs['Yaw'].append([])

        else:

            for dlc in self.FASTpref['dlc_settings']['IEC']:

                if 'DLC' in dlc.keys():
                    iec.dlc_inputs['DLC'].append(dlc['DLC'])
                else:
                    iec.dlc_inputs['DLC'].append([])

                if 'U' in dlc.keys():
                    iec.dlc_inputs['U'].append(dlc['U'])
                else:
                    if dlc['DLC'] == 1.4:
                        iec.dlc_inputs['U'].append([float(inputs['Vrated'])-2., float(inputs['Vrated']), float(inputs['Vrated'])+2.])
                    elif dlc['DLC'] == 5.1:
                        iec.dlc_inputs['U'].append([float(inputs['Vrated'])-2., float(inputs['Vrated'])+2., float(inputs['V_cutout'])])
                    elif dlc['DLC'] == 6.1:
                        iec.dlc_inputs['U'].append([float(inputs['V_extreme50'])])
                    elif dlc['DLC'] == 6.3:
                        iec.dlc_inputs['U'].append([float(inputs['V_extreme1'])])
                    else:
                        iec.dlc_inputs['U'].append([])

                if 'Seeds' in dlc.keys():
                    iec.dlc_inputs['Seeds'].append(dlc['Seeds'])
                else:
                    iec.dlc_inputs['Seeds'].append([])

                if 'Yaw' in dlc.keys():
                    iec.dlc_inputs['Yaw'].append(dlc['Yaw'])
                else:
                    iec.dlc_inputs['Yaw'].append([])

        iec.transient_dir_change        = '-'
        iec.transient_shear_orientation = 'v'
        if self.options['modeling_options']['Level3']['flag']:
            iec.TMax    = self.options['modeling_options']['Level3']['simulation']['TMax']
        elif self.options['modeling_options']['Level2']['flag']:
            iec.TMax    = self.options['modeling_options']['Level2']['simulation']['TMax']
        else:
            raise Exception('Not running Level2 or Level3, no IEC cases generated')

        T0            = np.max([0. , iec.TMax - 600.])
        iec.TStart    = (iec.TMax-T0)/2. + T0
        self.simtime  = iec.TMax - T0
        self.TMax     = iec.TMax
        self.T0       = T0

        # path management
        iec.wind_dir        = self.FAST_runDirectory
        if self.FASTpref['file_management']['Turbsim_exe'] != 'none':
            iec.Turbsim_exe     = self.Turbsim_exe
        iec.debug_level     = self.debug_level
        iec.overwrite       = False # TODO: elevate these options to analysis input file
        iec.run_dir         = self.FAST_runDirectory

        if self.mpi_run and self.options['opt_options']['driver']['optimization']['flag']:
            iec.parallel_windfile_gen = True
            iec.mpi_run               = self.FASTpref['analysis_settings']['mpi_run']
            iec.comm_map_down         = self.FASTpref['analysis_settings']['mpi_comm_map_down']
        else:
            iec.parallel_windfile_gen = False

        if powercurve:
            iec.case_name_base  = self.FAST_namingOut + '_powercurve'
        else:
            iec.case_name_base  = self.FAST_namingOut + '_IEC'

        # Run case setup, generate wind inputs
        case_list, case_name_list, dlc_list = iec.execute()


        return case_list, case_name_list, dlc_list

    def DLC_creation_powercurve(self, inputs, discrete_inputs, fst_vt):

        if len(self.FASTpref['dlc_settings']['Power_Curve']['U']) > 0: # todo: need a warning if no powercurve wind speeds are specified and DLC 1.1 is not set
        
            if self.FASTpref['dlc_settings']['Power_Curve']['turbulent_power_curve']:

                case_list, case_name, dlc_list_IEC = self.DLC_creation_IEC(inputs, discrete_inputs, fst_vt, powercurve=True)

            else:
                U     = self.FASTpref['dlc_settings']['Power_Curve']['U']
                omega = np.interp(U, inputs['U_init'], inputs['Omega_init'])
                pitch = np.interp(U, inputs['U_init'], inputs['pitch_init'])

                # wind speeds
                case_inputs = {}
                case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
                case_inputs[("InflowWind","HWindSpeed")] = {'vals':U, 'group':1}
                case_inputs[("ElastoDyn","RotSpeed")]    = {'vals':omega, 'group':1}
                case_inputs[("ElastoDyn","BlPitch1")]    = {'vals':pitch, 'group':1}
                case_inputs[("ElastoDyn","BlPitch2")]    = case_inputs[("ElastoDyn","BlPitch1")]
                case_inputs[("ElastoDyn","BlPitch3")]    = case_inputs[("ElastoDyn","BlPitch1")]

                # Set DT according to OLAF guidelines
                if fst_vt['AeroDyn15']['WakeMod'] == 3:
                    dt_wanted, _, _, _, _ = OLAFParams(inputs['Omega_init'])
                    case_inputs[("Fst","DT")]               = {'vals':dt_wanted, 'group':1}

                case_list, case_name = CaseGen_General(case_inputs, self.FAST_runDirectory, self.FAST_namingOut + '_powercurve')

            dlc_list = [0.]*len(case_name)

            return case_list, case_name, dlc_list

        else:
            return [], [], []

    def post_process(self, summary_stats, extreme_table, DELs, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs):

        # Analysis
        outputs, discrete_outputs = self.get_blade_loading(summary_stats, extreme_table, inputs, discrete_inputs, outputs, discrete_outputs)
        outputs = self.get_tower_loading(summary_stats, extreme_table, inputs, outputs)

        # SubDyn is only supported in Level3: linearization in OpenFAST will be available in 3.0.0
        if self.options['modeling_options']['flags']['monopile'] and self.options['modeling_options']['Level3']['flag']:
            outputs = self.get_monopile_loading(summary_stats, extreme_table, inputs, outputs)
        outputs, discrete_outputs = self.calculate_AEP(summary_stats, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs)

        if self.FASTpref['dlc_settings']['run_IEC']:
            outputs, discrete_outputs = self.get_weighted_DELs(summary_stats, DELs, inputs, discrete_inputs, outputs, discrete_outputs)  

        outputs, discrete_outputs = self.get_control_measures(summary_stats, inputs, discrete_inputs, outputs, discrete_outputs)

        if self.options['modeling_options']['flags']['floating']:
            outputs, discrete_outputs = self.get_floating_measures(summary_stats, inputs, discrete_inputs, outputs, discrete_outputs)

    def get_blade_loading(self, sum_stats, extreme_table, inputs, discrete_inputs, outputs, discrete_outputs):
        """
        Find the spanwise loading along the blade span.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """
        
        # Determine blade with the maximum deflection magnitude
        if self.n_blades == 2:
            defl_mag = [max(sum_stats['TipDxc1']['max']), max(sum_stats['TipDxc2']['max'])]
        else:
            defl_mag = [max(sum_stats['TipDxc1']['max']), max(sum_stats['TipDxc2']['max']), max(sum_stats['TipDxc3']['max'])]

        # Select channels for the blade identified above
        if np.argmax(defl_mag) == 0:
            blade_chans_Fz = ["RootFzb1", "Spn1FLzb1", "Spn2FLzb1", "Spn3FLzb1", "Spn4FLzb1", "Spn5FLzb1", "Spn6FLzb1", "Spn7FLzb1", "Spn8FLzb1", "Spn9FLzb1"]
            blade_chans_Mx = ["RootMxb1", "Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1"]
            blade_chans_My = ["RootMyb1", "Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
            tip_max_chan   = "TipDxc1"

        if np.argmax(defl_mag) == 1:
            blade_chans_Fz = ["RootFzb2", "Spn1FLzb2", "Spn2FLzb2", "Spn3FLzb2", "Spn4FLzb2", "Spn5FLzb2", "Spn6FLzb2", "Spn7FLzb2", "Spn8FLzb2", "Spn9FLzb2"]
            blade_chans_Mx = ["RootMxb2", "Spn1MLxb2", "Spn2MLxb2", "Spn3MLxb2", "Spn4MLxb2", "Spn5MLxb2", "Spn6MLxb2", "Spn7MLxb2", "Spn8MLxb2", "Spn9MLxb2"]
            blade_chans_My = ["RootMyb2", "Spn1MLyb2", "Spn2MLyb2", "Spn3MLyb2", "Spn4MLyb2", "Spn5MLyb2", "Spn6MLyb2", "Spn7MLyb2", "Spn8MLyb2", "Spn9MLyb2"]
            tip_max_chan   = "TipDxc2"

        if np.argmax(defl_mag) == 2:            
            blade_chans_Fz = ["RootFzb3", "Spn1FLzb3", "Spn2FLzb3", "Spn3FLzb3", "Spn4FLzb3", "Spn5FLzb3", "Spn6FLzb3", "Spn7FLzb3", "Spn8FLzb3", "Spn9FLzb3"]
            blade_chans_Mx = ["RootMxb3", "Spn1MLxb3", "Spn2MLxb3", "Spn3MLxb3", "Spn4MLxb3", "Spn5MLxb3", "Spn6MLxb3", "Spn7MLxb3", "Spn8MLxb3", "Spn9MLxb3"]
            blade_chans_My = ["RootMyb3", "Spn1MLyb3", "Spn2MLyb3", "Spn3MLyb3", "Spn4MLyb3", "Spn5MLyb3", "Spn6MLyb3", "Spn7MLyb3", "Spn8MLyb3", "Spn9MLyb3"]
            tip_max_chan   = "TipDxc3"

        # Get the maximum out of plane blade deflection
        outputs["max_TipDxc"] = np.max(sum_stats[tip_max_chan]['max'])
        # Return moments around x and y and axial force along blade span at instance of largest out of plane blade tip deflection
        Fz = [extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])][var] for var in blade_chans_Fz]
        Mx = [extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])][var] for var in blade_chans_Mx]
        My = [extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])][var] for var in blade_chans_My]
        if np.any(np.isnan(Fz)):
            print('WARNING: nans found in Fz extremes')
            Fz[np.isnan(Fz)] = 0.0
        if np.any(np.isnan(Mx)):
            print('WARNING: nans found in Mx extremes')
            Mx[np.isnan(Mx)] = 0.0
        if np.any(np.isnan(My)):
            print('WARNING: nans found in My extremes')
            My[np.isnan(My)] = 0.0
        spline_Fz = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((Fz, 0.)))
        spline_Mx = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((Mx, 0.)))
        spline_My = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((My, 0.)))

        r = inputs['r']
        Fz_out = spline_Fz(r).flatten()
        Mx_out = spline_Mx(r).flatten()
        My_out = spline_My(r).flatten()

        outputs['blade_maxTD_Mx'] = Mx_out
        outputs['blade_maxTD_My'] = My_out
        outputs['blade_maxTD_Fz'] = Fz_out

        # Determine maximum root moment
        if self.n_blades == 2:
            blade_root_flap_moment = max([max(sum_stats['RootMyb1']['max']), max(sum_stats['RootMyb2']['max'])])
            blade_root_oop_moment  = max([max(sum_stats['RootMyc1']['max']), max(sum_stats['RootMyc2']['max'])])
            blade_root_tors_moment  = max([max(sum_stats['RootMzb1']['max']), max(sum_stats['RootMzb2']['max'])])
        else:
            blade_root_flap_moment = max([max(sum_stats['RootMyb1']['max']), max(sum_stats['RootMyb2']['max']), max(sum_stats['RootMyb3']['max'])])
            blade_root_oop_moment  = max([max(sum_stats['RootMyc1']['max']), max(sum_stats['RootMyc2']['max']), max(sum_stats['RootMyc3']['max'])])
            blade_root_tors_moment  = max([max(sum_stats['RootMzb1']['max']), max(sum_stats['RootMzb2']['max']), max(sum_stats['RootMzb3']['max'])])
        outputs['max_RootMyb'] = blade_root_flap_moment
        outputs['max_RootMyc'] = blade_root_oop_moment
        outputs['max_RootMzb'] = blade_root_tors_moment

        ## Get hub moments and forces in the non-rotating frame
        outputs['hub_Fxyz'] = np.array([extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['RotThrust'],
                                    extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFys'],
                                    extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFzs']])*1.e3
        outputs['hub_Mxyz'] = np.array([extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['RotTorq'],
                                    extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMys'],
                                    extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMzs']])*1.e3

        ## Post process aerodynamic data
        # Angles of attack - max, std, mean
        blade1_chans_aoa = ["B1N1Alpha", "B1N2Alpha", "B1N3Alpha", "B1N4Alpha", "B1N5Alpha", "B1N6Alpha", "B1N7Alpha", "B1N8Alpha", "B1N9Alpha"]
        blade2_chans_aoa = ["B2N1Alpha", "B2N2Alpha", "B2N3Alpha", "B2N4Alpha", "B2N5Alpha", "B2N6Alpha", "B2N7Alpha", "B2N8Alpha", "B2N9Alpha"]
        aoa_max_B1  = [np.max(sum_stats[var]['max'])    for var in blade1_chans_aoa]
        aoa_mean_B1 = [np.mean(sum_stats[var]['mean'])  for var in blade1_chans_aoa]
        aoa_std_B1  = [np.mean(sum_stats[var]['std'])   for var in blade1_chans_aoa]
        aoa_max_B2  = [np.max(sum_stats[var]['max'])    for var in blade2_chans_aoa]
        aoa_mean_B2 = [np.mean(sum_stats[var]['mean'])  for var in blade2_chans_aoa]
        aoa_std_B2  = [np.mean(sum_stats[var]['std'])   for var in blade2_chans_aoa]                
        if self.n_blades == 2:
            spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2], axis=0))
            spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_std_B1, aoa_std_B2], axis=0))
            spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2], axis=0))
        elif self.n_blades == 3:
            blade3_chans_aoa    = ["B3N1Alpha", "B3N2Alpha", "B3N3Alpha", "B3N4Alpha", "B3N5Alpha", "B3N6Alpha", "B3N7Alpha", "B3N8Alpha", "B3N9Alpha"]
            aoa_max_B3          = [np.max(sum_stats[var]['max'])    for var in blade3_chans_aoa]
            aoa_mean_B3         = [np.mean(sum_stats[var]['mean'])  for var in blade3_chans_aoa]
            aoa_std_B3          = [np.mean(sum_stats[var]['std'])   for var in blade3_chans_aoa]
            spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2, aoa_max_B3], axis=0))
            spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_max_B1, aoa_std_B2, aoa_std_B3], axis=0))
            spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2, aoa_mean_B3], axis=0))
        else:
            raise Exception('The calculations only support 2 or 3 bladed rotors')

        outputs['max_aoa']  = spline_aoa_max(r)
        outputs['std_aoa']  = spline_aoa_std(r)
        outputs['mean_aoa'] = spline_aoa_mean(r)

        return outputs, discrete_outputs

    def get_tower_loading(self, sum_stats, extreme_table, inputs, outputs):
        """
        Find the loading along the tower height.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """

        n_height_tow = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_tower']
        n_full_tow   = get_nfull(n_height_tow)

        tower_chans_Fx = ["TwrBsFxt", "TwHt1FLxt", "TwHt2FLxt", "TwHt3FLxt", "TwHt4FLxt", "TwHt5FLxt", "TwHt6FLxt", "TwHt7FLxt", "TwHt8FLxt", "TwHt9FLxt", "YawBrFxp"]
        tower_chans_Fy = ["TwrBsFyt", "TwHt1FLyt", "TwHt2FLyt", "TwHt3FLyt", "TwHt4FLyt", "TwHt5FLyt", "TwHt6FLyt", "TwHt7FLyt", "TwHt8FLyt", "TwHt9FLyt", "YawBrFyp"]
        tower_chans_Fz = ["TwrBsFzt", "TwHt1FLzt", "TwHt2FLzt", "TwHt3FLzt", "TwHt4FLzt", "TwHt5FLzt", "TwHt6FLzt", "TwHt7FLzt", "TwHt8FLzt", "TwHt9FLzt", "YawBrFzp"]
        tower_chans_Mx = ["TwrBsMxt", "TwHt1MLxt", "TwHt2MLxt", "TwHt3MLxt", "TwHt4MLxt", "TwHt5MLxt", "TwHt6MLxt", "TwHt7MLxt", "TwHt8MLxt", "TwHt9MLxt", "YawBrMxp"]
        tower_chans_My = ["TwrBsMyt", "TwHt1MLyt", "TwHt2MLyt", "TwHt3MLyt", "TwHt4MLyt", "TwHt5MLyt", "TwHt6MLyt", "TwHt7MLyt", "TwHt8MLyt", "TwHt9MLyt", "YawBrMyp"]
        tower_chans_Mz = ["TwrBsMzt", "TwHt1MLzt", "TwHt2MLzt", "TwHt3MLzt", "TwHt4MLzt", "TwHt5MLzt", "TwHt6MLzt", "TwHt7MLzt", "TwHt8MLzt", "TwHt9MLzt", "YawBrMzp"]


        fatb_max_chan   = "TwrBsMyt"

        # Get the maximum fore-aft moment at tower base
        outputs["max_TwrBsMyt"] = np.max(sum_stats[fatb_max_chan]['max'])
        # Return forces and moments along tower height at instance of largest fore-aft tower base moment
        Fx = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fx]
        Fy = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fy]
        Fz = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Fz]
        Mx = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Mx]
        My = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_My]
        Mz = [extreme_table[fatb_max_chan][np.argmax(sum_stats[fatb_max_chan]['max'])][var] for var in tower_chans_Mz]

        # Spline results on tower basic grid
        spline_Fx      = PchipInterpolator(self.Z_out_ED_twr, Fx)
        spline_Fy      = PchipInterpolator(self.Z_out_ED_twr, Fy)
        spline_Fz      = PchipInterpolator(self.Z_out_ED_twr, Fz)
        spline_Mx      = PchipInterpolator(self.Z_out_ED_twr, Mx)
        spline_My      = PchipInterpolator(self.Z_out_ED_twr, My)
        spline_Mz      = PchipInterpolator(self.Z_out_ED_twr, Mz)

        z_full = inputs['tower_monopile_z_full'][-n_full_tow:]
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        outputs['tower_maxMy_Fx'] = spline_Fx(z)
        outputs['tower_maxMy_Fy'] = spline_Fy(z)
        outputs['tower_maxMy_Fz'] = spline_Fz(z)
        outputs['tower_maxMy_Mx'] = spline_Mx(z)
        outputs['tower_maxMy_My'] = spline_My(z)
        outputs['tower_maxMy_Mz'] = spline_Mz(z)
        
        if not self.options['modeling_options']['flags']['monopile']:
            for k in ['Fx','Fy','Fz','Mx','My','Mz']:
                outputs[f'tower_monopile_maxMy_{k}'] = outputs[f'tower_maxMy_{k}']

        return outputs

    def get_monopile_loading(self, sum_stats, extreme_table, inputs, outputs):
        """
        Find the loading along the monopile length.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """

        n_height_mon = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_monopile']
        n_full_mon   = get_nfull(n_height_mon)

        monopile_chans_Fx = []
        monopile_chans_Fy = []
        monopile_chans_Fz = []
        monopile_chans_Mx = []
        monopile_chans_My = []
        monopile_chans_Mz = []
        k=1
        for i in range(len(self.Z_out_SD_mpl)):
            if k==9:
                Node=2
            else:
                Node=1
            monopile_chans_Fx += ["M" + str(k) + "N" + str(Node) + "FKxe"]
            monopile_chans_Fy += ["M" + str(k) + "N" + str(Node) + "FKye"]
            monopile_chans_Fz += ["M" + str(k) + "N" + str(Node) + "FKze"]
            monopile_chans_Mx += ["M" + str(k) + "N" + str(Node) + "MKxe"]
            monopile_chans_My += ["M" + str(k) + "N" + str(Node) + "MKye"]
            monopile_chans_Mz += ["M" + str(k) + "N" + str(Node) + "MKze"]
            k+=1

        max_chan   = "M1N1MKye"

        # # Get the maximum of signal M1N1MKye
        outputs["max_M1N1MKye"] = np.max(sum_stats[max_chan]['max'])
        # # Return forces and moments along tower height at instance of largest fore-aft tower base moment
        Fx = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fx]
        Fy = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fy]
        Fz = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Fz]
        Mx = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Mx]
        My = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_My]
        Mz = [extreme_table[max_chan][np.argmax(sum_stats[max_chan]['max'])][var] for var in monopile_chans_Mz]

        # # Spline results on grid of channel locations along the monopile
        spline_Fx      = PchipInterpolator(self.Z_out_SD_mpl, Fx)
        spline_Fy      = PchipInterpolator(self.Z_out_SD_mpl, Fy)
        spline_Fz      = PchipInterpolator(self.Z_out_SD_mpl, Fz)
        spline_Mx      = PchipInterpolator(self.Z_out_SD_mpl, Mx)
        spline_My      = PchipInterpolator(self.Z_out_SD_mpl, My)
        spline_Mz      = PchipInterpolator(self.Z_out_SD_mpl, Mz)

        z_full = inputs['tower_monopile_z_full'][:n_full_mon]
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        outputs['monopile_maxMy_Fx'] = spline_Fx(z)
        outputs['monopile_maxMy_Fy'] = spline_Fy(z)
        outputs['monopile_maxMy_Fz'] = spline_Fz(z)
        outputs['monopile_maxMy_Mx'] = spline_Mx(z)
        outputs['monopile_maxMy_My'] = spline_My(z)
        outputs['monopile_maxMy_Mz'] = spline_Mz(z)
        
        for k in ['Fx','Fy','Fz','Mx','My','Mz']:
            outputs[f'tower_monopile_maxMy_{k}'] = np.r_[outputs[f'monopile_maxMy_{k}'], outputs[f'tower_maxMy_{k}']]

        return outputs

    def calculate_AEP(self, sum_stats, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs):
        """
        Calculates annual energy production of the relevant DLCs in `case_list`.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        case_list : list
        dlc_list : list
        """
        ## Get AEP and power curve
        if self.FASTpref['dlc_settings']['run_power_curve']:

            # determine which dlc will be used for the powercurve calculations, allows using dlc 1.1 if specific power curve calculations were not run
            idx_pwrcrv    = [i for i, dlc in enumerate(dlc_list) if dlc==0.]
            idx_pwrcrv_11 = [i for i, dlc in enumerate(dlc_list) if dlc==1.1]
            if len(idx_pwrcrv) == 0 and len(idx_pwrcrv_11) > 0:
                idx_pwrcrv = idx_pwrcrv_11

            # sort out power curve stats
            # stats_pwrcrv = {}
            # for var in sum_stats.keys():
            #     if var != 'meta':
            #         stats_pwrcrv[var] = {}
            #         for stat in sum_stats[var].keys():
            #             stats_pwrcrv[var][stat] = [x for i, x in enumerate(sum_stats[var][stat]) if i in idx_pwrcrv]

            # stats_pwrcrv['meta'] = sum_stats['meta']

            stats_pwrcrv = sum_stats.iloc[idx_pwrcrv].copy()

            # Get windspeeds from case list 
            if self.FASTpref['dlc_settings']['Power_Curve']['turbulent_power_curve']:
                U = []
                for fname in [case[('InflowWind', 'Filename_Uni')] for i, case in enumerate(case_list) if i in idx_pwrcrv]:
                    fname = os.path.split(fname)[-1]
                    ntm      = fname.split('NTM')[-1].split('_')
                    ntm_U    = float(".".join(ntm[1].strip("U").split('.')[:-1]))
                    ntm_Seed = float(".".join(ntm[2].strip("Seed").split('.')[:-1]))
                    U.append(ntm_U)
            else:
                U = [float(case[('InflowWind', 'HWindSpeed')]) for i, case in enumerate(case_list) if i in idx_pwrcrv]

            # Calculate AEP and Performance Data
            if len(U) > 1 and self.fst_vt['Fst']['CompServo'] == 1:
                pp = PowerProduction(discrete_inputs['turbine_class'])
                pwr_curve_vars   = ["GenPwr", "RtAeroCp", "RotSpeed", "BldPitch1"]
                AEP, perf_data = pp.AEP(stats_pwrcrv, U, pwr_curve_vars)

                outputs['P_out']       = perf_data['GenPwr']['mean'] * 1.e3
                outputs['Cp_out']      = perf_data['RtAeroCp']['mean']
                outputs['Omega_out']   = perf_data['RotSpeed']['mean']
                outputs['pitch_out']   = perf_data['BldPitch1']['mean']
                outputs['AEP']         = AEP
            else:
                outputs['Cp_out']      = stats_pwrcrv['RtAeroCp']['mean']
                outputs['AEP']         = 0.0
                outputs['Omega_out']   = stats_pwrcrv['RotSpeed']['mean']
                outputs['pitch_out']   = stats_pwrcrv['BldPitch1']['mean']
                if self.fst_vt['Fst']['CompServo'] == 1:
                    outputs['P_out']       = stats_pwrcrv['GenPwr']['mean'][0] * 1.e3
                print('WARNING: OpenFAST is run at a single wind speed. AEP cannot be estimated.')

            outputs['V_out']       = np.unique(U)

        return outputs, discrete_outputs

    def get_weighted_DELs(self, sum_stats, DELs, inputs, discrete_inputs, outputs, discrete_outputs):
        """

        """

        # Get DELS from OpenFAST data
        # if self.fst_vt['Fst']['TMax'] - self.options['modeling_options']['openfast']['fst_settings'][('Fst','TStart')] < 60.:
        #     print('WARNING: the measurement window of the OpenFAST simulations is shorter than 60 seconds. No DEL can be estimated reliably.')

        if self.FASTpref['dlc_settings']['run_IEC']:
            U = []
            for dlc in self.FASTpref['dlc_settings']['IEC']:
                U_set       = dlc['U']
                num_seeds   = len(dlc['Seeds'])
                U_all       = [U_set]*num_seeds
                U_dlc = [u for uset in U_all for u in uset]
            U.extend(U_dlc)
            U.sort()          
            
        
        pp = PowerProduction(discrete_inputs['turbine_class'])
        ws_prob = pp.prob_WindDist(U, disttype='pdf')

        if self.options['opt_options']['merit_figure'] == 'DEL_RootMyb':
            if self.n_blades == 2:
                outputs['DEL_RootMyb'] = np.max([np.sum(ws_prob*DELs['RootMyb1']), 
                                                np.sum(ws_prob*DELs['RootMyb2'])])
            else:
                outputs['DEL_RootMyb'] = np.max([np.sum(ws_prob*DELs['RootMyb1']), 
                                                np.sum(ws_prob*DELs['RootMyb2']),
                                                np.sum(ws_prob*DELs['RootMyb3'])])

        if self.options['opt_options']['merit_figure'] == 'DEL_TwrBsMyt':
            outputs['DEL_TwrBsMyt'] = np.sum(ws_prob*DELs['TwrBsMyt'])

        return outputs, discrete_outputs

    def get_control_measures(self,sum_stats,inputs, discrete_inputs, outputs, discrete_outputs):
        ''' 
        calculate control measures:
            - rotor_overspeed

        given:
            - sum_stats : pd.DataFrame
        '''

        if self.options['opt_options']['constraints']['control']['rotor_overspeed']['flag']:
            outputs['rotor_overspeed'] = ( np.max(sum_stats['GenSpeed']['max']) * np.pi/30. / self.fst_vt['DISCON_in']['PC_RefSpd'] ) - 1.0

        return outputs, discrete_outputs

    def get_floating_measures(self,sum_stats,inputs, discrete_inputs, outputs, discrete_outputs):
        '''
        calculate floating measures:
            - Std_PtfmPitch (max over all dlcs if constraint, mean otheriwse)
            - Max_PtfmPitch

        given:
            - sum_stats : pd.DataFrame
        '''
        
        if self.options['opt_options']['merit_figure'] == 'Std_PtfmPitch':
            # Let's just average the standard deviation of PtfmPitch for now
            # TODO: weight based on WS distribution, or something else
            print("sum_stats['PtfmPitch']['std']:")   # for debugging
            print(sum_stats['PtfmPitch']['std'])   # for debugging
            outputs['Std_PtfmPitch'] = np.mean(sum_stats['PtfmPitch']['std'])

        if self.options['opt_options']['constraints']['control']['Max_PtfmPitch']['flag']:
            print("sum_stats['PtfmPitch']['max']:")  # for debugging
            print(sum_stats['PtfmPitch']['max'])  # for debugging
            outputs['Max_PtfmPitch']  = np.max(sum_stats['PtfmPitch']['max'])

        if self.options['opt_options']['constraints']['control']['Std_PtfmPitch']['flag']:
            print("sum_stats['PtfmPitch']['std']:")   # for debugging
            print(sum_stats['PtfmPitch']['std'])   # for debugging
            outputs['Std_PtfmPitch'] = np.max(sum_stats['PtfmPitch']['std'])

        return outputs, discrete_outputs

    
    def write_FAST(self, fst_vt, discrete_outputs):
        writer                   = InputWriter_OpenFAST()
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.execute()

        if self.debug_level > 0:
            print('RAN UPDATE: ', self.FAST_runDirectory, self.FAST_namingOut)


    def writeCpsurfaces(self, inputs):
        
        FASTpref  = self.options['modeling_options']['openfast']['FASTpref']
        file_name = os.path.join(FASTpref['file_management']['FAST_runDirectory'], FASTpref['file_management']['FAST_namingOut'] + '_Cp_Ct_Cq.dat')
        
        # Write Cp-Ct-Cq-TSR tables file
        n_pitch = len(inputs['pitch_vector'])
        n_tsr   = len(inputs['tsr_vector'])
        n_U     = len(inputs['U_vector'])
        
        file = open(file_name,'w')
        file.write('# ------- Rotor performance tables ------- \n')
        file.write('# ------------ Written using AeroElasticSE with data from CCBlade ------------\n')
        file.write('\n')
        file.write('# Pitch angle vector - x axis (matrix columns) (deg)\n')
        for i in range(n_pitch):
            file.write('%.2f   ' % inputs['pitch_vector'][i])
        file.write('\n# TSR vector - y axis (matrix rows) (-)\n')
        for i in range(n_tsr):
            file.write('%.2f   ' % inputs['tsr_vector'][i])
        file.write('\n# Wind speed vector - z axis (m/s)\n')
        for i in range(n_U):
            file.write('%.2f   ' % inputs['U_vector'][i])
        file.write('\n')
        
        file.write('\n# Power coefficient\n\n')
        
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cp_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
        
        file.write('\n#  Thrust coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Ct_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
        
        file.write('\n# Torque coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cq_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
            
        file.close()


        return file_name


    # def BladeFatigue(self, FAST_Output, case_list, dlc_list, inputs, outputs, discrete_inputs, discrete_outputs):

    #     # Perform rainflow counting
    #     if self.options['modeling_options']['General']['verbosity']:
    #         print('Running Rainflow Counting')
    #         sys.stdout.flush()

    #     rainflow = {}
    #     var_rainflow = ["RootMxb1", "Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1", "RootMyb1", "Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
    #     for i, (datai, casei, dlci) in enumerate(zip(FAST_Output, case_list, dlc_list)):
    #         if dlci in [1.1, 1.2]:
            
    #             # Get wind speed and seed of output file
    #             ntm  = casei[('InflowWind', 'FileName_BTS')].split('NTM')[-1].split('_')
    #             U    = float(".".join(ntm[1].strip("U").split('.')[:-1]))
    #             Seed = float(".".join(ntm[2].strip("Seed").split('.')[:-1]))

    #             if U not in list(rainflow.keys()):
    #                 rainflow[U]       = {}
    #             if Seed not in list(rainflow[U].keys()):
    #                 rainflow[U][Seed] = {}
                
    #             # Rainflow counting by var
    #             if len(var_rainflow) == 0:
    #                 var_rainflow = list(datai.keys())


    #             # index for start/end of time series
    #             idx_s = np.argmax(datai["Time"] >= self.T0)
    #             idx_e = np.argmax(datai["Time"] >= self.TMax) + 1

    #             for var in var_rainflow:
    #                 ranges, means = fatpack.find_rainflow_ranges(datai[var][idx_s:idx_e], return_means=True)

    #                 rainflow[U][Seed][var] = {}
    #                 rainflow[U][Seed][var]['rf_amp']  = ranges.tolist()
    #                 rainflow[U][Seed][var]['rf_mean'] = means.tolist()
    #                 rainflow[U][Seed][var]['mean']    = float(np.mean(datai[var]))

    #     # save_yaml(self.FAST_resultsDirectory, 'rainflow.yaml', rainflow)
    #     # rainflow = load_yaml(self.FatigueFile, package=1)

    #     # Setup fatigue calculations
    #     U       = list(rainflow.keys())
    #     Seeds   = list(rainflow[U[0]].keys())
    #     chans   = list(rainflow[U[0]][Seeds[0]].keys())
    #     r_gage  = np.r_[0., self.R_out_ED_bl]
    #     r_gage /= r_gage[-1]
    #     simtime = self.simtime
    #     n_seeds = float(len(Seeds))
    #     n_gage  = len(r_gage)

    #     r       = (inputs['r']-inputs['r'][0])/(inputs['r'][-1]-inputs['r'][0])
    #     m_default = 8. # assume default m=10  (8 or 12 also reasonable)
    #     m       = [mi if mi > 0. else m_default for mi in inputs['m']]  # Assumption: if no S-N slope is given for a material, use default value TODO: input['m'] is not connected, only using the default currently

    #     eps_uts = inputs['Xt'][:,0]/inputs['E'][:,0]
    #     eps_ucs = inputs['Xc'][:,0]/inputs['E'][:,0]
    #     gamma_m = 1.#inputs['gamma_m']
    #     gamma_f = 1.#inputs['gamma_f']
    #     yrs     = 20.  # TODO
    #     t_life  = 60.*60.*24*365.24*yrs
    #     U_bar   = inputs['V_mean_iec']

    #     # pdf of wind speeds
    #     binwidth = np.diff(U)
    #     U_bins   = np.r_[[U[0] - binwidth[0]/2.], [np.mean([U[i-1], U[i]]) for i in range(1,len(U))], [U[-1] + binwidth[-1]/2.]]
    #     pdf = np.diff(RayleighCDF(U_bins, xbar=U_bar))
    #     if sum(pdf) < 0.9:
    #         print('Warning: Cummulative probability of wind speeds in rotor_loads_defl_strains.BladeFatigue is low, sum of weights: %f' % sum(pdf))
    #         print('Mean winds speed: %f' % U_bar)
    #         print('Simulated wind speeds: ', U)
    #         sys.stdout.flush()

    #     # Materials of analysis layers
    #     te_ss_var_ok       = False
    #     te_ps_var_ok       = False
    #     spar_cap_ss_var_ok = False
    #     spar_cap_ps_var_ok = False
    #     for i_layer in range(self.n_layers):
    #         if self.te_ss_var in self.layer_name:
    #             te_ss_var_ok        = True
    #         if self.te_ps_var in self.layer_name:
    #             te_ps_var_ok        = True
    #         if self.spar_cap_ss_var in self.layer_name:
    #             spar_cap_ss_var_ok  = True
    #         if self.spar_cap_ps_var in self.layer_name:
    #             spar_cap_ps_var_ok  = True

    #     # if te_ss_var_ok == False:
    #     #     print('The layer at the trailing edge suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ss_var)
    #     # if te_ps_var_ok == False:
    #     #     print('The layer at the trailing edge pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ps_var)
    #     if spar_cap_ss_var_ok == False:
    #         print('The layer at the spar cap suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ss_var)
    #     if spar_cap_ps_var_ok == False:
    #         print('The layer at the spar cap pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ps_var)
    #     sys.stdout.flush()

    #     # Get blade properties at gage locations
    #     y_tc       = remap2grid(r, inputs['y_tc'], r_gage)
    #     x_tc       = remap2grid(r, inputs['x_tc'], r_gage)
    #     chord      = remap2grid(r, inputs['chord'], r_gage)
    #     rthick     = remap2grid(r, inputs['rthick'], r_gage)
    #     pitch_axis = remap2grid(r, inputs['pitch_axis'], r_gage)
    #     EIyy       = remap2grid(r, inputs['beam:EIyy'], r_gage)
    #     EIxx       = remap2grid(r, inputs['beam:EIxx'], r_gage)

    #     te_ss_mats = np.floor(remap2grid(r, inputs['te_ss_mats'], r_gage, axis=0)) # materials is section
    #     te_ps_mats = np.floor(remap2grid(r, inputs['te_ps_mats'], r_gage, axis=0))
    #     sc_ss_mats = np.floor(remap2grid(r, inputs['sc_ss_mats'], r_gage, axis=0))
    #     sc_ps_mats = np.floor(remap2grid(r, inputs['sc_ps_mats'], r_gage, axis=0))

    #     c_TE       = chord*(1.-pitch_axis) + y_tc
    #     c_SC       = chord*rthick/2. + x_tc #this is overly simplistic, using maximum thickness point, should use the actual profiles
    #     sys.stdout.flush()

    #     C_miners_SC_SS_gage = np.zeros((n_gage, self.n_mat, 2))
    #     C_miners_SC_PS_gage = np.zeros((n_gage, self.n_mat, 2))
    #     C_miners_TE_SS_gage = np.zeros((n_gage, self.n_mat, 2))
    #     C_miners_TE_PS_gage = np.zeros((n_gage, self.n_mat, 2))

    #     # Map channels to output matrix
    #     chan_map   = {}
    #     for i_var, var in enumerate(chans):
    #         # Determine spanwise position
    #         if 'Root' in var:
    #             i_span = 0
    #         elif 'Spn' in var and 'M' in var:
    #             i_span = int(var.strip('Spn').split('M')[0])
    #         else:
    #             # not a spanwise output channel, skip
    #             print('Fatigue Model: Skipping channel: %s, not a spanwise moment' % var)
    #             sys.stdout.flush()
    #             chans.remove(var)
    #             continue
    #         # Determine if edgewise of flapwise moment
    #         if 'M' in var and 'x' in var:
    #             # Flapwise
    #             axis = 1
    #         elif 'M' in var and 'y' in var:
    #             # Edgewise
    #             axis = 0
    #         else:
    #             # not an edgewise / flapwise moment, skip
    #             print('Fatigue Model: Skipping channel: "%s", not an edgewise/flapwise moment' % var)
    #             sys.stdout.flush()
    #             continue

    #         chan_map[var] = {}
    #         chan_map[var]['i_gage'] = i_span
    #         chan_map[var]['axis']   = axis

    #     # Map composite sections
    #     composite_map = [['TE', 'SS', te_ss_var_ok],
    #                      ['TE', 'PS', te_ps_var_ok],
    #                      ['SC', 'SS', spar_cap_ss_var_ok],
    #                      ['SC', 'PS', spar_cap_ps_var_ok]]

    #     if self.options['modeling_options']['General']['verbosity']:
    #         print("Running Miner's Rule calculations")
    #         sys.stdout.flush()

    #     ########
    #     # Loop through composite sections, materials, output channels, and simulations (wind speeds * seeds)
    #     for comp_i in composite_map:

    #         #skip this composite section?
    #         if not comp_i[2]:
    #             continue

    #         #
    #         C_miners = np.zeros((n_gage, self.n_mat, 2))
    #         if comp_i[0]       == 'TE':
    #             c = c_TE
    #             if comp_i[1]   == 'SS':
    #                 mats = te_ss_mats
    #             elif comp_i[1] == 'PS':
    #                 mats = te_ps_mats
    #         elif comp_i[0]     == 'SC':
    #             c = c_SC
    #             if comp_i[1]   == 'SS':
    #                 mats = sc_ss_mats
    #             elif comp_i[1] == 'PS':
    #                 mats = sc_ps_mats

    #         for i_mat in range(self.n_mat):

    #             for i_var, var in enumerate(chans):
    #                 i_gage = chan_map[var]['i_gage']
    #                 axis   = chan_map[var]['axis']

    #                 # skip if material at this spanwise location is not included in the composite section
    #                 if mats[i_gage, i_mat] == 0.:
    #                     continue

    #                 # Determine if edgewise of flapwise moment
    #                 pitch_axis_i = pitch_axis[i_gage]
    #                 chord_i      = chord[i_gage]
    #                 c_i          = c[i_gage]
    #                 if axis == 0:
    #                     EI_i     = EIxx[i_gage]
    #                 else:
    #                     EI_i     = EIyy[i_gage]

    #                 for i_u, u in enumerate(U):
    #                     for i_s, seed in enumerate(Seeds):
    #                         M_mean = np.array(rainflow[u][seed][var]['rf_mean']) * 1.e3
    #                         M_amp  = np.array(rainflow[u][seed][var]['rf_amp']) * 1.e3

    #                         for M_mean_i, M_amp_i in zip(M_mean, M_amp):
    #                             n_cycles = 1.
    #                             eps_mean = M_mean_i*c_i/EI_i 
    #                             eps_amp  = M_amp_i*c_i/EI_i

    #                             if eps_amp != 0.:
    #                                 Nf = ((eps_uts[i_mat] + np.abs(eps_ucs[i_mat]) - np.abs(2.*eps_mean*gamma_m*gamma_f - eps_uts[i_mat] + np.abs(eps_ucs[i_mat]))) / (2.*eps_amp*gamma_m*gamma_f))**m[i_mat]
    #                                 n  = n_cycles * t_life * pdf[i_u] / (simtime * n_seeds)
    #                                 C_miners[i_gage, i_mat, axis]  += n/Nf

    #         # Assign outputs
    #         if comp_i[0] == 'SC' and comp_i[1] == 'SS':
    #             outputs['C_miners_SC_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
    #         elif comp_i[0] == 'SC' and comp_i[1] == 'PS':
    #             outputs['C_miners_SC_PS'] = remap2grid(r_gage, C_miners, r, axis=0)
    #         # elif comp_i[0] == 'TE' and comp_i[1] == 'SS':
    #         #     outputs['C_miners_TE_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
    #         # elif comp_i[0] == 'TE' and comp_i[1] == 'PS':
    #         #     outputs['C_miners_TE_PS'] = remap2grid(r_gage, C_miners, r, axis=0)

    #     return outputs, discrete_outputs

def RayleighCDF(x, xbar=10.):
    return 1.0 - np.exp(-np.pi/4.0*(x/xbar)**2)

def OLAFParams(omega_rpm, deltaPsiDeg=6, nNWrot=2, nFWrot=10, nFWrotFree=3, nPerRot=None, totalRot=None, show=False):
    """
    Computes recommended time step and wake length based on the rotational speed in RPM
 
    INPUTS:
     - omega_rpm: rotational speed in RPM
     - deltaPsiDeg : azimuthal discretization in deg
     - nNWrot : number of near wake rotations
     - nFWrot : total number of far wake rotations
     - nFWrotFree : number of far wake rotations that are free
 
        deltaPsiDeg  -  nPerRot
             5            72   
             6            60   
             7            51.5 
             8            45   
    """
    omega_rpm = np.asarray(omega_rpm)
    omega = omega_rpm*2*np.pi/60
    T = 2*np.pi/omega
    if nPerRot is not None:
        dt_wanted    = np.around(T/nPerRot,4)
    else:
        dt_wanted    = np.around(deltaPsiDeg/(6*omega_rpm),4)
        nPerRot = int(2*np.pi /(deltaPsiDeg*np.pi/180))
 
    nNWPanel     = nNWrot*nPerRot
    nFWPanel     = nFWrot*nPerRot
    nFWPanelFree = nFWrotFree*nPerRot
 
    if totalRot is None:
        totalRot = (nNWrot + nFWrot)*3 # going three-times through the entire wake
 
    tMax = dt_wanted*nPerRot*totalRot
 
    if show:
        print(dt_wanted              , '  dt')
        print(int      (nNWPanel    ), '  nNWPanel          ({} rotations)'.format(nNWrot))
        print(int      (nFWPanel    ), '  FarWakeLength     ({} rotations)'.format(nFWrot))
        print(int      (nFWPanelFree), '  FreeFarWakeLength ({} rotations)'.format(nFWrotFree))
        print(tMax              , '  Tmax ({} rotations)'.format(totalRot))
 
    return dt_wanted, tMax, nNWPanel, nFWPanel, nFWPanelFree

class ModesElastoDyn(ExplicitComponent):
    """
    Component that adds a multiplicative factor to axial, torsional, and flap-edge coupling stiffness to mimic ElastoDyn
    
    Parameters
    ----------
    EA : numpy array[n_span], [N]
        1D array of the actual axial stiffness
    EIxy : numpy array[n_span], [Nm2]
        1D array of the actual flap-edge coupling stiffness
    GJ : numpy array[n_span], [Nm2]
        1D array of the actual torsional stiffness
    G  : numpy array[n_mat], [N/m2]
        1D array of the actual shear stiffness of the materials
    
    Returns
    -------
    EA_stiff : numpy array[n_span], [N]
        1D array of the stiff axial stiffness
    EIxy_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff flap-edge coupling stiffness
    GJ_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff torsional stiffness
    G_stiff  : numpy array[n_mat], [N/m2]
        1D array of the stiff shear stiffness of the materials
    
    """    
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        n_span          = self.options['modeling_options']['WISDEM']['RotorSE']['n_span']
        n_mat           = self.options['modeling_options']['materials']['n_mat']

        self.add_input('EA',    val=np.zeros(n_span), units='N',        desc='axial stiffness')
        self.add_input('EIxy',  val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',    val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')

        self.add_input('G',     val=np.zeros([n_mat, 3]), units='Pa',   desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')


        self.add_output('EA_stiff',  val=np.zeros(n_span), units='N',        desc='artifically stiff axial stiffness')
        self.add_output('EIxy_zero', val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff coupled flap-edge stiffness')
        self.add_output('GJ_stiff',  val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('G_stiff',   val=np.zeros([n_mat, 3]), units='Pa',   desc='artificially stif 2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')

    def compute(self, inputs, outputs):

        k = 10.

        outputs['EA_stiff']   = inputs['EA']   * k
        outputs['EIxy_zero']  = inputs['EIxy'] * 0.
        outputs['GJ_stiff']   = inputs['GJ']   * k
        outputs['G_stiff']    = inputs['G']    * k
