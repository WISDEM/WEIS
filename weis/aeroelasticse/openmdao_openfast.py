import numpy as np
import os, shutil, sys, platform
import copy
from scipy.interpolate                      import PchipInterpolator
from openmdao.api                           import ExplicitComponent
from wisdem.commonse.mpi_tools              import MPI
from wisdem.towerse      import NFREQ, get_nfull
import wisdem.commonse.utilities              as util
from wisdem.rotorse.rotor_power             import eval_unsteady
from weis.aeroelasticse.FAST_writer         import InputWriter_OpenFAST
from weis.aeroelasticse.FAST_reader         import InputReader_OpenFAST
import weis.aeroelasticse.runFAST_pywrapper as fastwrap
from weis.aeroelasticse.FAST_post         import FAST_IO_timeseries
from wisdem.floatingse.floating_frame import NULL, NNODES_MAX, NELEM_MAX
from wisdem.floatingse.member import get_nfull as get_nfull_float
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from functools import partial
from pCrunch import PowerProduction
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.control.LinearModel import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse import FileTools
from weis.aeroelasticse.turbsim_file   import TurbSimFile
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.utils import OLAFParams
from ROSCO_toolbox import control_interface as ROSCO_ci
from pCrunch.io import OpenFASTOutput
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams


weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

import pickle

if MPI:
    from mpi4py   import MPI

if platform.system() == 'Windows':
    lib_ext = '.dll'
elif platform.system() == 'Darwin':
    lib_ext = '.dylib'
else:
    lib_ext = '.so'

def make_coarse_grid(s_grid, diam):

    s_coarse = [s_grid[0]]
    slope = np.diff(diam) / np.diff(s_grid)
    for k in range(slope.size-1):
        if np.abs(slope[k]-slope[k+1]) > 1e-2:
            s_coarse.append(s_grid[k+1])
    s_coarse.append(s_grid[-1])
    return np.array(s_coarse)

    
class FASTLoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        mat_init_options = modopt['materials']

        self.n_blades      = modopt['assembly']['number_of_blades']
        self.n_span        = n_span    = rotorse_options['n_span']
        self.n_pc          = n_pc      = rotorse_options['n_pc']

        # Environmental Conditions needed regardless of where model comes from
        self.add_input('V_cutin',     val=0.0, units='m/s',      desc='Minimum wind speed where turbine operates (cut-in)')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed where turbine operates (cut-out)')
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_input('hub_height',                val=0.0, units='m', desc='hub height')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbulence class')
        self.add_input('Rtip',              val=0.0, units='m', desc='dimensional radius of tip')
        self.add_input('shearExp',    val=0.0,                   desc='shear exponent')

        if not self.options['modeling_options']['Level3']['from_openfast']:
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

            n_freq_blade = int(rotorse_options['n_freq']/2)
            n_pc         = int(rotorse_options['n_pc'])

            n_height_tow = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_tower']
            n_height_mon = self.options['modeling_options']['WISDEM']['TowerSE']['n_height_monopile']
            n_height     = self.options['modeling_options']['WISDEM']['TowerSE']['n_height']
            n_full_tow   = get_nfull(n_height_tow)
            n_full_mon   = get_nfull(n_height_mon)
            n_full       = get_nfull(n_height)
            n_freq_tower = int(NFREQ/2)

            self.n_xy          = n_xy      = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
            self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
            self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
            self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1

            self.te_ss_var       = rotorse_options['te_ss']
            self.te_ps_var       = rotorse_options['te_ps']
            self.spar_cap_ss_var = rotorse_options['spar_cap_ss']
            self.spar_cap_ps_var = rotorse_options['spar_cap_ps']

            n_height_tow = modopt['WISDEM']['TowerSE']['n_height_tower']
            n_height_mon = modopt['WISDEM']['TowerSE']['n_height_monopile']
            n_height     = modopt['WISDEM']['TowerSE']['n_height']
            if modopt['flags']['floating']:
                n_full_tow   = get_nfull_float(n_height_tow)
                n_full_mon   = get_nfull_float(n_height_mon)
                n_full       = get_nfull_float(n_height)
            else:
                n_full_tow   = get_nfull(n_height_tow)
                n_full_mon   = get_nfull(n_height_mon)
                n_full       = get_nfull(n_height)
            n_freq_tower = int(NFREQ/2)
            n_freq_blade = int(rotorse_options['n_freq']/2)
            n_pc         = int(rotorse_options['n_pc'])

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
            self.add_discrete_input("platform_elem_memid", [0]*NELEM_MAX)
            self.add_input("platform_total_center_of_mass", np.zeros(3), units="m")
            self.add_input("platform_mass", 0.0, units="kg")
            self.add_input("platform_I_total", np.zeros(6), units="kg*m**2")

            if modopt['flags']["floating"]:
                n_member = modopt["floating"]["members"]["n_members"]
                for k in range(n_member):
                    n_height_mem = modopt["floating"]["members"]["n_height"][k]
                    self.add_input(f"member{k}:joint1", np.zeros(3), units="m")
                    self.add_input(f"member{k}:joint2", np.zeros(3), units="m")
                    self.add_input(f"member{k}:s", np.zeros(n_height_mem))
                    self.add_input(f"member{k}:s_ghost1", 0.0)
                    self.add_input(f"member{k}:s_ghost2", 0.0)
                    self.add_input(f"member{k}:outer_diameter", np.zeros(n_height_mem), units="m")
                    self.add_input(f"member{k}:wall_thickness", np.zeros(n_height_mem-1), units="m")

            # Turbine level inputs
            self.add_discrete_input('rotor_orientation',val='upwind', desc='Rotor orientation, either upwind or downwind.')
            self.add_input('control_ratedPower',        val=0.,  units='W',    desc='machine power rating')
            self.add_input('control_maxOmega',          val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
            self.add_input('control_maxTS',             val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
            self.add_input('cone',             val=0.0, units='deg',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
            self.add_input('tilt',             val=0.0, units='deg',   desc='Nacelle uptilt angle. A standard machine has positive values.')
            self.add_input('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')

            # Initial conditions
            self.add_input('U',        val=np.zeros(n_pc), units='m/s', desc='wind speeds')
            self.add_input('Omega',    val=np.zeros(n_pc), units='rpm', desc='rotation speeds to run')
            self.add_input('pitch',    val=np.zeros(n_pc), units='deg', desc='pitch angles to run')

            # Cp-Ct-Cq surfaces
            self.add_input('Cp_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero power coefficient')
            self.add_input('Ct_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero thrust coefficient')
            self.add_input('Cq_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero torque coefficient')
            self.add_input('pitch_vector',  val=np.zeros(n_pitch), units='deg',  desc='Pitch vector used')
            self.add_input('tsr_vector',    val=np.zeros(n_tsr),                 desc='TSR vector used')
            self.add_input('U_vector',      val=np.zeros(n_U),     units='m/s',  desc='Wind speed vector used')

            # Environmental conditions
            self.add_input('V_R25',       val=0.0, units='m/s',      desc='region 2.5 transition wind speed')
            self.add_input('Vgust',       val=0.0, units='m/s',      desc='gust wind speed')
            self.add_input('V_extreme1',  val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 1-year retunr period')
            self.add_input('V_extreme50', val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 50-year retunr period')
            self.add_input('V_mean_iec',  val=0.0, units='m/s',      desc='IEC mean wind for turbulence class')
            
            self.add_input('rho',         val=0.0, units='kg/m**3',  desc='density of air')
            self.add_input('mu',          val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
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
            mooropt = modopt["mooring"]
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

            # Inputs required for fatigue processing
            self.add_input('lifetime', val=25.0, units='yr', desc='Turbine design lifetime')
            self.add_input('blade_sparU_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_sparL_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_teU_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_teL_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_root_sparU_load2stress',   val=np.ones(6), units="m**2",  desc='Blade root upper spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_root_sparL_load2stress',   val=np.ones(6), units="m**2",  desc='Blade root lower spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teU_load2stress',   val=np.ones(6), units="m**2",  desc='Blade max chord upper trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teL_load2stress',   val=np.ones(6), units="m**2",  desc='Blade max chord lower trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_wohlerexp',   val=1.0,   desc='Low speed shaft Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_wohlerA',     val=1.0,   desc='Low speed shaft parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_ultstress',   val=1.0, units="Pa",   desc='Low speed shaft Ultimate stress for material')
            self.add_input('lss_axial_load2stress',   val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_shear_load2stress',   val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_wohlerexp',   val=np.ones(n_height-1),   desc='Tower-monopile Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_wohlerA',     val=np.ones(n_height-1),   desc='Tower-monopile parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_ultstress',   val=np.ones(n_height-1), units="Pa",   desc='Tower-monopile ultimate stress for material')
            self.add_input('tower_axial_load2stress',   val=np.ones([n_height-1,6]), units="m**2",  desc='Tower-monopile coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_shear_load2stress',   val=np.ones([n_height-1,6]), units="m**2",  desc='Tower-monopile coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
        

        # DLC options
        n_ws_dlc11 = modopt['DLC_driver']['n_ws_dlc11']

        # OpenFAST options
        OFmgmt = modopt['General']['openfast_configuration']
        self.model_only = OFmgmt['model_only']
        FAST_directory_base = OFmgmt['OF_run_dir']
        # If the path is relative, make it an absolute path
        if not os.path.isabs(FAST_directory_base):
            FAST_directory_base = os.path.join(os.getcwd(), FAST_directory_base)
        # Flag to clear OpenFAST run folder. Use it only if disk space is an issue
        self.clean_FAST_directory = False
        self.FAST_InputFile = OFmgmt['OF_run_fst']
        # File naming changes whether in MPI or not
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            self.FAST_runDirectory = os.path.join(FAST_directory_base,'rank_%000d'%int(rank))
            self.FAST_namingOut = self.FAST_InputFile+'_%000d'%int(rank)
        else:
            self.FAST_runDirectory = FAST_directory_base
            self.FAST_namingOut = self.FAST_InputFile
        self.wind_directory = os.path.join(self.FAST_runDirectory, 'wind')
        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)
        if not os.path.exists(self.wind_directory):
            os.mkdir(self.wind_directory)
        # Number of cores used outside of MPI. If larger than 1, the multiprocessing module is called
        self.cores = OFmgmt['cores']
        self.case = {}
        self.channels = {}
        self.mpi_run = False
        if 'mpi_run' in OFmgmt.keys():
            self.mpi_run         = OFmgmt['mpi_run']
            if self.mpi_run:
                self.mpi_comm_map_down   = OFmgmt['mpi_comm_map_down']

        # User-defined FAST library/executable
        if OFmgmt['FAST_exe'] != 'none':
            if os.path.isabs(OFmgmt['FAST_exe']):
                self.FAST_exe = OFmgmt['FAST_exe']
            else:
                self.FAST_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             OFmgmt['FAST_exe'])
        else:
            self.FAST_exe = 'none'

        if OFmgmt['FAST_lib'] != 'none':
            if os.path.isabs(OFmgmt['FAST_lib']):
                self.FAST_lib = OFmgmt['FAST_lib']
            else:
                self.FAST_lib = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             OFmgmt['FAST_lib'])
        else:
            self.FAST_lib = 'none'

        # Rotor power outputs
        if modopt['DLC_driver']['n_ws_dlc11'] > 0:
            self.add_output('V_out', val=np.zeros(n_ws_dlc11), units='m/s', desc='wind speed vector from the OF simulations')
            self.add_output('P_out', val=np.zeros(n_ws_dlc11), units='W', desc='rotor electrical power')
            self.add_output('Cp_out', val=np.zeros(n_ws_dlc11), desc='rotor aero power coefficient')
            self.add_output('Omega_out', val=np.zeros(n_ws_dlc11), units='rpm', desc='rotation speeds to run')
            self.add_output('pitch_out', val=np.zeros(n_ws_dlc11), units='deg', desc='pitch angles to run')
            self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production reconstructed from the openfast simulations')

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')

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

        self.add_output('max_TwrBsMyt',val=0.0, units='kN*m', desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt',val=0.0, units='kN*m', desc='damage equivalent load of tower base bending moment in fore-aft direction')
        
        # Tower outputs
        if not self.options['modeling_options']['Level3']['from_openfast']:
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

        # Fatigue output
        self.add_output('damage_blade_root_sparU', val=0.0, desc="Miner's rule cumulative damage to upper spar cap at blade root")
        self.add_output('damage_blade_root_sparL', val=0.0, desc="Miner's rule cumulative damage to lower spar cap at blade root")
        self.add_output('damage_blade_maxc_teU', val=0.0, desc="Miner's rule cumulative damage to upper trailing edge at blade max chord")
        self.add_output('damage_blade_maxc_teL', val=0.0, desc="Miner's rule cumulative damage to lower trailing edge at blade max chord")
        self.add_output('damage_lss', val=0.0, desc="Miner's rule cumulative damage to low speed shaft at hub attachment")
        self.add_output('damage_tower_base', val=0.0, desc="Miner's rule cumulative damage at tower base")
        self.add_output('damage_monopile_base', val=0.0, desc="Miner's rule cumulative damage at monopile base")

        self.add_discrete_output('fst_vt_out', val={})

        # Iteration counter for openfast calls. Initialize at -1 so 0 after first call
        self.of_inumber = -1
        self.sim_idx = -1

        if modopt['Level2']['flag']:
            self.lin_pkl_file_name = os.path.join(self.options['opt_options']['general']['folder_output'], 'ABCD_matrices.pkl')
            ABCD_list = []

            with open(self.lin_pkl_file_name, 'wb') as handle:
                pickle.dump(ABCD_list, handle)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        modopt = self.options['modeling_options']
        #print(impl.world_comm().rank, 'Rotor_fast','start')
        sys.stdout.flush()

        if modopt['Level2']['flag']:
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

        if not modopt['Level3']['from_openfast']:
            fst_vt = self.update_FAST_model(fst_vt, inputs, discrete_inputs)
        else:
            fast_reader = InputReader_OpenFAST()
            fast_reader.FAST_InputFile  = modopt['Level3']['openfast_file']   # FAST input file (ext=.fst)
            if os.path.isabs(modopt['Level3']['openfast_dir']):
                fast_reader.FAST_directory  = modopt['Level3']['openfast_dir']   # Path to fst directory files
            else:
                fast_reader.FAST_directory  = os.path.join(weis_dir, modopt['Level3']['openfast_dir'])
            fast_reader.path2dll            = modopt['General']['openfast_configuration']['path2dll']   # Path to dll file
            fast_reader.execute()
            fst_vt = fast_reader.fst_vt
            fst_vt = self.load_FAST_model_opts(fst_vt)

            # Fix TwrTI: WEIS modeling options have it as a single value...
            if not isinstance(fst_vt['AeroDyn15']['TwrTI'],list):
                fst_vt['AeroDyn15']['TwrTI'] = [fst_vt['AeroDyn15']['TwrTI']] * len(fst_vt['AeroDyn15']['TwrElev'])

            # Fix AddF0: Should be a n x 1 array (list of lists):
            fst_vt['HydroDyn']['AddF0'] = [[F0] for F0 in fst_vt['HydroDyn']['AddF0']]

            if modopt['ROSCO']['flag']:
                fst_vt['DISCON_in'] = modopt['General']['openfast_configuration']['fst_vt']['DISCON_in']
                
                
        if self.model_only == True:
            # Write input OF files, but do not run OF
            self.write_FAST(fst_vt, discrete_outputs)
        else:
            # Write OF model and run
            summary_stats, extreme_table, DELs, Damage, case_list, case_name, chan_time, dlc_generator  = self.run_FAST(inputs, discrete_inputs, fst_vt)

            if modopt['Level2']['flag']:
                LinearTurbine = LinearTurbineModel(
                self.FAST_runDirectory,
                self.lin_case_name,
                nlin=modopt['Level2']['linearization']['NLinTimes']
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
                    self.FAST_runDirectory,
                    'OutOps.yaml',OutOps)

                # Run linear simulation:

                # Get case list, wind inputs should have already been generated
                if modopt['Level2']['simulation']['flag']:

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
                    dam = {}
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

                        output = OpenFASTOutput.from_dict(l2_out, sim_name, magnitude_channels=self.magnitude_channels)

                        _name, _ss, _et, _dl, _dam = self.la._process_output(output)
                        ss[_name] = _ss
                        et[_name] = _et
                        dl[_name] = _dl
                        dam[_name] = _dam
                        ct.append(l2_out)

                        output.df.to_pickle(os.path.join(self.FAST_runDirectory,sim_name+'.p'))

                        summary_stats, extreme_table, DELs, Damage = self.la.post_process(ss, et, dl, dam)

        # Post process at both Level 2 and 3
        self.post_process(summary_stats, extreme_table, DELs, Damage, case_list, dlc_generator, chan_time, inputs, discrete_inputs, outputs, discrete_outputs)

        # delete run directory. not recommended for most cases, use for large parallelization problems where disk storage will otherwise fill up
        if self.clean_FAST_directory:
            try:
                shutil.rmtree(self.FAST_runDirectory)
            except:
                print('Failed to delete directory: %s'%self.FAST_runDirectory)

    def init_FAST_model(self):

        modopt = self.options['modeling_options']
        fst_vt = modopt['General']['openfast_configuration']['fst_vt']

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

        fst_vt = self.load_FAST_model_opts(fst_vt)

        return fst_vt

    def load_FAST_model_opts(self,fst_vt):

        modeling_options = self.options['modeling_options']

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
        
        fst_vt['ServoDyn']['DLL_FileName'] = modeling_options['General']['openfast_configuration']['path2dll']

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

        modopt = self.options['modeling_options']

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
        fst_vt['ElastoDyn']['OverHang']   = k*inputs['overhang'][0] / np.cos(np.deg2rad(inputs['tilt'][0])) # OpenFAST defines the overhang tilted (!)
        fst_vt['ElastoDyn']['GBoxEff']    = inputs['gearbox_efficiency'][0] * 100.
        fst_vt['ElastoDyn']['GBRatio']    = inputs['gearbox_ratio'][0]

        # Update ServoDyn
        fst_vt['ServoDyn']['GenEff']       = float(inputs['generator_efficiency']/inputs['gearbox_efficiency']) * 100.
        fst_vt['ServoDyn']['PitManRat(1)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(2)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(3)'] = float(inputs['max_pitch_rate'])

        # Structural Control: these could be defaulted modeling options, but we will add them as DVs later, so we'll hard code them here for now
        fst_vt['ServoDyn']['NumBStC']       = 0
        fst_vt['ServoDyn']['BStCfiles']     = "unused"
        fst_vt['ServoDyn']['NumNStC']       = 0
        fst_vt['ServoDyn']['NStCfiles']     = "unused"
        fst_vt['ServoDyn']['NumTStC']       = 0
        fst_vt['ServoDyn']['TStCfiles']     = "unused"
        fst_vt['ServoDyn']['NumSStC']       = 0
        fst_vt['ServoDyn']['SStCfiles']     = "unused"

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
        tower_top_height = float(inputs['hub_height']) - float(inputs['distance_tt_hub']) # Height of tower above ground level [onshore] or MSL [offshore] (meters)
        # The Twr2Shft is just the difference between hub height, tower top height, and sin(tilt)*overhang
        fst_vt['ElastoDyn']['Twr2Shft']  = float(inputs['hub_height']) - tower_top_height - abs(fst_vt['ElastoDyn']['OverHang'])*np.sin(np.deg2rad(inputs['tilt'][0]))
        fst_vt['ElastoDyn']['GenIner']   = float(inputs['GenIner'])

        # Mass and inertia inputs
        fst_vt['ElastoDyn']['TipMass(1)'] = 0.
        fst_vt['ElastoDyn']['TipMass(2)'] = 0.
        fst_vt['ElastoDyn']['TipMass(3)'] = 0.

        tower_base_height = max(float(inputs['tower_base_height']), float(inputs["platform_total_center_of_mass"][2]))
        fst_vt['ElastoDyn']['TowerBsHt'] = tower_base_height # Height of tower base above ground level [onshore] or MSL [offshore] (meters)
        fst_vt['ElastoDyn']['TowerHt']   = tower_top_height

        # TODO: There is some confusion on PtfmRefzt
        # DZ: based on the openfast r-tests:
        #   if this is floating, the z ref. point is 0.  Is this the reference that platform_total_center_of_mass is relative to?
        #   if fixed bottom, it's the tower base height.
        if modopt['flags']['floating']:
            fst_vt['ElastoDyn']['PtfmMass'] = float(inputs["platform_mass"])
            fst_vt['ElastoDyn']['PtfmRIner'] = float(inputs["platform_I_total"][0])
            fst_vt['ElastoDyn']['PtfmPIner'] = float(inputs["platform_I_total"][1])
            fst_vt['ElastoDyn']['PtfmYIner'] = float(inputs["platform_I_total"][2])
            fst_vt['ElastoDyn']['PtfmCMxt'] = float(inputs["platform_total_center_of_mass"][0])
            fst_vt['ElastoDyn']['PtfmCMyt'] = float(inputs["platform_total_center_of_mass"][1])
            fst_vt['ElastoDyn']['PtfmCMzt'] = float(inputs["platform_total_center_of_mass"][2])
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
        twr_d     = inputs['tower_outer_diameter']
        twr_index = np.argmin(abs(twr_elev - np.maximum(1.0, tower_base_height)))
        cd_index  = 0
        if twr_elev[twr_index] <= 1.:
            twr_index += 1
            cd_index  += 1
        fst_vt['AeroDyn15']['NumTwrNds'] = len(twr_elev[twr_index:])
        fst_vt['AeroDyn15']['TwrElev']   = twr_elev[twr_index:]
        fst_vt['AeroDyn15']['TwrDiam']   = twr_d[twr_index:]
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
        n_height_mon = modopt['WISDEM']['TowerSE']['n_height_monopile']
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
        BlCrvAC, BlSwpAC = self.get_ac_axis(inputs)
        fst_vt['AeroDynBlade']['BlCrvAC']  = BlCrvAC
        fst_vt['AeroDynBlade']['BlSwpAC']  = BlSwpAC
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(BlCrvAC)/np.gradient(r)))
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
            fst_vt['AeroDyn15']['AFTabMod'] = modopt["WISDEM"]["RotorSE"]["AFTabMod"]
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
                if modopt['General']['openfast_configuration']['generate_af_coords']:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = '@"AF{:02d}_Coords.txt"'.format(i)
                else:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = '0'

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
        if modopt['flags']['monopile']:
            mono_index = twr_index+1 # Duplicate intersection point
            n_joints = len(twr_d[1:mono_index]) # Omit submerged pile
            n_members = n_joints - 1
            itrans = n_joints - 1
            fst_vt['SubDyn']['JointXss'] = np.zeros( n_joints )
            fst_vt['SubDyn']['JointYss'] = np.zeros( n_joints )
            fst_vt['SubDyn']['JointZss'] = twr_elev[1:mono_index]
            fst_vt['SubDyn']['NReact'] = 1
            fst_vt['SubDyn']['RJointID'] = [1]
            fst_vt['SubDyn']['RctTDXss'] = fst_vt['SubDyn']['RctTDYss'] = fst_vt['SubDyn']['RctTDZss'] = [1]
            fst_vt['SubDyn']['RctRDXss'] = fst_vt['SubDyn']['RctRDYss'] = fst_vt['SubDyn']['RctRDZss'] = [1]
            fst_vt['SubDyn']['NInterf'] = 1
            fst_vt['SubDyn']['IJointID'] = [n_joints]
            fst_vt['SubDyn']['MJointID1'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['MJointID2'] = np.arange( n_members, dtype=np.int_ ) + 2
            fst_vt['SubDyn']['YoungE1'] = inputs['tower_E'][1:mono_index]
            fst_vt['SubDyn']['ShearG1'] = inputs['tower_G'][1:mono_index]
            fst_vt['SubDyn']['MatDens1'] = inputs['tower_rho'][1:mono_index]
            fst_vt['SubDyn']['XsecD'] = util.nodal2sectional(twr_d[1:mono_index])[0] # Don't need deriv
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

        elif modopt['flags']['floating']:
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
            if modopt['floating']['transition_joint'] is None:
                fst_vt['SubDyn']['NInterf'] = 0
                fst_vt['SubDyn']['IJointID'] = []
            else:
                fst_vt['SubDyn']['NInterf'] = 1
                fst_vt['SubDyn']['IJointID'] = [itrans+1]
            fst_vt['SubDyn']['MJointID1'] = N1+1
            fst_vt['SubDyn']['MJointID2'] = N2+1

            fst_vt['SubDyn']['YoungE1'] = inputs["platform_elem_E"][:n_members]
            fst_vt['SubDyn']['ShearG1'] = inputs["platform_elem_G"][:n_members]
            fst_vt['SubDyn']['MatDens1'] = inputs["platform_elem_rho"][:n_members]
            fst_vt['SubDyn']['XsecD'] = inputs["platform_elem_D"][:n_members]
            fst_vt['SubDyn']['XsecT'] = inputs["platform_elem_t"][:n_members]

        # SubDyn inputs- offshore generic
        if modopt['flags']['offshore']:
            if fst_vt['SubDyn']['SDdeltaT']<=-999.0: fst_vt['SubDyn']['SDdeltaT'] = "DEFAULT"
            fst_vt['SubDyn']['JDampings'] = [str(m) for m in fst_vt['SubDyn']['JDampings']]
            fst_vt['SubDyn']['GuyanDamp'] = np.vstack( tuple([fst_vt['SubDyn']['GuyanDamp'+str(m+1)] for m in range(6)]) )
            fst_vt['SubDyn']['Rct_SoilFile'] = [""]*fst_vt['SubDyn']['NReact']
            fst_vt['SubDyn']['NJoints'] = n_joints
            fst_vt['SubDyn']['JointID'] = np.arange( n_joints, dtype=np.int_) + 1
            fst_vt['SubDyn']['JointType'] = np.ones( n_joints, dtype=np.int_)
            fst_vt['SubDyn']['JointDirX'] = fst_vt['SubDyn']['JointDirY'] = fst_vt['SubDyn']['JointDirZ'] = np.zeros( n_joints )
            fst_vt['SubDyn']['JointStiff'] = np.zeros( n_joints )
            fst_vt['SubDyn']['ItfTDXss'] = fst_vt['SubDyn']['ItfTDYss'] = fst_vt['SubDyn']['ItfTDZss'] = [1]
            fst_vt['SubDyn']['ItfRDXss'] = fst_vt['SubDyn']['ItfRDYss'] = fst_vt['SubDyn']['ItfRDZss'] = [1]
            fst_vt['SubDyn']['NMembers'] = n_members
            fst_vt['SubDyn']['MemberID'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['MPropSetID1'] = fst_vt['SubDyn']['MPropSetID2'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['MType'] = np.ones( n_members, dtype=np.int_ )
            fst_vt['SubDyn']['NPropSets'] = n_members
            fst_vt['SubDyn']['PropSetID1'] = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['SubDyn']['NCablePropSets'] = 0
            fst_vt['SubDyn']['NRigidPropSets'] = 0
            fst_vt['SubDyn']['NCOSMs'] = 0
            fst_vt['SubDyn']['NXPropSets'] = 0
            fst_vt['SubDyn']['NCmass'] = 2 if inputs['gravity_foundation_mass'] > 0.0 else 1
            fst_vt['SubDyn']['CMJointID'] = [itrans+1]
            fst_vt['SubDyn']['JMass'] = [float(inputs['transition_piece_mass'])]
            fst_vt['SubDyn']['JMXX'] = [inputs['transition_piece_I'][0]]
            fst_vt['SubDyn']['JMYY'] = [inputs['transition_piece_I'][1]]
            fst_vt['SubDyn']['JMZZ'] = [inputs['transition_piece_I'][2]]
            fst_vt['SubDyn']['JMXY'] = fst_vt['SubDyn']['JMXZ'] = fst_vt['SubDyn']['JMYZ'] = [0.0]
            fst_vt['SubDyn']['MCGX'] = fst_vt['SubDyn']['MCGY'] = fst_vt['SubDyn']['MCGZ'] = [0.0]
            if inputs['gravity_foundation_mass'] > 0.0:
                fst_vt['SubDyn']['CMJointID'] += [1]
                fst_vt['SubDyn']['JMass'] += [float(inputs['gravity_foundation_mass'])]
                fst_vt['SubDyn']['JMXX'] += [inputs['gravity_foundation_I'][0]]
                fst_vt['SubDyn']['JMYY'] += [inputs['gravity_foundation_I'][1]]
                fst_vt['SubDyn']['JMZZ'] += [inputs['gravity_foundation_I'][2]]
                fst_vt['SubDyn']['JMXY'] += [0.0]
                fst_vt['SubDyn']['JMXZ'] += [0.0]
                fst_vt['SubDyn']['JMYZ'] += [0.0]
                fst_vt['SubDyn']['MCGX'] += [0.0]
                fst_vt['SubDyn']['MCGY'] += [0.0]
                fst_vt['SubDyn']['MCGZ'] += [0.0]


        # HydroDyn inputs
        if modopt['flags']['monopile']:
            z_coarse = make_coarse_grid(twr_elev[1:mono_index], twr_d[1:mono_index])
            n_joints = len(z_coarse)
            n_members = n_joints - 1
            joints_xyz = np.c_[np.zeros((n_joints,2)), z_coarse]
            d_coarse = np.interp(z_coarse, twr_elev[1:mono_index], twr_d[1:mono_index])
            t_coarse = util.sectional_interp(z_coarse, twr_elev[1:mono_index], inputs['tower_wall_thickness'][1:mono_index-1])
            N1 = np.arange( n_members, dtype=np.int_ ) + 1
            N2 = np.arange( n_members, dtype=np.int_ ) + 2
            
        elif modopt['flags']['floating']:
            joints_xyz = np.empty((0, 3))
            N1 = np.array([], dtype=np.int_)
            N2 = np.array([], dtype=np.int_)
            d_coarse = np.array([])
            t_coarse = np.array([])
            
            # Look over members and grab all nodes and internal connections
            n_member = modopt["floating"]["members"]["n_members"]
            for k in range(n_member):
                s_grid = inputs[f"member{k}:s"]
                idiam = inputs[f"member{k}:outer_diameter"]
                s_coarse = make_coarse_grid(s_grid, idiam)
                s_coarse = np.unique( np.minimum( np.maximum(s_coarse, inputs[f"member{k}:s_ghost1"]), inputs[f"member{k}:s_ghost2"]) )
                id_coarse = np.interp(s_coarse, s_grid, idiam)
                it_coarse = util.sectional_interp(s_coarse, s_grid, inputs[f"member{k}:wall_thickness"])
                xyz0 = inputs[f"member{k}:joint1"]
                xyz1 = inputs[f"member{k}:joint2"]
                dxyz = xyz1 - xyz0
                inode_xyz = np.outer(s_coarse, dxyz) + xyz0[np.newaxis, :]
                inode_range = np.arange(inode_xyz.shape[0] - 1)

                nk = joints_xyz.shape[0]
                N1 = np.append(N1, nk + inode_range + 1)
                N2 = np.append(N2, nk + inode_range + 2)
                d_coarse = np.append(d_coarse, id_coarse)
                t_coarse = np.append(t_coarse, it_coarse)
                joints_xyz = np.append(joints_xyz, inode_xyz, axis=0)
                
        if modopt['flags']['offshore']:
            fst_vt['HydroDyn']['WtrDens'] = float(inputs['rho_water'])
            fst_vt['HydroDyn']['WtrDpth'] = float(inputs['water_depth'])
            fst_vt['HydroDyn']['MSL2SWL'] = 0.0
            fst_vt['HydroDyn']['WaveHs'] = float(inputs['Hsig_wave'])
            fst_vt['HydroDyn']['WaveTp'] = float(inputs['Tsig_wave'])
            if fst_vt['HydroDyn']['WavePkShp']<=-999.0: fst_vt['HydroDyn']['WavePkShp'] = "DEFAULT"
            fst_vt['HydroDyn']['WaveDir'] = float(inputs['beta_wave'])
            fst_vt['HydroDyn']['WaveDirRange'] = np.rad2deg(fst_vt['HydroDyn']['WaveDirRange'])
            fst_vt['HydroDyn']['WaveElevxi'] = [str(m) for m in fst_vt['HydroDyn']['WaveElevxi']]
            fst_vt['HydroDyn']['WaveElevyi'] = [str(m) for m in fst_vt['HydroDyn']['WaveElevyi']]
            fst_vt['HydroDyn']['CurrSSDir'] = "DEFAULT" if fst_vt['HydroDyn']['CurrSSDir']<=-999.0 else np.rad2deg(fst_vt['HydroDyn']['CurrSSDir'])
            fst_vt['HydroDyn']['AddF0'] = np.array( fst_vt['HydroDyn']['AddF0'] ).reshape(-1,1)
            fst_vt['HydroDyn']['AddCLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddCLin'+str(m+1)] for m in range(6)]) )
            fst_vt['HydroDyn']['AddBLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddBLin'+str(m+1)] for m in range(6)]) )
            BQuad = np.vstack( tuple([fst_vt['HydroDyn']['AddBQuad'+str(m+1)] for m in range(6)]) )
            if np.any(BQuad):
                print('WARNING: You are adding in additional drag terms that may double count strip theory estimated viscous drag terms.  Please zero out the BQuad entries or use modeling options SimplCd/a/p and/or potential_model_override and/or potential_bem_members to suppress strip theory for the members')
            fst_vt['HydroDyn']['AddBQuad'] = BQuad
            fst_vt['HydroDyn']['NAxCoef'] = 1
            fst_vt['HydroDyn']['AxCoefID'] = 1 + np.arange( fst_vt['HydroDyn']['NAxCoef'], dtype=np.int_)
            fst_vt['HydroDyn']['AxCd'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
            fst_vt['HydroDyn']['AxCa'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
            fst_vt['HydroDyn']['AxCp'] = np.ones( fst_vt['HydroDyn']['NAxCoef'] )
            # Use coarse member nodes for HydroDyn
                
            # Tweak z-position
            idx = np.where(joints_xyz[:,2]==-fst_vt['HydroDyn']['WtrDpth'])[0]
            if len(idx) > 0:
                joints_xyz[idx,2] -= 1e-2
            # Store data
            n_joints = joints_xyz.shape[0]
            n_members = N1.shape[0]
            imembers = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['HydroDyn']['NJoints'] = n_joints
            fst_vt['HydroDyn']['JointID'] = 1 + np.arange( n_joints, dtype=np.int_)
            fst_vt['HydroDyn']['Jointxi'] = joints_xyz[:,0]
            fst_vt['HydroDyn']['Jointyi'] = joints_xyz[:,1]
            fst_vt['HydroDyn']['Jointzi'] = joints_xyz[:,2]
            fst_vt['HydroDyn']['NPropSets'] = n_members
            fst_vt['HydroDyn']['PropSetID'] = imembers
            fst_vt['HydroDyn']['PropD'] = d_coarse
            fst_vt['HydroDyn']['PropThck'] = t_coarse
            fst_vt['HydroDyn']['NMembers'] = n_members
            fst_vt['HydroDyn']['MemberID'] = imembers
            fst_vt['HydroDyn']['MJointID1'] = N1
            fst_vt['HydroDyn']['MJointID2'] = N2
            fst_vt['HydroDyn']['MPropSetID1'] = fst_vt['HydroDyn']['MPropSetID2'] = imembers
            fst_vt['HydroDyn']['MDivSize'] = 0.5*np.ones( fst_vt['HydroDyn']['NMembers'] )
            fst_vt['HydroDyn']['MCoefMod'] = np.ones( fst_vt['HydroDyn']['NMembers'], dtype=np.int_)
            fst_vt['HydroDyn']['JointAxID'] = np.ones( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['JointOvrlp'] = np.zeros( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['NCoefDpth'] = 0
            fst_vt['HydroDyn']['NCoefMembers'] = 0
            fst_vt['HydroDyn']['NFillGroups'] = 0
            fst_vt['HydroDyn']['NMGDepths'] = 0

            if modopt["Level1"]["potential_model_override"] == 1:
                # Strip theory only, no BEM
                fst_vt['HydroDyn']['PropPot'] = [False] * fst_vt['HydroDyn']['NMembers']
            elif modopt["Level1"]["potential_model_override"] == 2:
                # BEM only, no strip theory
                fst_vt['HydroDyn']['SimplCd'] = fst_vt['HydroDyn']['SimplCdMG'] = 0.0
                fst_vt['HydroDyn']['SimplCa'] = fst_vt['HydroDyn']['SimplCaMG'] = 0.0
                fst_vt['HydroDyn']['SimplCp'] = fst_vt['HydroDyn']['SimplCpMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCd'] = fst_vt['HydroDyn']['SimplAxCdMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCa'] = fst_vt['HydroDyn']['SimplAxCaMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCp'] = fst_vt['HydroDyn']['SimplAxCpMG'] = 0.0
                fst_vt['HydroDyn']['PropPot'] = [True] * fst_vt['HydroDyn']['NMembers']
            else:
                PropPotBool = [False] * fst_vt['HydroDyn']['NMembers']
                for k in range(fst_vt['HydroDyn']['NMembers']):
                    idx = discrete_inputs['platform_elem_memid'][k]
                    PropPotBool[k] = modopt["Level1"]["model_potential"][idx]
                fst_vt['HydroDyn']['PropPot'] = PropPotBool

            if fst_vt['HydroDyn']['NBody'] > 1:
                raise Exception('Multiple HydroDyn bodies (NBody > 1) is currently not supported in WEIS')

            # Offset of body reference point
            fst_vt['HydroDyn']['PtfmRefxt']     = 0
            fst_vt['HydroDyn']['PtfmRefyt']     = 0
            fst_vt['HydroDyn']['PtfmRefzt']     = 0
            fst_vt['HydroDyn']['PtfmRefztRot']  = 0

            # If we're using the potential model, need these settings that aren't default
            if fst_vt['HydroDyn']['PotMod'] == 1:
                fst_vt['HydroDyn']['ExctnMod'] = 1
                fst_vt['HydroDyn']['RdtnMod'] = 1
                fst_vt['HydroDyn']['RdtnDT'] = "DEFAULT"

            # scale PtfmVol0 based on platform mass, temporary solution to buoyancy issue where spar's heave is very sensitive to platform mass
            if fst_vt['HydroDyn']['PtfmMass_Init']:
                fst_vt['HydroDyn']['PtfmVol0'] = float(inputs['platform_displacement']) * (1 + ((fst_vt['ElastoDyn']['PtfmMass'] / fst_vt['HydroDyn']['PtfmMass_Init']) - 1) * .9 )  #* 1.04 # 8029.21
            else:
                fst_vt['HydroDyn']['PtfmVol0'] = float(inputs['platform_displacement'])


        # Moordyn inputs
        if modopt["flags"]["mooring"]:
            mooropt = modopt["mooring"]
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
                    fst_vt['MoorDyn']['Type'][id2].lower().find('fix') >= 0):
                    fst_vt['MoorDyn']['NodeFair'][k] = id1+1
                    fst_vt['MoorDyn']['NodeAnch'][k] = id2+1
                elif (fst_vt['MoorDyn']['Type'][id2].lower() == 'vessel' and
                    fst_vt['MoorDyn']['Type'][id1].lower().find('fix') >= 0):
                    fst_vt['MoorDyn']['NodeFair'][k] = id2+1
                    fst_vt['MoorDyn']['NodeAnch'][k] = id1+1
                else:
                    print(discrete_inputs['node_names'])
                    print(mooropt["node1"][k], mooropt["node2"][k])
                    print(fst_vt['MoorDyn']['Type'][id1], fst_vt['MoorDyn']['Type'][id2])
                    raise ValueError('Mooring line seems to be between unknown endpoint types.')

            for key in fst_vt['MoorDyn']:
                fst_vt['MAP'][key] = copy.copy(fst_vt['MoorDyn'][key])

            for idx, node_type in enumerate(fst_vt['MAP']['Type']):
                if node_type == 'fixed':
                    fst_vt['MAP']['Type'][idx] = 'fix'

            # TODO: FIXME: these values are hardcoded for the IEA15MW linearization studies
            fst_vt['MAP']['LineType'] = ['main', 'main', 'main']
            fst_vt['MAP']['CB'] = np.ones(n_lines)
            fst_vt['MAP']['CIntDamp'] = np.zeros(n_lines)
            fst_vt['MAP']['Ca'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdn'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdt'] = np.zeros(n_lines)
            fst_vt['MAP']['B'] = np.zeros( n_nodes )
            fst_vt['MAP']['Option'] = ["outer_tol 1e-5"]

        return fst_vt

    def output_channels(self):
        modopt = self.options['modeling_options']

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
        channels_out += ["RotThrust", "LSShftFxs", "LSShftFys", "LSShftFzs", "LSShftFxa", "LSShftFya", "LSShftFza"]
        channels_out += ["RotTorq", "LSSTipMxs", "LSSTipMys", "LSSTipMzs", "LSSTipMxa", "LSSTipMya", "LSSTipMza"]
        channels_out += ["B1N1Alpha", "B1N2Alpha", "B1N3Alpha", "B1N4Alpha", "B1N5Alpha", "B1N6Alpha", "B1N7Alpha", "B1N8Alpha", "B1N9Alpha", "B2N1Alpha", "B2N2Alpha", "B2N3Alpha", "B2N4Alpha", "B2N5Alpha", "B2N6Alpha", "B2N7Alpha", "B2N8Alpha","B2N9Alpha"]
        channels_out += ["PtfmSurge", "PtfmSway", "PtfmHeave", "PtfmRoll", "PtfmPitch", "PtfmYaw","NcIMURAys"]
        if self.n_blades == 3:
            channels_out += ["TipDxc3", "TipDyc3", "TipDzc3", "RootMxc3", "RootMyc3", "RootMzc3", "TipDxb3", "TipDyb3", "TipDzb3", "RootMxb3",
                             "RootMyb3", "RootMzb3", "RootFxc3", "RootFyc3", "RootFzc3", "RootFxb3", "RootFyb3", "RootFzb3", "BldPitch3"]
            channels_out += ["B3N1Alpha", "B3N2Alpha", "B3N3Alpha", "B3N4Alpha", "B3N5Alpha", "B3N6Alpha", "B3N7Alpha", "B3N8Alpha", "B3N9Alpha"]

        # Channels for distributed aerodynamic control
        if self.options['modeling_options']['ROSCO']['Flp_Mode']:   # we're doing flap control
            channels_out += ['BLFLAP1', 'BLFLAP2', 'BLFLAP3']

        # Channels for wave outputs
        if modopt['flags']['offshore']:
            channels_out += ["Wave1Elev","WavesF1xi","WavesF1zi","WavesM1yi"]
            channels_out += ["WavesF2xi","WavesF2yi","WavesF2zi","WavesM2xi","WavesM2yi","WavesM2zi"]

        # Channels for monopile-based structure
        if modopt['flags']['monopile']:
            if modopt['Level3']['simulation']['CompSub']:
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
            else:
                raise Exception('CompSub must be 1 in the modeling options to run SubDyn and compute monopile loads')

        # Floating output channels
        if modopt['flags']['floating']:
            channels_out += ["PtfmPitch", "PtfmRoll", "PtfmYaw", "PtfmSurge", "PtfmSway", "PtfmHeave"]

        channels = {}
        for var in channels_out:
            channels[var] = True

        return channels

    def run_FAST(self, inputs, discrete_inputs, fst_vt):

        modopt = self.options['modeling_options']
        DLCs = modopt['DLC_driver']['DLCs']
        # Initialize the DLC generator
        cut_in = float(inputs['V_cutin'])
        cut_out = float(inputs['V_cutout'])
        rated = float(inputs['Vrated'])
        ws_class = discrete_inputs['turbine_class']
        wt_class = discrete_inputs['turbulence_class']
        hub_height = float(inputs['hub_height'])
        rotorD = float(inputs['Rtip'])*2.
        PLExp = float(inputs['shearExp'])
        fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
        fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
        metocean = modopt['DLC_driver']['metocean_conditions']
        dlc_generator = DLCGenerator(cut_in, cut_out, rated, ws_class, wt_class, fix_wind_seeds, fix_wave_seeds, metocean)
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)

        # Initialize parametric inputs
        WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
        WindFile_name = [''] * dlc_generator.n_cases
        rot_speed_initial = np.zeros(dlc_generator.n_cases)
        pitch_initial = np.zeros(dlc_generator.n_cases)
        WindHd = np.zeros(dlc_generator.n_cases)
        WaveHs = np.zeros(dlc_generator.n_cases)
        WaveTp = np.zeros(dlc_generator.n_cases)
        WaveHd = np.zeros(dlc_generator.n_cases)
        WaveGamma = np.zeros(dlc_generator.n_cases)
        WaveSeed1 = np.zeros(dlc_generator.n_cases, dtype=int)
        TMax = np.zeros(dlc_generator.n_cases)
        TStart = np.zeros(dlc_generator.n_cases)

        for i_case in range(dlc_generator.n_cases):
            if dlc_generator.cases[i_case].turbulent_wind:
                # Assign values common to all DLCs
                # Wind turbulence class
                dlc_generator.cases[i_case].IECturbc = wt_class
                # Reference height for wind speed
                dlc_generator.cases[i_case].RefHt = hub_height
                # Center of wind grid (TurbSim confusingly calls it HubHt)
                dlc_generator.cases[i_case].HubHt = hub_height
                # Height of wind grid, it stops 1 mm above the ground
                dlc_generator.cases[i_case].GridHeight = 2. * hub_height - 1.e-3
                # If OLAF is called, make wind grid high and big
                if fst_vt['AeroDyn15']['WakeMod'] == 3:
                    dlc_generator.cases[i_case].HubHt *= 3.
                    dlc_generator.cases[i_case].GridHeight *= 3.
                # Width of wind grid, same of height
                dlc_generator.cases[i_case].GridWidth = dlc_generator.cases[i_case].GridHeight
                # Power law exponent of wind shear
                dlc_generator.cases[i_case].PLExp = PLExp
                # Length of wind grids
                dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time

        # Generate wind files
        if MPI and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
            # mpi comm management
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            sub_ranks = self.mpi_comm_map_down[rank]
            size = len(sub_ranks)

            N_cases = dlc_generator.n_cases # total number of cases
            N_loops = int(np.ceil(float(N_cases)/float(size)))  # number of times function calls need to "loop"
            # iterate loops
            for i in range(N_loops):
                idx_s = i*size
                idx_e = min((i+1)*size, N_cases)

                for idx, i_case in enumerate(np.arange(idx_s,idx_e)):
                    data = [partial(generate_wind_files, dlc_generator, self.FAST_namingOut, self.wind_directory, rotorD, hub_height), i_case]
                    rank_j = sub_ranks[idx]
                    comm.send(data, dest=rank_j, tag=0)

                for idx, i_case in enumerate(np.arange(idx_s, idx_e)):
                    rank_j = sub_ranks[idx]
                    WindFile_type[i_case] , WindFile_name[i_case] = comm.recv(source=rank_j, tag=1)
        else:
            for i_case in range(dlc_generator.n_cases):
                WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
                    dlc_generator, self.FAST_namingOut, self.wind_directory, rotorD, hub_height, i_case)

        # Set initial rotor speed and pitch if the WT operates in this DLC and available,
        # otherwise set pitch to 90 deg and rotor speed to 0 rpm when not operating
        # set rotor speed to rated and pitch to 15 deg if operating
        for i_case in range(dlc_generator.n_cases):
            if dlc_generator.cases[i_case].turbine_status == 'operating':
                # We have initial conditions from WISDEM
                if ('U' in inputs) and ('Omega' in inputs) and ('pitch' in inputs):
                    rot_speed_initial[i_case] = np.interp(dlc_generator.cases[i_case].URef, inputs['U'], inputs['Omega'])
                    pitch_initial[i_case] = np.interp(dlc_generator.cases[i_case].URef, inputs['U'], inputs['pitch'])
                else:
                    rot_speed_initial[i_case]   = fst_vt['DISCON_in']['PC_RefSpd'] * 30 / np.pi
                    pitch_initial[i_case]       = 15
            else:
                rot_speed_initial[i_case] = 0.
                pitch_initial[i_case] = 90.
            # Wave inputs to HydroDyn
            WindHd[i_case] = dlc_generator.cases[i_case].wind_heading
            WaveHs[i_case] = dlc_generator.cases[i_case].wave_height
            WaveTp[i_case] = dlc_generator.cases[i_case].wave_period
            WaveHd[i_case] = dlc_generator.cases[i_case].wave_heading
            WaveGamma[i_case] = dlc_generator.cases[i_case].wave_gamma
            WaveSeed1[i_case] = dlc_generator.cases[i_case].wave_seed1
            TMax[i_case] = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
            TStart[i_case] = dlc_generator.cases[i_case].transient_time


        # Parameteric inputs
        case_inputs = {}
        # Main fst
        case_inputs[("Fst","TMax")] = {'vals':TMax, 'group':1}
        case_inputs[("Fst","TStart")] = {'vals':TStart, 'group':1}
        # Inflow wind
        case_inputs[("InflowWind","WindType")] = {'vals':WindFile_type, 'group':1}
        case_inputs[("InflowWind","FileName_BTS")] = {'vals':WindFile_name, 'group':1}
        case_inputs[("InflowWind","Filename_Uni")] = {'vals':WindFile_name, 'group':1}
        case_inputs[("InflowWind","RefLength")] = {'vals':[rotorD], 'group':0}
        case_inputs[("InflowWind","PropagationDir")] = {'vals':WindHd, 'group':1}
        # Initial conditions for rotor speed and pitch
        case_inputs[("ElastoDyn","RotSpeed")] = {'vals':rot_speed_initial, 'group':1}
        case_inputs[("ElastoDyn","BlPitch1")] = {'vals':pitch_initial, 'group':1}
        case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
        case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
        # Inputs to HydroDyn
        case_inputs[("HydroDyn","WaveHs")] = {'vals':WaveHs, 'group':1}
        case_inputs[("HydroDyn","WaveTp")] = {'vals':WaveTp, 'group':1}
        case_inputs[("HydroDyn","WaveDir")] = {'vals':WaveHd, 'group':1}
        case_inputs[("HydroDyn","WavePkShp")] = {'vals':WaveGamma, 'group':1}
        case_inputs[("HydroDyn","WaveSeed1")] = {'vals':WaveSeed1, 'group':1}

        # Append current DLC to full list of cases
        case_list, case_name = CaseGen_General(case_inputs, self.FAST_runDirectory, self.FAST_InputFile)
        channels= self.output_channels()
        
        
        # FAST wrapper setup
        # JJ->DZ: here is the first point in logic for linearization
        if modopt['Level2']['flag']:
            linearization_options               = modopt['Level2']['linearization']

            # Use openfast binary until library works
            fastBatch                           = LinearFAST(**linearization_options)
            fastBatch.FAST_lib                  = None      # linearization not working with library
            fastBatch.fst_vt                    = fst_vt
            fastBatch.cores                     = self.cores

            lin_case_list, lin_case_name        = fastBatch.gen_linear_cases(inputs)
            fastBatch.case_list                 = lin_case_list
            fastBatch.case_name_list            = lin_case_name

            # Save this list of linear cases for making linear model, not the best solution, but it works
            self.lin_case_name                  = lin_case_name
        else:
            fastBatch                           = fastwrap.runFAST_pywrapper_batch()
            fastBatch.case_list                 = case_list
            fastBatch.case_name_list            = case_name     
        
        fastBatch.channels          = channels
        fastBatch.FAST_runDirectory = self.FAST_runDirectory
        fastBatch.FAST_InputFile    = self.FAST_InputFile
        fastBatch.fst_vt            = fst_vt
        fastBatch.keep_time         = modopt['General']['openfast_configuration']['save_timeseries']
        fastBatch.post              = FAST_IO_timeseries
        fastBatch.use_exe           = modopt['General']['openfast_configuration']['use_exe']
        if self.FAST_exe != 'none':
            fastBatch.FAST_exe          = self.FAST_exe
        if self.FAST_lib != 'none':
            fastBatch.FAST_lib          = self.FAST_lib

        fastBatch.overwrite_outfiles = True  #<--- Debugging only, set to False to prevent OpenFAST from running if the .outb already exists

        # Initialize fatigue channels and setings
        # TODO: Stress Concentration Factor?
        magnitude_channels = dict( fastwrap.magnitude_channels_default )
        fatigue_channels =  dict( fastwrap.fatigue_channels_default )

        # Blade fatigue: spar caps at the root (upper & lower?), TE at max chord
        if not modopt['Level3']['from_openfast']:
            for u in ['U','L']:
                blade_fatigue_root = FatigueParams(load2stress=1.0,
                                                lifetime=inputs['lifetime'],
                                                slope=inputs[f'blade_spar{u}_wohlerexp'],
                                                ult_stress=inputs[f'blade_spar{u}_ultstress'],
                                                S_intercept=inputs[f'blade_spar{u}_wohlerA'])
                blade_fatigue_te = FatigueParams(load2stress=1.0,
                                                lifetime=inputs['lifetime'],
                                                slope=inputs[f'blade_te{u}_wohlerexp'],
                                                ult_stress=inputs[f'blade_te{u}_ultstress'],
                                                S_intercept=inputs[f'blade_te{u}_wohlerA'])
                
                for k in range(1,self.n_blades+1):
                    blade_root_Fz = blade_fatigue_root.copy()
                    blade_root_Fz.load2stress = inputs[f'blade_root_spar{u}_load2stress'][2]
                    fatigue_channels[f'RootSpar{u}_Fzb{k}'] = blade_root_Fz
                    magnitude_channels[f'RootSpar{u}_Fzb{k}'] = [f'RootFzb{k}']

                    blade_root_Mx = blade_fatigue_root.copy()
                    blade_root_Mx.load2stress = inputs[f'blade_root_spar{u}_load2stress'][3]
                    fatigue_channels[f'RootSpar{u}_Mxb{k}'] = blade_root_Mx
                    magnitude_channels[f'RootSpar{u}_Mxb{k}'] = [f'RootMxb{k}']

                    blade_root_My = blade_fatigue_root.copy()
                    blade_root_My.load2stress = inputs[f'blade_root_spar{u}_load2stress'][4]
                    fatigue_channels[f'RootSpar{u}_Myb{k}'] = blade_root_My
                    magnitude_channels[f'RootSpar{u}_Myb{k}'] = [f'RootMyb{k}']

                    blade_maxc_Fz = blade_fatigue_te.copy()
                    blade_maxc_Fz.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][2]
                    fatigue_channels[f'Spn2te{u}_FLzb{k}'] = blade_maxc_Fz
                    magnitude_channels[f'Spn2te{u}_FLzb{k}'] = [f'Spn2FLzb{k}']

                    blade_maxc_Mx = blade_fatigue_te.copy()
                    blade_maxc_Mx.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][3]
                    fatigue_channels[f'Spn2te{u}_MLxb{k}'] = blade_maxc_Mx
                    magnitude_channels[f'Spn2te{u}_MLxb{k}'] = [f'Spn2MLxb{k}']

                    blade_maxc_My = blade_fatigue_te.copy()
                    blade_maxc_My.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][4]
                    fatigue_channels[f'Spn2te{u}_MLyb{k}'] = blade_maxc_My
                    magnitude_channels[f'Spn2te{u}_MLyb{k}'] = [f'Spn2MLyb{k}']

            # Low speed shaft fatigue
            lss_fatigue = FatigueParams(load2stress=1.0,
                                        lifetime=inputs['lifetime'],
                                        slope=inputs['lss_wohlerexp'],
                                        ult_stress=inputs['lss_ultstress'],
                                        S_intercept=inputs['lss_wohlerA'])        
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['x','yz']):
                        idx = 3*ik+ix
                        lss_fatigue_ii = lss_fatigue.copy()
                        lss_fatigue_ii.load2stress = inputs[f'lss_{sstr}_load2stress'][idx]
                        fatigue_channels[f'LSShft{s}{k}{x}a'] = lss_fatigue_ii
                        if ix==0:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['RotThrust'] if ik==0 else ['RotTorq']
                        else:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['LSShftFya', 'LSShftFza'] if ik==0 else ['LSSTipMya', 'LSSTipMza']

            # Fatigue at the tower base
            n_height_mon = modopt['WISDEM']['TowerSE']['n_height_monopile']
            tower_fatigue_base = FatigueParams(load2stress=1.0,
                                            lifetime=inputs['lifetime'],
                                            slope=inputs['tower_wohlerexp'][n_height_mon],
                                            ult_stress=inputs['tower_ultstress'][n_height_mon],
                                            S_intercept=inputs['tower_wohlerA'][n_height_mon],)
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['xy','z']):
                        idx = 3*ik+2*ix
                        tower_fatigue_ii = tower_fatigue_base.copy()
                        tower_fatigue_ii.load2stress = inputs[f'tower_{sstr}_load2stress'][n_height_mon,idx]
                        fatigue_channels[f'TwrBs{s}{k}{x}t'] = tower_fatigue_ii
                        magnitude_channels[f'TwrBs{s}{k}{x}t'] = [f'TwrBs{k}{x}t'] if x=='z' else [f'TwrBs{k}xt', f'TwrBs{k}yt']

            # Fatigue at monopile base (mudline)
            if modopt['flags']['monopile']:
                monopile_fatigue_base = FatigueParams(load2stress=1.0,
                                                    lifetime=inputs['lifetime'],
                                                    slope=inputs['tower_wohlerexp'][0],
                                                    ult_stress=inputs['tower_ultstress'][0],
                                                    S_intercept=inputs['tower_wohlerA'][0])
                for s in ['Ax','Sh']:
                    sstr = 'axial' if s=='Ax' else 'shear'
                    for ik, k in enumerate(['F','M']):
                        for ix, x in enumerate(['xy','z']):
                            idx = 3*ik+2*ix
                            monopile_fatigue_ii = monopile_fatigue_base.copy()
                            monopile_fatigue_ii.load2stress = inputs[f'tower_{sstr}_load2stress'][0,idx]
                            fatigue_channels[f'M1N1{s}{k}K{x}e'] = monopile_fatigue_ii
                            magnitude_channels[f'M1N1{s}{k}K{x}e'] = [f'M1N1{k}K{x}e'] if x=='z' else [f'M1N1{k}Kxe', f'M1N1{k}Kye']

        # Store settings
        fastBatch.goodman            = modopt['General']['goodman_correction'] # Where does this get placed in schema?
        fastBatch.fatigue_channels   = fatigue_channels
        fastBatch.magnitude_channels = magnitude_channels
        self.la = LoadsAnalysis(
            outputs=[],
            magnitude_channels=magnitude_channels,
            fatigue_channels=fatigue_channels,
        )
        self.magnitude_channels = magnitude_channels

        # Run FAST
        if self.mpi_run and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
            summary_stats, extreme_table, DELs, Damage, chan_time = fastBatch.run_mpi(self.mpi_comm_map_down)
        else:
            if self.cores == 1:
                summary_stats, extreme_table, DELs, Damage, chan_time = fastBatch.run_serial()
            else:
                summary_stats, extreme_table, DELs, Damage, chan_time = fastBatch.run_multi(self.cores)

        self.fst_vt = fst_vt
        self.of_inumber = self.of_inumber + 1
        sys.stdout.flush()

        return summary_stats, extreme_table, DELs, Damage, case_list, case_name, chan_time, dlc_generator

    def post_process(self, summary_stats, extreme_table, DELs, damage, case_list, dlc_generator, chan_time, inputs, discrete_inputs, outputs, discrete_outputs):
        modopt = self.options['modeling_options']

        # Analysis
        if self.options['modeling_options']['flags']['blade']:
            outputs, discrete_outputs = self.get_blade_loading(summary_stats, extreme_table, inputs, discrete_inputs, outputs, discrete_outputs)
        if self.options['modeling_options']['flags']['tower']:
            outputs = self.get_tower_loading(summary_stats, extreme_table, inputs, outputs)
        # SubDyn is only supported in Level3: linearization in OpenFAST will be available in 3.0.0
        if modopt['flags']['monopile'] and modopt['Level3']['flag']:
            outputs = self.get_monopile_loading(summary_stats, extreme_table, inputs, outputs)

        if modopt['DLC_driver']['n_ws_dlc11'] > 0:
            outputs, discrete_outputs = self.calculate_AEP(summary_stats, case_list, dlc_generator, discrete_inputs, outputs, discrete_outputs)
            outputs, discrete_outputs = self.get_weighted_DELs(dlc_generator, DELs, damage, discrete_inputs, outputs, discrete_outputs)
        
        outputs, discrete_outputs = self.get_control_measures(summary_stats, inputs, discrete_inputs, outputs, discrete_outputs)

        if modopt['flags']['floating']:
            outputs, discrete_outputs = self.get_floating_measures(summary_stats, inputs, discrete_inputs,
                                                                   outputs, discrete_outputs)

        # Save Data
        if modopt['General']['openfast_configuration']['save_timeseries']:
            self.save_timeseries(chan_time)

        if modopt['General']['openfast_configuration']['save_iterations']:
            self.save_iterations(summary_stats,DELs)

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

        modopt = self.options['modeling_options']
        n_height_tow = modopt['WISDEM']['TowerSE']['n_height_tower']
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

        if not modopt['flags']['monopile']:
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

        modopt = self.options['modeling_options']
        n_height_mon = modopt['WISDEM']['TowerSE']['n_height_monopile']
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

    def calculate_AEP(self, sum_stats, case_list, dlc_generator, discrete_inputs, outputs, discrete_outputs):
        """
        Calculates annual energy production of the relevant DLCs in `case_list`.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        case_list : list
        dlc_list : list
        """
        ## Get AEP and power curve

        # determine which dlc will be used for the powercurve calculations, allows using dlc 1.1 if specific power curve calculations were not run

        idx_pwrcrv = []
        U = []
        for i_case in range(dlc_generator.n_cases):
            if dlc_generator.cases[i_case].label == '1.1':
                idx_pwrcrv = np.append(idx_pwrcrv, i_case)
                U = np.append(U, dlc_generator.cases[i_case].URef)

        stats_pwrcrv = sum_stats.iloc[idx_pwrcrv].copy()

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

    def get_weighted_DELs(self, dlc_generator, DELs, damage, discrete_inputs, outputs, discrete_outputs):
        modopt = self.options['modeling_options']

        # See if we have fatigue DLCs
        U = np.zeros(dlc_generator.n_cases)
        ifat = []
        for k in range(dlc_generator.n_cases):
            U[k] = dlc_generator.cases[k].URef
            
            if dlc_generator.cases[k].label in ['1.2', '6.4']:
                ifat.append( k )

        # If fatigue DLCs are present, then limit analysis to those only
        if len(ifat) > 0:
            U = U[ifat]
            DELs = DELs.iloc[ ifat ]
            damage = damage.iloc[ ifat ]
        
        # Get wind distribution probabilities, make sure they are normalized
        # This should also take care of averaging across seeds
        pp = PowerProduction(discrete_inputs['turbine_class'])
        ws_prob = pp.prob_WindDist(U, disttype='pdf')
        ws_prob /= ws_prob.sum()

        # Scale all DELs and damage by probability and collapse over the various DLCs (inner dot product)
        # Also work around NaNs
        DELs = DELs.fillna(0.0).multiply(ws_prob, axis=0).sum()
        damage = damage.fillna(0.0).multiply(ws_prob, axis=0).sum()
        
        # Standard DELs for blade root and tower base
        outputs['DEL_RootMyb'] = np.max([DELs[f'RootMyb{k+1}'] for k in range(self.n_blades)])
        outputs['DEL_TwrBsMyt'] = DELs['TwrBsM']
            
        # Compute total fatigue damage in spar caps at blade root and trailing edge at max chord location
        if not modopt['Level3']['from_openfast']:
            for k in range(1,self.n_blades+1):
                for u in ['U','L']:
                    damage[f'BladeRootSpar{u}_Axial{k}'] = (damage[f'RootSpar{u}_Fzb{k}'] +
                                                        damage[f'RootSpar{u}_Mxb{k}'] +
                                                        damage[f'RootSpar{u}_Myb{k}'])
                    damage[f'BladeMaxcTE{u}_Axial{k}'] = (damage[f'Spn2te{u}_FLzb{k}'] +
                                                        damage[f'Spn2te{u}_MLxb{k}'] +
                                                        damage[f'Spn2te{u}_MLyb{k}'])

            # Compute total fatigue damage in low speed shaft, tower base, monopile base
            damage['LSSAxial'] = 0.0
            damage['LSSShear'] = 0.0
            damage['TowerBaseAxial'] = 0.0
            damage['TowerBaseShear'] = 0.0
            damage['MonopileBaseAxial'] = 0.0
            damage['MonopileBaseShear'] = 0.0
            for s in ['Ax','Sh']:
                sstr = 'Axial' if s=='Ax' else 'Shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['x','yz']):
                        damage[f'LSS{sstr}'] += damage[f'LSShft{s}{k}{x}a']
                    for ix, x in enumerate(['xy','z']):
                        damage[f'TowerBase{sstr}'] += damage[f'TwrBs{s}{k}{x}t']
                        if modopt['flags']['monopile'] and modopt['Level3']['flag']:
                            damage[f'MonopileBase{sstr}'] += damage[f'M1N1{s}{k}K{x}e']

            # Assemble damages
            outputs['damage_blade_root_sparU'] = np.max([damage[f'BladeRootSparU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_root_sparL'] = np.max([damage[f'BladeRootSparL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teU'] = np.max([damage[f'BladeMaxcTEU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teL'] = np.max([damage[f'BladeMaxcTEL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_lss'] = np.sqrt( damage['LSSAxial']**2 + damage['LSSShear']**2 )
            outputs['damage_tower_base'] = np.sqrt( damage['TowerBaseAxial']**2 + damage['TowerBaseShear']**2 )
            outputs['damage_monopile_base'] = np.sqrt( damage['MonopileBaseAxial']**2 + damage['MonopileBaseShear']**2 )

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

    def get_ac_axis(self, inputs):
        
        # Get the absolute offset between pitch axis (rotation center) and aerodynamic center
        ch_offset = inputs['chord'] * (inputs['ac'] - inputs['le_location'])
        # Rotate it by the twist using the AD15 coordinate system
        x , y = util.rotate(0., 0., 0., ch_offset, -np.deg2rad(inputs['theta']))
        # Apply offset to determine the AC axis
        BlCrvAC = inputs['ref_axis_blade'][:,0] + x
        BlSwpAC = inputs['ref_axis_blade'][:,1] + y
        
        return BlCrvAC, BlSwpAC
    

    def write_FAST(self, fst_vt, discrete_outputs):
        writer                   = InputWriter_OpenFAST()
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.execute()

    def writeCpsurfaces(self, inputs):

        modopt = self.options['modeling_options']
        FASTpref  = modopt['openfast']['FASTpref']
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

    def save_timeseries(self,chan_time):
        '''
        Save ALL the timeseries: each iteration and openfast run thereof
        '''

        # Make iteration directory
        save_dir = os.path.join(self.FAST_runDirectory,'iteration_'+str(self.of_inumber),'timeseries')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save each timeseries as a pickled dataframe
        for i_ts, timeseries in enumerate(chan_time):
            output = OpenFASTOutput.from_dict(timeseries, self.FAST_namingOut)
            output.df.to_pickle(os.path.join(save_dir,self.FAST_namingOut + '_' + str(i_ts) + '.p'))

    def save_iterations(self,summ_stats,DELs):
        '''
        Save summary stats, DELs of each iteration
        '''

        # Make iteration directory
        save_dir = os.path.join(self.FAST_runDirectory,'iteration_'+str(self.of_inumber))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save dataframes as pickles
        summ_stats.to_pickle(os.path.join(save_dir,'summary_stats.p'))
        DELs.to_pickle(os.path.join(save_dir,'DELs.p'))
