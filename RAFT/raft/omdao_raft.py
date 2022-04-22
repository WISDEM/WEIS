import openmdao.api as om
import raft
import numpy as np
import pickle, os
import copy
from itertools import compress

ndim = 3
ndof = 6
class RAFT_OMDAO(om.ExplicitComponent):
    """
    RAFT OpenMDAO Wrapper API

    """
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('turbine_options')
        self.options.declare('mooring_options')
        self.options.declare('member_options')
        self.options.declare('analysis_options')

    def setup(self):

        # unpack options
        modeling_opt = self.options['modeling_options']
        analysis_options = self.options['analysis_options']
        
        nfreq = modeling_opt['nfreq']
        n_cases = modeling_opt['n_cases']

        turbine_opt = self.options['turbine_options']
        turbine_npts = turbine_opt['npts']
        n_gain = turbine_opt['PC_GS_n']
        n_span = turbine_opt['n_span']
        n_aoa = turbine_opt['n_aoa']
        n_Re = turbine_opt['n_Re']
        n_tab = turbine_opt['n_tab']
        n_pc = turbine_opt['n_pc']
        n_af = turbine_opt['n_af']
        af_used_names = turbine_opt['af_used_names']
        n_af_span = len(af_used_names)
        
        members_opt = self.options['member_options']
        nmembers = members_opt['nmembers']
        member_npts = members_opt['npts']
        member_npts_lfill = members_opt['npts_lfill']
        member_npts_rho_fill = members_opt['npts_rho_fill']
        member_ncaps = members_opt['ncaps']
        member_nreps = members_opt['nreps']
        member_shape = members_opt['shape']
        member_scalar_t = members_opt['scalar_thicknesses']
        member_scalar_d = members_opt['scalar_diameters']
        member_scalar_coeff = members_opt['scalar_coefficients']
        n_ballast_type = members_opt['n_ballast_type']

        mooring_opt = self.options['mooring_options']
        nlines = mooring_opt['nlines']
        nline_types = mooring_opt['nline_types']
        nconnections = mooring_opt['nconnections']

        # turbine inputs
        self.add_input('turbine_mRNA', val=0.0, units='kg', desc='RNA mass')
        self.add_input('turbine_IxRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local x axis')
        self.add_input('turbine_IrRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local y or z axes')
        self.add_input('turbine_xCG_RNA', val=0.0, units='m', desc='x location of RNA center of mass')
        
        self.add_input('turbine_hHub', val=0.0, units='m', desc='Hub height above water line')
        self.add_input('turbine_overhang', val=0.0, units='m', desc='Overhang of rotor apex from tower centerline (parallel to ground)')
        self.add_input('turbine_Fthrust', val=0.0, units='N', desc='Temporary thrust force to use')
        self.add_input('turbine_yaw_stiffness', val=0.0, units='N*m', desc='Additional yaw stiffness to apply if not modeling crowfoot in the mooring system')
        # tower inputs
        self.add_input('turbine_tower_rA', val=np.zeros(ndim), units='m', desc='End A coordinates')
        self.add_input('turbine_tower_rB', val=np.zeros(ndim), units='m', desc='End B coordinates')
        self.add_input('turbine_tower_gamma', val=0.0, units='deg', desc='Twist angle about z-axis')
        self.add_input('turbine_tower_stations', val=np.zeros(turbine_npts), desc='Location of stations along axis, will be normalized along rA to rB')
        if turbine_opt['scalar_diameters']:
            self.add_input('turbine_tower_d', val=0.0, units='m', desc='Diameters if circular or side lengths if rectangular')
        else:
            if turbine_opt['shape'] == 'circ' or 'square':
                self.add_input('turbine_tower_d', val=np.zeros(turbine_npts), units='m', desc='Diameters if circular or side lengths if rectangular')
            elif turbine_opt['shape'] == 'rect':
                self.add_input('turbine_tower_d', val=np.zeros(2 * turbine_npts), units='m', desc='Diameters if circular or side lengths if rectangular')

        if turbine_opt['scalar_thicknesses']:
            self.add_input('turbine_tower_t', val=0.0, units='m', desc='Wall thicknesses at station locations')
        else:
            self.add_input('turbine_tower_t', val=np.zeros(turbine_npts), units='m', desc='Wall thicknesses at station locations')

        if turbine_opt['scalar_coefficients']:
            self.add_input('turbine_tower_Cd', val=0.0, desc='Transverse drag coefficient')
            self.add_input('turbine_tower_Ca', val=0.0, desc='Transverse added mass coefficient')
            self.add_input('turbine_tower_CdEnd', val=0.0, desc='End axial drag coefficient')
            self.add_input('turbine_tower_CaEnd', val=0.0, desc='End axial added mass coefficient')
        else:
            self.add_input('turbine_tower_Cd', val=np.zeros(turbine_npts), desc='Transverse drag coefficient')
            self.add_input('turbine_tower_Ca', val=np.zeros(turbine_npts), desc='Transverse added mass coefficient')
            self.add_input('turbine_tower_CdEnd', val=np.zeros(turbine_npts), desc='End axial drag coefficient')
            self.add_input('turbine_tower_CaEnd', val=np.zeros(turbine_npts), desc='End axial added mass coefficient')
        self.add_input('turbine_tower_rho_shell', val=0.0, units='kg/m**3', desc='Material density')

        # control inputs
        self.add_input('rotor_PC_GS_angles',     val=np.zeros(n_gain),   units='rad',        desc='Gain-schedule table: pitch angles')
        self.add_input('rotor_PC_GS_Kp',         val=np.zeros(n_gain),   units='s',          desc='Gain-schedule table: pitch controller kp gains')
        self.add_input('rotor_PC_GS_Ki',         val=np.zeros(n_gain),                       desc='Gain-schedule table: pitch controller ki gains')
        self.add_input('Fl_Kp',                  val=0.0,                        desc='Floating feedback gain')
        self.add_input('rotor_inertia',          val=0.0,    units='kg*m**2',    desc='Rotor inertia')
        self.add_input('rotor_TC_VS_Kp',         val=0.0,   units='s',          desc='Gain-schedule table: torque controller kp gains')
        self.add_input('rotor_TC_VS_Ki',         val=0.0,                       desc='Gain-schedule table: torque controller ki gains')
        # Blade and rotor inputs
        self.add_discrete_input('nBlades', val=3, desc='number of blades')
        self.add_input('tilt', val=0.0, units='deg', desc='shaft upward tilt angle relative to horizontal')
        self.add_input('precone', val=0.0, units='deg', desc='hub precone angle')
        self.add_input('wind_reference_height', val=0.0, units='m', desc='hub height used for power-law wind profile. U = Uref*(z/hubHt)**shearExp')
        self.add_input('hub_radius', val=0.0, units='m', desc='radius of hub')
        self.add_input("gear_ratio", val=1.0, desc="Total gear ratio of drivetrain (use 1.0 for direct)")
        self.add_input('blade_r', val=np.zeros(n_span), units='m', desc='locations defining the blade along z-axis of blade coordinate system')
        self.add_input('blade_chord', val=np.zeros(n_span), units='m', desc='corresponding chord length at each section')
        self.add_input('blade_theta', val=np.zeros(n_span), units='deg', desc='corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---positive twist decreases angle of attack')
        self.add_input('blade_Rtip', val=0.0, units='m', desc='radius of blade tip')
        self.add_input('blade_precurve', val=np.zeros(n_span), units='m', desc='location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`')
        self.add_input('blade_precurveTip', val=0.0, units='m', desc='location of blade pitch axis in x-direction at the tip (analogous to Rtip)')
        self.add_input('blade_presweep', val=np.zeros(n_span), units='m', desc='location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`')
        self.add_input('blade_presweepTip', val=0.0, units='m', desc='location of blade pitch axis in y-direction at the tip (analogous to Rtip)')
        # Airfoils
        self.add_discrete_input("airfoils_name", val=n_af * [""], desc="1D array of names of airfoils.")
        self.add_input("airfoils_position", val=np.zeros(n_af_span), desc="1D array of the non dimensional positions of the airfoils af_used defined along blade span.")
        self.add_input("airfoils_r_thick", val=np.zeros(n_af), desc="1D array of the relative thicknesses of each airfoil.")
        self.add_input("airfoils_aoa", val=np.zeros(n_aoa), units="rad", desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.")
        self.add_input("airfoils_cl", val=np.zeros((n_af, n_aoa, n_Re, n_tab)), desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.")
        self.add_input("airfoils_cd", val=np.zeros((n_af, n_aoa, n_Re, n_tab)), desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.")
        self.add_input("airfoils_cm", val=np.zeros((n_af, n_aoa, n_Re, n_tab)), desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.")
        self.add_input("rotor_powercurve_v", val=np.zeros(n_pc), units="m/s", desc="wind vector")
        self.add_input("rotor_powercurve_omega_rpm", val=np.zeros(n_pc), units="rpm", desc="rotor rotational speed")
        self.add_input("rotor_powercurve_pitch", val=np.zeros(n_pc), units="deg", desc="rotor pitch schedule")
        self.add_input("rho_air", val=1.225, units="kg/m**3", desc="Density of air")
        self.add_input("rho_water", val=1025.0, units="kg/m**3", desc="Density of sea water")
        self.add_input("mu_air", val=1.81e-5, units="kg/(m*s)", desc="Dynamic viscosity of air")
        self.add_input("shear_exp", val=0.2, desc="Shear exponent of the wind.")
        self.add_input('rated_rotor_speed', val=0.0, units='rpm',  desc='rotor rotation speed at rated')

        # DLCs
        self.add_discrete_input('raft_dlcs', val=[[]]*n_cases, desc='DLC case table for RAFT with each row a new case and headings described by the keys')
        self.add_discrete_input('raft_dlcs_keys', val=['wind_speed', 'wind_heading', 'turbulence',
                                                       'turbine_status', 'yaw_misalign', 'wave_spectrum',
                                                       'wave_period', 'wave_height', 'wave_heading'],
                                desc='DLC case table column headings')
        
        # member inputs
        for i in range(1, nmembers + 1):

            mnpts = member_npts[i - 1]
            mnpts_lfill = member_npts_lfill[i - 1]
            mncaps = member_ncaps[i - 1]
            mnreps = member_nreps[i - 1]
            mshape = member_shape[i - 1]
            scalar_t = member_scalar_t[i - 1]
            scalar_d = member_scalar_d[i - 1]
            scalar_coeff = member_scalar_coeff[i - 1]
            m_name = f'platform_member{i}_'

            self.add_input(m_name+'heading', val=np.zeros(mnreps), units='deg', desc='Heading rotation of column about z axis (for repeated members)')
            self.add_input(m_name+'rA', val=np.zeros(ndim), units='m', desc='End A coordinates')
            self.add_input(m_name+'rB', val=np.zeros(ndim), units='m', desc='End B coordinates')
            self.add_input(m_name+'s_ghostA', val=0.0, desc='Non-dimensional location where overlap point begins at end A')
            self.add_input(m_name+'s_ghostB', val=1.0, desc='Non-dimensional location where overlap point begins at end B')
            self.add_input(m_name+'gamma', val=0.0, units='deg', desc='Twist angle about the member z axis')
            # ADD THIS AS AN OPTION IN WEIS
            self.add_discrete_input(m_name+'potMod', val=False, desc='Whether to model the member with potential flow')
            self.add_input(m_name+'stations', val=np.zeros(mnpts), desc='Location of stations along axis, will be normalized from end A to B')
            # updated version to better handle 'diameters' between circular and rectangular members
            if mshape == 'circ' or mshape == 'square':
                if scalar_d:
                    self.add_input(m_name+'d', val=0.0, units='m', desc='Constant diameter of the whole member')
                else:
                    self.add_input(m_name+'d', val=np.zeros(mnpts), units='m', desc='Diameters at each station along the member')
            elif mshape == 'rect':
                if scalar_d:
                    self.add_input(m_name+'d', val=[0.0, 0.0], units='m', desc='Constant side lengths of the whole member')
                else:
                    self.add_input(m_name+'d', val=np.zeros([mnpts,2]), units='m', desc='Side lengths at each station along the member')
            ''' original version of handling diameters
            if scalar_d:
                self.add_input(m_name+'d', val=0.0, units='m', desc='Diameters if circular, side lengths if rectangular')
            else:
                if mshape == 'circ' or 'square':
                    self.add_input(m_name+'d', val=np.zeros(mnpts), units='m', desc='Diameters if circular, side lengths if rectangular')
                elif mshape == 'rect':
                    self.add_input(m_name+'d', val=np.zeros(2 * mnpts), units='m', desc='Diameters if circular, side lengths if rectangular')
            '''
            if scalar_t:
                self.add_input(m_name+'t', val=0.0, units='m', desc='Wall thicknesses')
            else:
                self.add_input(m_name+'t', val=np.zeros(mnpts), units='m', desc='Wall thicknesses')
            if scalar_coeff:
                self.add_input(m_name+'Cd', val=0.0, desc='Transverse drag coefficient')
                self.add_input(m_name+'Ca', val=0.0, desc='Transverse added mass coefficient')
                self.add_input(m_name+'CdEnd', val=0.0, desc='End axial drag coefficient')
                self.add_input(m_name+'CaEnd', val=0.0, desc='End axial added mass coefficient')
            else:
                self.add_input(m_name+'Cd', val=np.zeros(mnpts), desc='Transverse drag coefficient')
                self.add_input(m_name+'Ca', val=np.zeros(mnpts), desc='Transverse added mass coefficient')
                self.add_input(m_name+'CdEnd', val=np.zeros(mnpts), desc='End axial drag coefficient')
                self.add_input(m_name+'CaEnd', val=np.zeros(mnpts), desc='End axial added mass coefficient')
            self.add_input(m_name+'rho_shell', val=0.0, units='kg/m**3', desc='Material density')
            # optional
            self.add_input(m_name+'l_fill', val=np.zeros(mnpts_lfill), units='m', desc='Fill heights of ballast in each section')
            self.add_input(m_name+'rho_fill', val=np.zeros(mnpts_lfill), units='kg/m**3', desc='Material density of ballast in each section')
            self.add_input(m_name+'cap_stations', val=np.zeros(mncaps), desc='Location along member of any inner structures (same scaling as stations')
            self.add_input(m_name+'cap_t', val=np.zeros(mncaps), units='m', desc='Thickness of any internal structures')
            self.add_input(m_name+'cap_d_in', val=np.zeros(mncaps), units='m', desc='Inner diameter of internal structures')
            self.add_input(m_name+'ring_spacing', val=0.0, desc='Spacing of internal structures placed based on spacing.  Dimension is same as used in stations')
            self.add_input(m_name+'ring_t', val=0.0, units='m', desc='Effective thickness of any internal structures placed based on spacing')
            self.add_input(m_name+'ring_h', val=0.0, units='m', desc='Effective web height of internal structures placed based on spacing')
            
        # mooring inputs
        self.add_input('mooring_water_depth', val=0.0, units='m', desc='Uniform water depth')
        # connection points
        for i in range(1, nconnections + 1):
            pt_name = f'mooring_point{i}_'
            self.add_discrete_input(pt_name+'name', val=f'line{i}', desc='Mooring point identifier')
            self.add_discrete_input(pt_name+'type', val='fixed', desc='Mooring connection type')
            self.add_input(pt_name+'location', val=np.zeros(ndim), units='m', desc='Coordinates of mooring connection')
        # lines
        for i in range(1, nlines + 1):
            pt_name = f'mooring_line{i}_'
            self.add_discrete_input(pt_name+'endA', val='default', desc='End A coordinates')
            self.add_discrete_input(pt_name+'endB', val='default', desc='End B coordinates')
            self.add_discrete_input(pt_name+'type', val='mooring_line_type1', desc='Mooring line type')
            self.add_input(pt_name+'length', val=0.0, units='m', desc='Length of line')
        # line types
        for i in range(1, nline_types + 1):
            lt_name = f'mooring_line_type{i}_'
            self.add_discrete_input(lt_name+'name', val='default', desc='Name of line type')
            self.add_input(lt_name+'diameter', val=0.0, units='m', desc='Diameter of mooring line type')
            self.add_input(lt_name+'mass_density', val=0.0, units='kg/m**3', desc='Mass density of line type')
            self.add_input(lt_name+'stiffness', val=0.0, desc='Stiffness of line type')
            self.add_input(lt_name+'breaking_load', val=0.0, desc='Breaking load of line type')
            self.add_input(lt_name+'cost', val=0.0, units='USD', desc='Cost of mooring line type')
            self.add_input(lt_name+'transverse_added_mass', val=0.0, desc='Transverse added mass')
            self.add_input(lt_name+'tangential_added_mass', val=0.0, desc='Tangential added mass')
            self.add_input(lt_name+'transverse_drag', val=0.0, desc='Transverse drag')
            self.add_input(lt_name+'tangential_drag', val=0.0, desc='Tangential drag')

        # outputs
        # properties
        self.add_output('properties_tower mass', units='kg', desc='Tower mass')
        self.add_output('properties_tower CG', val=np.zeros(ndim), units='m', desc='Tower center of gravity')
        self.add_output('properties_substructure mass', val=0.0, units='kg', desc='Substructure mass')
        self.add_output('properties_substructure CG', val=np.zeros(ndim), units='m', desc='Substructure center of gravity')
        self.add_output('properties_shell mass', val=0.0, units='kg', desc='Shell mass')
        self.add_output('properties_ballast mass', val=np.zeros(n_ballast_type), units='m', desc='Ballast mass')
        self.add_output('properties_ballast densities', val=np.zeros(n_ballast_type), units='kg', desc='Ballast densities')
        self.add_output('properties_total mass', val=0.0, units='kg', desc='Total mass of system')
        self.add_output('properties_total CG', val=np.zeros(ndim), units='m', desc='Total system center of gravity')
        self.add_output('properties_roll inertia at subCG', val=np.zeros(ndim), units='kg*m**2', desc='Roll inertia sub CG')
        self.add_output('properties_pitch inertia at subCG', val=np.zeros(ndim), units='kg*m**2', desc='Pitch inertia sub CG')
        self.add_output('properties_yaw inertia at subCG', val=np.zeros(ndim), units='kg*m**2', desc='Yaw inertia sub CG')
        self.add_output('properties_Buoyancy (pgV)', val=0.0, units='N', desc='Buoyancy (pgV)')
        self.add_output('properties_Center of Buoyancy', val=np.zeros(ndim), units='m', desc='Center of buoyancy')
        self.add_output('properties_C stiffness matrix', val=np.zeros((ndof,ndof)), units='Pa', desc='C stiffness matrix')
        self.add_output('properties_F_lines0', val=np.zeros(nconnections), units='N', desc='Mean mooring force')
        self.add_output('properties_C_lines0', val=np.zeros((ndof,ndof)), units='Pa', desc='Mooring stiffness')
        self.add_output('properties_M support structure', val=np.zeros((ndof,ndof)), units='kg', desc='Mass matrix for platform')
        self.add_output('properties_A support structure', val=np.zeros((ndof,ndof)), desc='Added mass matrix for platform')
        self.add_output('properties_C support structure', val=np.zeros((ndof,ndof)), units='Pa', desc='Stiffness matrix for platform')
        # response
        self.add_output('response_frequencies', val=np.zeros(nfreq), units='Hz', desc='Response frequencies')
        self.add_output('response_wave elevation', val=np.zeros(nfreq), units='m', desc='Wave elevation')
        self.add_output('response_surge RAO', val=np.zeros(nfreq), units='m', desc='Surge RAO')
        self.add_output('response_sway RAO', val=np.zeros(nfreq), units='m', desc='Sway RAO')
        self.add_output('response_heave RAO', val=np.zeros(nfreq), units='m', desc='Heave RAO')
        self.add_output('response_pitch RAO', val=np.zeros(nfreq), units='rad', desc='Pitch RAO')
        self.add_output('response_roll RAO', val=np.zeros(nfreq), units='rad', desc='Roll RAO')
        self.add_output('response_yaw RAO', val=np.zeros(nfreq), units='rad', desc='Yaw RAO')
        self.add_output('response_nacelle acceleration', val=np.zeros(nfreq), units='m/s**2', desc='Nacelle acceleration')
        # case specific, note: only DLCs supported in RAFT will have non-zero outputs
        names = ['surge','sway','heave','roll','pitch','yaw','AxRNA','Mbase','omega','torque','power','bPitch','Tmoor']
        stats = ['avg','std','max','PSD','DEL']
        for n in names:
            for s in stats:
                if s == 'DEL' and not n in ['Tmoor','Mbase']: continue
                iout = f'{n}_{s}'
                
                if n == 'Tmoor':
                    myval = np.zeros((n_cases, 2*nlines)) if s not in ['PSD'] else np.zeros((n_cases, 2*nlines, nfreq))
                else:
                    myval = np.zeros(n_cases) if s not in ['PSD'] else np.zeros((n_cases, nfreq))
                
                if n in ['surge','sway','heave']:
                    myunit = 'm'
                elif n in ['roll','pitch','yaw']:
                    myunit = 'rad'
                elif n in ['AxRNA']:
                    myunit = 'm/s/s'
                elif n in ['Mbase']:
                    myunit = 'N*m'
                    
                self.add_output('stats_'+iout, val=myval, units=myunit)
        # Other case outputs
        self.add_output('stats_wind_PSD', val=np.zeros((n_cases,nfreq)), desc='Power spectral density of wind input')
        self.add_output('stats_wave_PSD', val=np.zeros((n_cases,nfreq)), desc='Power spectral density of wave input')
        
        # Aggregate outputs
        self.add_output('Max_Offset', val = 0, desc = 'Maximum distance in surge/sway direction', units = 'm') 
        self.add_output('heave_avg', val = 0, desc = 'Average heave over all cases', units = 'm') 
        self.add_output('Max_PtfmPitch', val = 0, desc = 'Maximum platform pitch over all cases', units = 'deg') 
        self.add_output('Std_PtfmPitch', val = 0, desc = 'Average platform pitch std. over all cases', units = 'deg') 
        self.add_output('max_nacelle_Ax', val = 0, desc = 'Maximum nacelle accelleration over all cases', units = 'm/s**2') 
        self.add_output('rotor_overspeed', val = 0, desc = 'Fraction above rated rotor speed') 
        self.add_output('max_tower_base', val = 0, desc = 'Maximum tower base moment over all cases', units = 'N*m') 

        # Combined outputs for OpenFAST
        self.add_output("platform_total_center_of_mass", np.zeros(3), units="m")
        self.add_output("platform_displacement", 0.0, desc='Volumetric platform displacement', units='m**3')
        self.add_output("platform_mass", 0.0, units="kg")
        self.add_output("platform_I_total", np.zeros(6), units="kg*m**2")
        
        self.i_design = 0
        if modeling_opt['save_designs']:
            if not os.path.exists(os.path.join(analysis_options['general']['folder_output'],'raft_designs')):
                os.makedirs(os.path.join(analysis_options['general']['folder_output'],'raft_designs'))
                
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        turbine_opt = self.options['turbine_options']
        mooring_opt = self.options['mooring_options']
        members_opt = self.options['member_options']
        modeling_opt = self.options['modeling_options']
        analysis_options = self.options['analysis_options']

        #turbine_npts = turbine_opt['npts']

        nmembers = members_opt['nmembers']
        member_npts = members_opt['npts']
        member_npts_lfill = members_opt['npts_lfill']
        #member_npts_rho_fill = members_opt['npts_rho_fill']
        member_ncaps = members_opt['ncaps']
        member_nreps = members_opt['nreps']
        member_shapes = members_opt['shape']
        member_scalar_t = members_opt['scalar_thicknesses']
        member_scalar_d = members_opt['scalar_diameters']
        member_scalar_coeff = members_opt['scalar_coefficients']
        
        nlines = mooring_opt['nlines']
        nline_types = mooring_opt['nline_types']
        nconnections = mooring_opt['nconnections']

        # set up design
        design = {}
        design['type'] = ['input dictionary for RAFT']
        design['name'] = ['spiderfloat']
        design['comments'] = ['none']
        
        design['settings'] = {}
        design['settings']['XiStart'] = float(modeling_opt['xi_start'])
        design['settings']['min_freq'] = float(modeling_opt['min_freq'])
        design['settings']['max_freq'] = float(modeling_opt['max_freq'])
        design['settings']['nIter'] = int(modeling_opt['nIter'])

        # Environment layer data
        design['site'] = {}
        design['site']['water_depth'] = float(inputs['mooring_water_depth'])
        design['site']['rho_air'] = float(inputs['rho_air'])
        design['site']['rho_water'] = float(inputs['rho_water'])
        design['site']['mu_air' ] = float(inputs['mu_air'])
        design['site']['shearExp'] = float(inputs['shear_exp'])
        
        # RNA properties
        design['turbine'] = {}
        design['turbine']['mRNA']          = float(inputs['turbine_mRNA'])
        design['turbine']['IxRNA']         = float(inputs['turbine_IxRNA'])
        design['turbine']['IrRNA']         = float(inputs['turbine_IrRNA'])
        design['turbine']['xCG_RNA']       = float(inputs['turbine_xCG_RNA'])
        design['turbine']['hHub']          = float(inputs['turbine_hHub'])
        design['turbine']['overhang']      = float(inputs['turbine_overhang'])
        design['turbine']['Fthrust']       = float(inputs['turbine_Fthrust'])
        design['turbine']['yaw_stiffness'] = float(inputs['turbine_yaw_stiffness'])
        design['turbine']['gear_ratio']    = float(inputs['gear_ratio'])

        # Tower
        design['turbine']['tower'] = {}
        design['turbine']['tower']['name'] = 'tower'
        design['turbine']['tower']['type'] = 1
        design['turbine']['tower']['rA'] = inputs['turbine_tower_rA']
        design['turbine']['tower']['rB'] = inputs['turbine_tower_rB']
        design['turbine']['tower']['shape'] = turbine_opt['shape']
        design['turbine']['tower']['gamma'] = inputs['turbine_tower_gamma']
        design['turbine']['tower']['stations'] = inputs['turbine_tower_stations']
        if turbine_opt['scalar_diameters']:
            design['turbine']['tower']['d'] = float(inputs['turbine_tower_d'])
        else:
            design['turbine']['tower']['d'] = inputs['turbine_tower_d']
        if turbine_opt['scalar_thicknesses']:
            design['turbine']['tower']['t'] = float(inputs['turbine_tower_t'])
        else:
            design['turbine']['tower']['t'] = inputs['turbine_tower_t']
        if turbine_opt['scalar_coefficients']:
            design['turbine']['tower']['Cd'] = float(inputs['turbine_tower_Cd'])
            design['turbine']['tower']['Ca'] = float(inputs['turbine_tower_Ca'])
            design['turbine']['tower']['CdEnd'] = float(inputs['turbine_tower_CdEnd'])
            design['turbine']['tower']['CaEnd'] = float(inputs['turbine_tower_CaEnd'])
        else:
            design['turbine']['tower']['Cd'] = inputs['turbine_tower_Cd']
            design['turbine']['tower']['Ca'] = inputs['turbine_tower_Ca']
            design['turbine']['tower']['CdEnd'] = inputs['turbine_tower_CdEnd']
            design['turbine']['tower']['CaEnd'] = inputs['turbine_tower_CaEnd']
        design['turbine']['tower']['rho_shell'] = float(inputs['turbine_tower_rho_shell'])

        # Blades and rotors
        design['turbine']['nBlades']    = int(discrete_inputs['nBlades'])
        design['turbine']['shaft_tilt'] = float(inputs['tilt'])
        design['turbine']['precone']    = float(inputs['precone'])
        design['turbine']['Zhub']       = float(inputs['wind_reference_height'])
        design['turbine']['Rhub']       = float(inputs['hub_radius'])
        design['turbine']['I_drivetrain']    = float(inputs['rotor_inertia'])

        design['turbine']['blade'] = {}
        design['turbine']['blade']['geometry']    = np.c_[inputs['blade_r'],
                                                          inputs['blade_chord'],
                                                          inputs['blade_theta'],
                                                          inputs['blade_precurve'],
                                                          inputs['blade_presweep']]
        design['turbine']['blade']['Rtip']        = float(inputs['blade_Rtip'])
        design['turbine']['blade']['precurveTip'] = float(inputs['blade_precurveTip'])
        design['turbine']['blade']['presweepTip'] = float(inputs['blade_presweepTip'])
        design['turbine']['blade']['airfoils']    = list(zip(inputs['airfoils_position'], turbine_opt['af_used_names']))
        # airfoils data
        n_af = turbine_opt['n_af']
        design['turbine']['airfoils'] = [dict() for m in range(n_af)] #Note: doesn't work [{}]*n_af
        for i in range(n_af):
            design['turbine']['airfoils'][i]['name'] = discrete_inputs['airfoils_name'][i]
            design['turbine']['airfoils'][i]['relative_thickness'] = inputs['airfoils_r_thick'][i]
            design['turbine']['airfoils'][i]['data'] = np.c_[inputs['airfoils_aoa'] * raft.helpers.rad2deg(1),
                                                             inputs['airfoils_cl'][i,:,0,0],
                                                             inputs['airfoils_cd'][i,:,0,0],
                                                             inputs['airfoils_cm'][i,:,0,0]]

        # Control
        design['turbine']['pitch_control'] = {}
        design['turbine']['pitch_control']['GS_Angles']    = inputs['rotor_PC_GS_angles']
        design['turbine']['pitch_control']['GS_Kp']        = inputs['rotor_PC_GS_Kp']
        design['turbine']['pitch_control']['GS_Ki']        = inputs['rotor_PC_GS_Ki']
        design['turbine']['pitch_control']['Fl_Kp']        = float(inputs['Fl_Kp'])
        design['turbine']['torque_control'] = {}
        design['turbine']['torque_control']['VS_KP'] = float(inputs['rotor_TC_VS_Kp'])
        design['turbine']['torque_control']['VS_KI'] = float(inputs['rotor_TC_VS_Ki'])

        # Operations
        design['turbine']['wt_ops'] = {}
        design['turbine']['wt_ops']['v'] = inputs['rotor_powercurve_v']
        design['turbine']['wt_ops']['omega_op'] = inputs['rotor_powercurve_omega_rpm']
        design['turbine']['wt_ops']['pitch_op'] = inputs['rotor_powercurve_pitch']
        
        # Platform members
        design['platform'] = {}
        design['platform']['potModMaster'] = int(modeling_opt['potential_model_override'])
        design['platform']['dlsMax'] = float(modeling_opt['dls_max'])
        # lowest BEM freq needs to be just below RAFT min_freq because of interpolation in RAFT
        if float(modeling_opt['min_freq_BEM']) >= modeling_opt['min_freq']:
            modeling_opt['min_freq_BEM'] = modeling_opt['min_freq'] - 1e-7
        design['platform']['min_freq_BEM'] = float(modeling_opt['min_freq_BEM'])
        design['platform']['members'] = [dict() for m in range(nmembers)] #Note: doesn't work [{}]*nmembers
        for i in range(nmembers):
            m_name = f'platform_member{i+1}_'
            m_shape = member_shapes[i]
            mnpts_lfill = member_npts_lfill[i]
            mncaps = member_ncaps[i]
            mnreps = member_nreps[i]
            #mnpts = member_npts[i]

            # Set stations and end points that account for intersections/ghost segments
            rA_0 = inputs[m_name+'rA']
            rB_0 = inputs[m_name+'rB']
            s_ghostA = inputs[m_name+'s_ghostA']
            s_ghostB = inputs[m_name+'s_ghostB']
            s_0 = inputs[m_name+'stations']
            idx = np.logical_and(s_0>=s_ghostA, s_0<=s_ghostB)
            s_grid = np.unique(np.r_[s_ghostA, s_0[idx], s_ghostB])
            mnpts = len(idx)
            rA = rA_0 + s_ghostA*(rB_0-rA_0)
            rB = rA_0 + s_ghostB*(rB_0-rA_0)
            design['platform']['members'][i]['name'] = m_name
            design['platform']['members'][i]['type'] = i + 2
            design['platform']['members'][i]['rA'] = rA
            design['platform']['members'][i]['rB'] = rB
            design['platform']['members'][i]['shape'] = m_shape
            design['platform']['members'][i]['gamma'] = float(inputs[m_name+'gamma'])
            design['platform']['members'][i]['potMod'] = discrete_inputs[m_name+'potMod']
            design['platform']['members'][i]['stations'] = s_grid
            
            # updated version to better handle 'diameters' between circular and rectangular members
            if m_shape == 'circ' or m_shape == 'square':
                if member_scalar_d[i]:
                    design['platform']['members'][i]['d'] = [float(inputs[m_name+'d'])]*mnpts
                else:
                    design['platform']['members'][i]['d'] = np.interp(s_grid, s_0, inputs[m_name+'d'])
            elif m_shape == 'rect':
                if member_scalar_d[i]:
                    design['platform']['members'][i]['d'] = [inputs[m_name+'d']]*mnpts
                else:
                    design['platform']['members'][i]['d'] = np.interp(s_grid, s_0, inputs[m_name+'d'])
            ''' original version of handling diameters
            if member_scalar_d[i]:
                design['platform']['members'][i]['d'] = float(inputs[m_name+'d'])
            else:
                design['platform']['members'][i]['d'] = np.interp(s_grid, s_0, inputs[m_name+'d'])
            '''
            if member_scalar_t[i]:
                design['platform']['members'][i]['t'] = float(inputs[m_name+'t'])
            else:
                design['platform']['members'][i]['t'] = np.interp(s_grid, s_0, inputs[m_name+'t'])
            if member_scalar_coeff[i]:
                design['platform']['members'][i]['Cd'] = float(inputs[m_name+'Cd'])
                design['platform']['members'][i]['Ca'] = float(inputs[m_name+'Ca'])
                design['platform']['members'][i]['CdEnd'] = float(inputs[m_name+'CdEnd'])
                design['platform']['members'][i]['CaEnd'] = float(inputs[m_name+'CaEnd'])
            else:
                design['platform']['members'][i]['Cd'] = np.interp(s_grid, s_0, inputs[m_name+'Cd'])
                design['platform']['members'][i]['Ca'] = np.interp(s_grid, s_0, inputs[m_name+'Ca'])
                design['platform']['members'][i]['CdEnd'] = np.interp(s_grid, s_0, inputs[m_name+'CdEnd'])
                design['platform']['members'][i]['CaEnd'] = np.interp(s_grid, s_0, inputs[m_name+'CaEnd'])
            design['platform']['members'][i]['rho_shell'] = float(inputs[m_name+'rho_shell'])
            if mnreps > 0:
                design['platform']['members'][i]['heading'] = inputs[m_name+'heading']
            if mnpts_lfill > 0:
                design['platform']['members'][i]['l_fill'] = inputs[m_name+'l_fill']
                design['platform']['members'][i]['rho_fill'] = inputs[m_name+'rho_fill']
            if ( (mncaps > 0) or (inputs[m_name+'ring_spacing'] > 0) ):
                # Member discretization
                s_height = s_grid[-1] - s_grid[0]
                # Get locations of internal structures based on spacing
                ring_spacing = inputs[m_name+'ring_spacing']
                n_stiff = 0 if ring_spacing == 0.0 else int(np.floor(s_height / ring_spacing))
                s_ring = (np.arange(1, n_stiff + 0.1) - 0.5) * (ring_spacing / s_height)
                d_ring = np.interp(s_ring, s_grid, design['platform']['members'][i]['d'])
                # Combine internal structures based on spacing and defined positions
                s_cap_0 = inputs[m_name+'cap_stations']
                idx_cap = np.logical_and(s_cap_0>=s_ghostA, s_cap_0<=s_ghostB)
                s_cap, isort = np.unique(np.r_[s_ghostA, s_cap_0[idx_cap], s_ghostB], return_index=True)
                t_cap = np.r_[inputs[m_name+'cap_t'][0], inputs[m_name+'cap_t'][idx_cap], inputs[m_name+'cap_t'][-1]][isort]
                di_cap = np.zeros(s_cap.shape)
                # No end caps at joints
                if s_ghostA > 0.0:
                    s_cap = s_cap[1:]
                    t_cap = t_cap[1:]
                    di_cap = di_cap[1:]
                if s_ghostB < 1.0:
                    s_cap = s_cap[:-1]
                    t_cap = t_cap[:-1]
                    di_cap = di_cap[:-1]
                # Combine with ring stiffeners
                s_cap = np.r_[s_ring, s_cap]
                t_cap = np.r_[inputs[m_name+'ring_t']*np.ones(n_stiff), t_cap]
                di_cap = np.r_[d_ring-2*inputs[m_name+'ring_h'], di_cap]
                # Store vectors in sorted order
                if len(s_cap) > 0:
                    isort = np.argsort(s_cap)
                    design['platform']['members'][i]['cap_stations'] = s_cap[isort]
                    design['platform']['members'][i]['cap_t'] = t_cap[isort]
                    design['platform']['members'][i]['cap_d_in'] = di_cap[isort]

        design['mooring'] = {}
        design['mooring']['water_depth'] = float(inputs['mooring_water_depth'])
        design['mooring']['points'] = [dict() for m in range(nconnections)] #Note: doesn't work [{}]*nconnections
        for i in range(0, nconnections):
            pt_name = f'mooring_point{i+1}_'
            design['mooring']['points'][i]['name'] = discrete_inputs[pt_name+'name']
            design['mooring']['points'][i]['type'] = discrete_inputs[pt_name+'type']
            design['mooring']['points'][i]['location'] = inputs[pt_name+'location']
            if design['mooring']['points'][i]['type'].lower() == 'fixed':
                design['mooring']['points'][i]['anchor_type'] = 'drag_embedment' #discrete_inputs[pt_name+'type']

        design['mooring']['lines'] = [dict() for m in range(nlines)] #Note: doesn't work [{}]*nlines
        for i in range(0, nlines):
            ml_name = f'mooring_line{i+1}_'
            design['mooring']['lines'][i]['name'] = f'line{i+1}'
            design['mooring']['lines'][i]['endA'] = discrete_inputs[ml_name+'endA']
            design['mooring']['lines'][i]['endB'] = discrete_inputs[ml_name+'endB']
            design['mooring']['lines'][i]['type'] = discrete_inputs[ml_name+'type']
            design['mooring']['lines'][i]['length'] = inputs[ml_name+'length']
        design['mooring']['line_types'] = [dict() for m in range(nline_types)] #Note: doesn't work [{}]*nline_types
        for i in range(0, nline_types):
            lt_name = f'mooring_line_type{i+1}_'
            design['mooring']['line_types'][i]['name'] = discrete_inputs[lt_name+'name']
            design['mooring']['line_types'][i]['diameter'] = float(inputs[lt_name+'diameter'])
            design['mooring']['line_types'][i]['mass_density'] = float(inputs[lt_name+'mass_density'])
            design['mooring']['line_types'][i]['stiffness'] = float(inputs[lt_name+'stiffness'])
            design['mooring']['line_types'][i]['breaking_load'] = float(inputs[lt_name+'breaking_load'])
            design['mooring']['line_types'][i]['cost'] = float(inputs[lt_name+'cost'])
            design['mooring']['line_types'][i]['transverse_added_mass'] = float(inputs[lt_name+'transverse_added_mass'])
            design['mooring']['line_types'][i]['tangential_added_mass'] = float(inputs[lt_name+'tangential_added_mass'])
            design['mooring']['line_types'][i]['transverse_drag'] = float(inputs[lt_name+'transverse_drag'])
            design['mooring']['line_types'][i]['tangential_drag'] = float(inputs[lt_name+'tangential_drag'])
        design['mooring']['anchor_types'] = [dict() for m in range(1)] #Note: doesn't work [{}]*anchor_types
        design['mooring']['anchor_types'][0]['name'] = 'drag_embedment'
        design['mooring']['anchor_types'][0]['mass'] = 1e3
        design['mooring']['anchor_types'][0]['cost'] = 1e4
        design['mooring']['anchor_types'][0]['max_vertical_load'] = 0.0
        design['mooring']['anchor_types'][0]['max_lateral_load'] = 1e5

        # DLCs
        # Only give RAFT valid RAFT cases, spectral wind
        turb_ind = discrete_inputs['raft_dlcs_keys'].index('turbulence')
        turb_type = [case_data[turb_ind] for case_data in discrete_inputs['raft_dlcs']]
        case_mask = [
            ('NTM' in tt or 'ETM' in tt or 'EWM' in tt)
             for tt in turb_type]

        design['cases'] = {}
        design['cases']['keys'] = discrete_inputs['raft_dlcs_keys']
        design['cases']['data'] = list(compress(discrete_inputs['raft_dlcs'],case_mask))    # filter cases by case_mask

        # Debug
        if modeling_opt['save_designs']:
            with open(
                os.path.join(analysis_options['general']['folder_output'],'raft_designs',
                f'raft_design_{self.i_design}.pkl'), 'wb') as handle:
                pickle.dump(design, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.i_design += 1
                
        # set up the model
        model = raft.Model(design)
        model.analyzeUnloaded(
            ballast= modeling_opt['trim_ballast'], 
            heave_tol = modeling_opt['heave_tol']
            )
        
        # option to generate seperate HAMS data for level 2 or 3, with higher res settings
        if False: #preprocessBEM:
            model.preprocess_HAMS(dw=dwFAST, wMax=wMaxFAST0, dz=dzFAST, da=daFAST)  
        
        # option to run level 1 load cases
        if True: #processCases:
            model.analyzeCases(runPyHAMS=modeling_opt['runPyHAMS'], meshDir=modeling_opt['BEM_dir'])
            
        # get and process results
        results = model.calcOutputs()
        # Pattern matching for "responses" annd "properties"
        outs = self.list_outputs(out_stream=None, all_procs=True)
        for i in range(len(outs)):
            if outs[i][0].startswith('properties_'):
                name = outs[i][0].split('properties_')[1]
                outputs['properties_'+name] = results['properties'][name]
            elif outs[i][0].startswith('response_'):
                name = outs[i][0].split('response_')[1]
                if np.iscomplex(results['response'][name]).any():
                    outputs['response_'+name] = np.abs(results['response'][name])
                else:
                    outputs['response_'+name] = results['response'][name]


        # Pattern matching for case-by-case outputs
        names = ['surge','sway','heave','roll','pitch','yaw','AxRNA','Mbase','omega','torque','power','bPitch','Tmoor']
        stats = ['avg','std','max','PSD','DEL']
        case_mask = np.array(case_mask)
        for n in names:
            for s in stats:
                if s == 'DEL' and not n in ['Tmoor','Mbase']: continue
                iout = f'{n}_{s}'
                outputs['stats_'+iout][case_mask] = results['case_metrics'][iout]

        # Other case outputs
        for n in ['wind_PSD','wave_PSD']:
            outputs['stats_'+n][case_mask,:] = results['case_metrics'][n]

        # Compute some aggregate outputs manually
        outputs['Max_Offset'] = np.sqrt(outputs['stats_surge_max'][case_mask]**2 + outputs['stats_sway_max'][case_mask]**2).max()
        outputs['heave_avg'] = outputs['stats_heave_avg'][case_mask].mean()
        outputs['Max_PtfmPitch'] = outputs['stats_pitch_max'][case_mask].max()
        outputs['Std_PtfmPitch'] = outputs['stats_pitch_std'][case_mask].mean()
        outputs['max_nacelle_Ax'] = outputs['stats_AxRNA_std'][case_mask].max()
        outputs['rotor_overspeed'] = (outputs['stats_omega_max'][case_mask].max() - inputs['rated_rotor_speed']) / inputs['rated_rotor_speed']
        outputs['max_tower_base'] = outputs['stats_Mbase_max'][case_mask].max()
        
        # Combined outputs for OpenFAST
        outputs['platform_displacement'] = model.fowtList[0].V
        outputs["platform_total_center_of_mass"] = outputs['properties_substructure CG']
        outputs["platform_mass"] = outputs["properties_substructure mass"]
        # Note: Inertia calculated for each case
        outputs["platform_I_total"][:3] = np.r_[outputs['properties_roll inertia at subCG'][0],
                                           outputs['properties_pitch inertia at subCG'][0],
                                           outputs['properties_yaw inertia at subCG'][0]]
