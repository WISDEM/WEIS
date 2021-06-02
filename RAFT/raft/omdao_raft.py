import openmdao.api as om
import raft
import numpy as np

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

    def setup(self):

        # unpack options
        modeling_opt = self.options['modeling_options']
        nfreq = modeling_opt['nfreq']

        turbine_opt = self.options['turbine_options']
        turbine_npts = turbine_opt['npts']

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

        mooring_opt = self.options['mooring_options']
        nlines = mooring_opt['nlines']
        nline_types = mooring_opt['nline_types']
        nconnections = mooring_opt['nconnections']

        # frequency domain
        self.add_input('frequency_range', val=np.zeros(nfreq), units='Hz', desc='Frequency range to compute response over')

        # turbine inputs
        self.add_input('turbine_mRNA', val=0.0, units='kg', desc='RNA mass')
        self.add_input('turbine_IxRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local x axis')
        self.add_input('turbine_IrRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local y or z axes')
        self.add_input('turbine_xCG_RNA', val=0.0, units='m', desc='x location of RNA center of mass')
        
        self.add_input('turbine_hHub', val=0.0, units='m', desc='Hub height above water line')
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
        self.add_input('rotor_PC_GS_angles',     val=np.zeros(turbine_opt['PC_GS_n']),   units='rad',        desc='Gain-schedule table: pitch angles')
        self.add_input('rotor_PC_GS_Kp',         val=np.zeros(turbine_opt['PC_GS_n']),   units='s',          desc='Gain-schedule table: pitch controller kp gains')
        self.add_input('rotor_PC_GS_Ki',         val=np.zeros(turbine_opt['PC_GS_n']),                       desc='Gain-schedule table: pitch controller ki gains')
        self.add_input('rotor_Fl_Kp',            val=0.0,                        desc='Floating feedback gain')
        self.add_input('rotor_inertia',          val=0.0,    units='kg*m**2',    desc='Rotor inertia')

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
        nballast = np.sum(member_npts_rho_fill)
        self.add_output('properties_tower mass', units='kg', desc='Tower mass')
        self.add_output('properties_tower CG', val=np.zeros(ndim), units='m', desc='Tower center of gravity')
        self.add_output('properties_substructure mass', val=0.0, units='kg', desc='Substructure mass')
        self.add_output('properties_substructure CG', val=np.zeros(ndim), units='m', desc='Substructure center of gravity')
        self.add_output('properties_shell mass', val=0.0, units='kg', desc='Shell mass')
        self.add_output('properties_ballast mass', val=np.zeros(nballast), units='m', desc='Ballast mass')
        self.add_output('properties_ballast densities', val=np.zeros(nballast), units='kg', desc='Ballast densities')
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
        self.add_output('response_pitch RAO', val=np.zeros(nfreq), units='m', desc='Pitch RAO')
        self.add_output('response_roll RAO', val=np.zeros(nfreq), units='m', desc='Roll RAO')
        self.add_output('response_yaw RAO', val=np.zeros(nfreq), units='m', desc='Yaw RAO')
        self.add_output('response_nacelle acceleration', val=np.zeros(nfreq), units='m/s**2', desc='Nacelle acceleration')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        turbine_opt = self.options['turbine_options']
        mooring_opt = self.options['mooring_options']
        members_opt = self.options['member_options']
        modeling_opt = self.options['modeling_options']

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
        
        design['potModMaster'] = int(modeling_opt['potModMaster'])
        design['XiStart'] = float(modeling_opt['XiStart'])
        design['nIter'] = int(modeling_opt['nIter'])
        design['dlsMax'] = float(modeling_opt['dlsMax'])

        # TODO: these float conversions are messy
        design['turbine'] = {}
        design['turbine']['mRNA'] = float(inputs['turbine_mRNA'])
        design['turbine']['IxRNA'] = float(inputs['turbine_IxRNA'])
        design['turbine']['IrRNA'] = float(inputs['turbine_IrRNA'])
        design['turbine']['xCG_RNA'] = float(inputs['turbine_xCG_RNA'])
        design['turbine']['hHub'] = float(inputs['turbine_hHub'])
        design['turbine']['Fthrust'] = float(inputs['turbine_Fthrust'])
        design['turbine']['yaw_stiffness'] = float(inputs['turbine_yaw_stiffness'])
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

        design['turbine']['control'] = {}
        design['turbine']['control']['PC_GS_Angles']    = inputs['rotor_PC_GS_angles']
        design['turbine']['control']['PC_GS_Kp']        = inputs['rotor_PC_GS_Kp']
        design['turbine']['control']['PC_GS_Ki']        = inputs['rotor_PC_GS_Ki']
        design['turbine']['control']['Fl_Kp']           = float(inputs['rotor_Fl_Kp'])
        design['turbine']['control']['I_drivetrain']    = float(inputs['rotor_inertia'])

        design['platform'] = {}
        design['platform']['members'] = [dict() for i in range(nmembers)]
        for i in range(nmembers):
            m_name = f'platform_member{i+1}_'
            m_shape = member_shapes[i]
            mnpts_lfill = member_npts_lfill[i]
            mncaps = member_ncaps[i]
            mnreps = member_nreps[i]
            mnpts = member_npts[i]
            
            design['platform']['members'][i]['name'] = m_name
            design['platform']['members'][i]['type'] = i + 2
            design['platform']['members'][i]['rA'] = inputs[m_name+'rA']
            design['platform']['members'][i]['rB'] = inputs[m_name+'rB']
            design['platform']['members'][i]['shape'] = m_shape
            design['platform']['members'][i]['gamma'] = float(inputs[m_name+'gamma'])
            design['platform']['members'][i]['potMod'] = discrete_inputs[m_name+'potMod']
            design['platform']['members'][i]['stations'] = inputs[m_name+'stations']
            
            # updated version to better handle 'diameters' between circular and rectangular members
            if m_shape == 'circ' or m_shape == 'square':
                if member_scalar_d[i]:
                    design['platform']['members'][i]['d'] = [float(inputs[m_name+'d'])]*mnpts
                else:
                    design['platform']['members'][i]['d'] = inputs[m_name+'d']
            elif m_shape == 'rect':
                if member_scalar_d[i]:
                    design['platform']['members'][i]['d'] = [inputs[m_name+'d']]*mnpts
                else:
                    design['platform']['members'][i]['d'] = inputs[m_name+'d']
            ''' original version of handling diameters
            if member_scalar_d[i]:
                design['platform']['members'][i]['d'] = float(inputs[m_name+'d'])
            else:
                design['platform']['members'][i]['d'] = inputs[m_name+'d']
            '''
            if member_scalar_t[i]:
                design['platform']['members'][i]['t'] = float(inputs[m_name+'t'])
            else:
                design['platform']['members'][i]['t'] = inputs[m_name+'t']
            if member_scalar_coeff[i]:
                design['platform']['members'][i]['Cd'] = float(inputs[m_name+'Cd'])
                design['platform']['members'][i]['Ca'] = float(inputs[m_name+'Ca'])
                design['platform']['members'][i]['CdEnd'] = float(inputs[m_name+'CdEnd'])
                design['platform']['members'][i]['CaEnd'] = float(inputs[m_name+'CaEnd'])
            else:
                design['platform']['members'][i]['Cd'] = inputs[m_name+'Cd']
                design['platform']['members'][i]['Ca'] = inputs[m_name+'Ca']
                design['platform']['members'][i]['CdEnd'] = inputs[m_name+'CdEnd']
                design['platform']['members'][i]['CaEnd'] = inputs[m_name+'CaEnd']
            design['platform']['members'][i]['rho_shell'] = float(inputs[m_name+'rho_shell'])
            if mnreps > 0:
                design['platform']['members'][i]['heading'] = inputs[m_name+'heading']
            if mnpts_lfill > 0:
                design['platform']['members'][i]['l_fill'] = inputs[m_name+'l_fill']
                design['platform']['members'][i]['rho_fill'] = inputs[m_name+'rho_fill']
            if ( (mncaps > 0) or (inputs[m_name+'ring_spacing'] > 0) ):
                # Member discretization
                s_grid = inputs[m_name+'stations']
                s_height = s_grid[-1] - s_grid[0]
                # Get locations of internal structures based on spacing
                ring_spacing = inputs[m_name+'ring_spacing']
                n_stiff = 0 if ring_spacing == 0.0 else int(np.floor(s_height / ring_spacing))
                s_ring = (np.arange(1, n_stiff + 0.1) - 0.5) * (ring_spacing / s_height)
                #d_ring = np.interp(s_ring, s_grid, inputs[m_name+'d'])
                d_ring = np.interp(s_ring, s_grid, design['platform']['members'][i]['d'])
                # Combine internal structures based on spacing and defined positions
                s_cap = np.r_[s_ring, inputs[m_name+'cap_stations']]
                t_cap = np.r_[inputs[m_name+'ring_t']*np.ones(n_stiff), inputs[m_name+'cap_t']]
                di_cap = np.r_[d_ring-2*inputs[m_name+'ring_h'], inputs[m_name+'cap_d_in']]
                # Store vectors in sorted order
                isort = np.argsort(s_cap)
                design['platform']['members'][i]['cap_stations'] = s_cap[isort]
                design['platform']['members'][i]['cap_t'] = t_cap[isort]
                design['platform']['members'][i]['cap_d_in'] = di_cap[isort]

        design['mooring'] = {}
        design['mooring']['water_depth'] = inputs['mooring_water_depth']
        design['mooring']['points'] = [dict() for i in range(nconnections)]
        for i in range(0, nconnections):
            pt_name = f'mooring_point{i+1}_'
            design['mooring']['points'][i]['name'] = discrete_inputs[pt_name+'name']
            design['mooring']['points'][i]['type'] = discrete_inputs[pt_name+'type']
            design['mooring']['points'][i]['location'] = inputs[pt_name+'location']

        design['mooring']['lines'] = [dict() for i in range(nlines)]
        for i in range(0, nlines):
            ml_name = f'mooring_line{i+1}_'
            design['mooring']['lines'][i]['name'] = f'line{i+1}'
            design['mooring']['lines'][i]['endA'] = discrete_inputs[ml_name+'endA']
            design['mooring']['lines'][i]['endB'] = discrete_inputs[ml_name+'endB']
            design['mooring']['lines'][i]['type'] = discrete_inputs[ml_name+'type']
            design['mooring']['lines'][i]['length'] = inputs[ml_name+'length']
        design['mooring']['line_types'] = [dict() for i in range(nline_types)]
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

        # grab the depth
        depth = float(design['mooring']['water_depth'])

        # set up frequency range for computing response over
        w = inputs['frequency_range']

        # create and run the model
        model = raft.Model(design, w=w, depth=depth)
        model.setEnv(spectrum="unit")
        model.calcSystemProps()
        model.solveEigen()
        model.calcMooringAndOffsets()
        model.solveDynamics()
        results = model.calcOutputs()

        outs = self.list_outputs(values=False, out_stream=None)
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
