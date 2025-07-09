import openmdao.api as om
import numpy as np
import wisdem.commonse.utilities as util
from wisdem.commonse.wind_wave_drag import cylinderDrag
from wisdem.commonse.environment import PowerWind
from raft.omdao_raft import RAFT_OMDAO
from weis.dlc_driver.dlc_generator    import DLCGenerator

class RAFT_WEIS(om.Group):

    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('analysis_options')

    def setup(self):
        # Stuff WEIS options into RAFT options structure
        weis_opt = self.options['modeling_options']
        analysis_options = self.options['analysis_options']

        raft_opt = {}
        # Level 1 options from WEIS
        for key in weis_opt['RAFT']:
            raft_opt[key] = weis_opt['RAFT'][key]

        # Other options needed for RAFT
        min_freq = weis_opt['RAFT']['min_freq']
        max_freq = weis_opt['RAFT']['max_freq']
        frequencies = np.arange(min_freq, max_freq+0.5*min_freq, min_freq)
        raft_opt['nfreq'] = len(frequencies)
        raft_opt['n_cases'] = weis_opt['DLC_driver']['n_cases']
        raft_opt['intersection_mesh'] = 0

        turbine_opt = {}
        turbine_opt['npts'] = weis_opt['WISDEM']['TowerSE']['n_height_tower']
        turbine_opt['scalar_thicknesses'] = turbine_opt['scalar_diameters'] = turbine_opt['scalar_coefficients'] = False
        turbine_opt['shape'] = 'circ'
        turbine_opt['PC_GS_n'] = weis_opt['ROSCO']['PC_GS_n']
        turbine_opt['n_span'] = weis_opt['WISDEM']['RotorSE']['n_span']
        turbine_opt['n_aoa'] = weis_opt['WISDEM']['RotorSE']['n_aoa']
        turbine_opt['n_Re'] = weis_opt['WISDEM']['RotorSE']['n_Re']
        turbine_opt['n_tab'] = weis_opt['WISDEM']['RotorSE']['n_tab']
        turbine_opt['n_pc'] = weis_opt['WISDEM']['RotorSE']['n_pc']
        turbine_opt['n_af'] = weis_opt['WISDEM']['RotorSE']['n_af']
        turbine_opt['af_used_names'] = weis_opt['WISDEM']['RotorSE']['af_used']

        members_opt = {}
        members_opt['nmembers'] = len(weis_opt["floating"]["members"]["name"])
        members_opt['npts'] = weis_opt["floating"]["members"]["n_height"]
        members_opt['npts_lfill'] = members_opt['npts_rho_fill'] = [int(m-1) for m in members_opt['npts']]
        members_opt['ncaps'] = weis_opt["floating"]["members"]["n_bulkheads"]
        members_opt['nreps'] = [0]*members_opt['nmembers']
        member_shape = weis_opt["floating"]["members"]["outer_shape"]
        # TODO: need to re-visit how to handle square. Can we just get away square with rect?
        member_shape = list(map(lambda x:x.replace("circular", "circ"), member_shape))  # replace all circular with circ
        member_shape = list(map(lambda x:x.replace("rectangular", "rect"), member_shape))
        members_opt['shape'] = member_shape
        members_opt['scalar_thicknesses'] = members_opt['scalar_diameters'] = [False]*members_opt['nmembers']
        members_opt['scalar_coefficients'] = [False]*members_opt['nmembers']
        members_opt['n_ballast_type'] = len(weis_opt["floating"]["members"]["ballast_types"])

        for k in range(members_opt['nmembers']):
            members_opt[f"platform_member{k+1}_potMod"] = weis_opt["RAFT"]["model_potential"][k]

        mooring_opt = {}
        mooring_opt['nlines'] = weis_opt['mooring']['n_lines']
        mooring_opt['nline_types'] = weis_opt['mooring']['n_line_types']
        mooring_opt['nconnections'] = weis_opt['mooring']['n_nodes']

        # Mooring
        for k in range(weis_opt['mooring']['n_nodes']):
            mooring_opt[f'mooring_point{k+1}_name'] = weis_opt['mooring']['node_names'][k]
            mooring_opt[f'mooring_point{k+1}_type'] = weis_opt['mooring']['node_type'][k]

        for k in range(weis_opt['mooring']['n_lines']):
            mooring_opt[f'mooring_line{k+1}_endA'] = weis_opt['mooring']["node1"][k]
            mooring_opt[f'mooring_line{k+1}_endB'] = weis_opt['mooring']["node2"][k]
            mooring_opt[f'mooring_line{k+1}_type'] = weis_opt['mooring']["line_type"][k]

        for k in range(weis_opt['mooring']['n_line_types']):
            mooring_opt[f'mooring_line_type{k+1}_name'] = weis_opt['mooring']["line_type_name"][k]

        self.add_subsystem('wind', PowerWind(nPoints = turbine_opt['npts']), promotes=['Uref','zref'])
        self.add_subsystem('pre', RAFT_WEIS_Prep(modeling_options=weis_opt,
                                                raft_options = raft_opt), promotes=['*'])
        self.add_subsystem('raft', RAFT_OMDAO(modeling_options=raft_opt,
                                              turbine_options=turbine_opt,
                                              mooring_options=mooring_opt,
                                              member_options=members_opt,
                                              analysis_options=analysis_options), promotes=['*'])
        self.connect('wind.U', 'tower_U')

class RAFT_WEIS_Prep(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('raft_options')

    def setup(self):
        opt = self.options['modeling_options']

        n_nodes = opt['mooring']['n_nodes']
        n_lines = opt['mooring']['n_lines']
        n_line_types = opt['mooring']['n_line_types']

        # Process tower layer sections
        n_height_tow = opt["WISDEM"]["TowerSE"]["n_height_tower"]
        n_layers_tow = opt["WISDEM"]["TowerSE"]["n_layers_tower"]
        self.add_input("tower_layer_thickness", val=np.zeros((n_layers_tow, n_height_tow)), units="m")
        self.add_input("tower_rho", val=np.zeros(n_height_tow-1), units="kg/m**3")
        self.add_input("tower_section_height", np.zeros(n_height_tow - 1), units="m", desc="sectional height")
        self.add_input("tower_torsional_stiffness", np.zeros(n_height_tow - 1), units="N*m**2", desc="sectional torsional stiffness")
        self.add_output("turbine_tower_t", val=np.zeros(n_height_tow), units="m")
        self.add_output("turbine_tower_rho_shell", val=0.0, units="kg/m**3")
        self.add_output("turbine_yaw_stiffness", 0.0, units="N*m", desc="sectional torsional stiffness")
        self.add_output('turbine_tower_gamma', val=0.0, units='deg', desc='Twist angle about z-axis')

        # Tower drag
        self.add_input("rho_air", 0.0, units="kg/m**3")
        self.add_input("mu_air", 0.0, units="kg/(m*s)")
        self.add_input('tower_U', val=np.zeros(n_height_tow), units='m/s', desc='Wind speed on the tower')
        self.add_input('turbine_tower_d', val=np.zeros(n_height_tow), units='m', desc='Diameters if circular or side lengths if rectangular')
        self.add_output('turbine_tower_Cd', val=np.zeros(n_height_tow), desc='Transverse drag coefficient')
        self.add_output('turbine_tower_Ca', val=np.zeros(n_height_tow), desc='Transverse added mass coefficient')
        self.add_output('turbine_tower_CdEnd', val=np.zeros(n_height_tow), desc='End axial drag coefficient')
        self.add_output('turbine_tower_CaEnd', val=np.zeros(n_height_tow), desc='End axial added mass coefficient')

        # RNA mass properties
        self.add_input("rna_I_TT", np.zeros(6), units="kg*m**2", desc='Moment of inertia at tower top [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]')
        self.add_input('turbine_mRNA', val=0.0, units='kg', desc='RNA mass')
        self.add_input('drive_height', val=0.0, units='m', desc='Distance from tower top to rotor apex (hub height)')
        self.add_input("rna_cm", np.zeros(3), units="m", desc='RNA center of mass from tower top')
        self.add_output('turbine_IxRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local x axis')
        self.add_output('turbine_IrRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local y or z axes')
        self.add_output('turbine_xCG_RNA', val=0.0, units='m', desc='x location of RNA center of mass')

        n_member = len(opt["floating"]["members"]["name"])
        self.add_input("member_variable_height", val=np.zeros(n_member))
        # Outer diameters and side lenghts are directly connected to the raft_omdao in glue_code
        # not going through here.
        # TODO: bring the outer_diameter connections here because we need some processing
        # due to the difference in how we store the side lengths
        for k in range(n_member):
            n_height = opt["floating"]["members"]["n_height"][k]
            n_layers = opt["floating"]["members"]["n_layers"][k]
            n_ball   = opt["floating"]["members"]["n_ballasts"][k]
            n_bulk   = opt["floating"]["members"]["n_bulkheads"][k]
            if opt["floating"]["members"]["outer_shape"][k] == "circular":
                self.add_input(f"member{k}:outer_diameter", val=np.zeros(n_height), units="m")
                self.add_input(f"member{k}:Ca", val=np.zeros(n_height))
                self.add_input(f"member{k}:Cd", val=np.zeros(n_height))
                self.add_output(f"platform_member{k+1}_d", val=np.zeros(n_height), units="m")
            elif opt["floating"]["members"]["outer_shape"][k] == "rectangular":
                self.add_input(f"member{k}:side_length_a", val=np.zeros(n_height), units="m")
                self.add_input(f"member{k}:side_length_b", val=np.zeros(n_height), units="m")
                self.add_input(f"member{k}:Ca", val=np.zeros(n_height))
                self.add_input(f"member{k}:Cd", val=np.zeros(n_height))
                self.add_input(f"member{k}:Cay", val=np.zeros(n_height))
                self.add_input(f"member{k}:Cdy", val=np.zeros(n_height))
                # RAFT collect a and b into d
                self.add_output(f"platform_member{k+1}_d", val=np.zeros([n_height,2]), units="m")
            self.add_input(f"member{k}:height", val=0.0, units="m")
            self.add_input(f"member{k}:layer_thickness", val=np.zeros((n_layers, n_height)), units="m")
            self.add_input(f"member{k}:rho", val=np.zeros(n_height-1), units="kg/m**3")
            self.add_input(f"member{k}:ballast_grid", val=np.zeros((n_ball,2)))
            self.add_input(f"member{k}:ballast_height", val=np.zeros(n_ball))
            self.add_input(f"member{k}:ballast_density", val=np.zeros(n_ball), units="kg/m**3")
            self.add_input(f"member{k}:ring_stiffener_web_height", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_web_thickness", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_flange_width", 1e-6, units="m")
            self.add_input(f"member{k}:ring_stiffener_flange_thickness", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_spacing", 1.0)
            self.add_input(f"platform_member{k+1}_stations", val=np.zeros(n_height))

            self.add_output(f"platform_member{k+1}_heading", val=np.zeros(0), units='deg')
            self.add_output(f"platform_member{k+1}_gamma", val=0.0, units='deg')
            self.add_output(f"platform_member{k+1}_t", val=np.zeros(n_height), units="m")
            self.add_output(f"platform_member{k+1}_rho_shell", val=0.0, units="kg/m**3")
            self.add_output(f"platform_member{k+1}_l_fill", val=np.zeros(n_height-1), units="m")
            self.add_output(f"platform_member{k+1}_rho_fill", val=np.zeros(n_height-1), units="kg/m**3")
            self.add_output(f"platform_member{k+1}_cap_d_in", val=np.zeros(n_bulk), units='m')
            self.add_output(f"platform_member{k+1}_ring_spacing", val=0.0)
            self.add_output(f"platform_member{k+1}_ring_t", val=0.0, units="m")
            self.add_output(f"platform_member{k+1}_ring_h", val=0.0, units="m")
            if opt["floating"]["members"]["outer_shape"][k] == "circular":
                self.add_output(f"platform_member{k+1}_Cd", val=0.8*np.ones(n_height))
                self.add_output(f"platform_member{k+1}_Ca", val=np.ones(n_height))
            elif opt["floating"]["members"]["outer_shape"][k] == "rectangular":
                self.add_output(f"platform_member{k+1}_Cd", val=0.8*np.ones([n_height, 2]))
                self.add_output(f"platform_member{k+1}_Ca", val=np.ones([n_height, 2]))                
            self.add_output(f"platform_member{k+1}_CdEnd", val=0.6*np.ones(n_height))
            self.add_output(f"platform_member{k+1}_CaEnd", val=0.6*np.ones(n_height))

        # Mooring inputs
        self.add_input('mooring_nodes', val=np.zeros((n_nodes, 3)), units='m', desc='Mooring node locations in global xyz')
        for k in range(n_nodes):
            self.add_output(f'mooring_point{k+1}_location', val=np.zeros(3), units='m', desc='Mooring node location')

        for k in range(n_lines):
            self.add_output(f'mooring_line{k+1}_length', val=0.0, units='m', desc='Length of line')

        self.add_input("unstretched_length", val=np.zeros(n_lines), units="m")
        self.add_input("line_diameter", val=np.zeros(n_lines), units="m")
        self.add_input("line_mass_density", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_stiffness", val=np.zeros(n_lines), units="N")
        self.add_input("line_breaking_load", val=np.zeros(n_lines), units="N")
        self.add_input("line_cost_rate", val=np.zeros(n_lines), units="USD/m")
        self.add_input("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_transverse_drag", val=np.zeros(n_lines))
        self.add_input("line_tangential_drag", val=np.zeros(n_lines))

        for k in range(n_line_types):
            self.add_output(f'mooring_line_type{k+1}_diameter', val=0.0, units='m', desc='Diameter of mooring line type')
            self.add_output(f'mooring_line_type{k+1}_mass_density', val=0.0, units='kg/m**3', desc='Mass density of line type')
            self.add_output(f'mooring_line_type{k+1}_stiffness', val=0.0, desc='Stiffness of line type')
            self.add_output(f'mooring_line_type{k+1}_breaking_load', val=0.0, desc='Breaking load of line type')
            self.add_output(f'mooring_line_type{k+1}_cost', val=0.0, units='USD', desc='Cost of mooring line type')
            self.add_output(f'mooring_line_type{k+1}_transverse_added_mass', val=0.0, desc='Transverse added mass')
            self.add_output(f'mooring_line_type{k+1}_tangential_added_mass', val=0.0, desc='Tangential added mass')
            self.add_output(f'mooring_line_type{k+1}_transverse_drag', val=0.0, desc='Transverse drag')
            self.add_output(f'mooring_line_type{k+1}_tangential_drag', val=0.0, desc='Tangential drag')

        # DLC cases to run
        self.add_input('V_cutin',     val=0.0, units='m/s',      desc='Minimum wind speed where turbine operates (cut-in)')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed where turbine operates (cut-out)')
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbulence class')


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        opt = self.options['modeling_options']
        raft_opt = self.options['raft_options']

        # Tower layer sections
        outputs['turbine_tower_t'] = inputs['tower_layer_thickness'].sum(axis=0)
        outputs['turbine_tower_rho_shell'] = inputs['tower_rho'].mean()
        k_tow_tor = inputs['tower_torsional_stiffness'] / inputs['tower_section_height']
        outputs['turbine_yaw_stiffness'] = 1.0/np.sum(1.0/k_tow_tor)

        # Tower drag
        Re = inputs['rho_air']*inputs['tower_U']*inputs['turbine_tower_d']/inputs['mu_air']
        cd, _ = cylinderDrag(Re)
        outputs['turbine_tower_Cd'] = cd

        # Move tower-top MoI to hub height
        m_rna = float(inputs['turbine_mRNA'][0])
        I_rna = util.assembleI( inputs['rna_I_TT'] )
        # Vector from WISDEM tower-top c.s. to raft tower center-line at hub height c.s.
        r = np.r_[0.0, 0.0, float(inputs['drive_height'][0])]
        outputs['turbine_xCG_RNA'] = (inputs['rna_cm'] - r)[0]
        I_rna_raft = util.unassembleI( I_rna + m_rna * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) )
        outputs['turbine_IxRNA'] = I_rna_raft[0]
        outputs['turbine_IrRNA'] = 0.5*(I_rna_raft[1] + I_rna_raft[2])

        # Floating member data structure transfer
        n_member = len(opt["floating"]["members"]["name"])
        var_height = inputs['member_variable_height']
        for k in range(n_member):

            # Member thickness
            outputs[f"platform_member{k+1}_t"] = inputs[f"member{k}:layer_thickness"].sum(axis=0)
            outputs[f"platform_member{k+1}_rho_shell"] = inputs[f"member{k}:rho"].mean()

            # Convert diameter and side lengths to RAFT
            if opt["floating"]["members"]["outer_shape"][k] == "circular":
                outputs[f"platform_member{k+1}_d"] = inputs[f"member{k}:outer_diameter"]
                outputs[f"platform_member{k+1}_Ca"] = inputs[f"member{k}:Ca"] if np.all(inputs[f"member{k}:Ca"]>0.0) else 1
                outputs[f"platform_member{k+1}_Cd"] = inputs[f"member{k}:Cd"] if np.all(inputs[f"member{k}:Cd"]>0.0) else 1
            elif opt["floating"]["members"]["outer_shape"][k] == "rectangular":
                outputs[f"platform_member{k+1}_d"][:, 0] = inputs[f"member{k}:side_length_a"]
                outputs[f"platform_member{k+1}_d"][:, 1] = inputs[f"member{k}:side_length_b"]
                outputs[f"platform_member{k+1}_Ca"][:, 0] = inputs[f"member{k}:Ca"] if np.all(inputs[f"member{k}:Ca"]>0.0) else 1
                outputs[f"platform_member{k+1}_Cd"][:, 0] = inputs[f"member{k}:Cd"] if np.all(inputs[f"member{k}:Cd"]>0.0) else 1
                outputs[f"platform_member{k+1}_Ca"][:, 1] = inputs[f"member{k}:Cay"] if np.all(inputs[f"member{k}:Cay"]>0.0) else 1
                outputs[f"platform_member{k+1}_Cd"][:, 0] = inputs[f"member{k}:Cdy"] if np.all(inputs[f"member{k}:Cdy"]>0.0) else 1

            # Ring stiffener discretization conversion
            if ( (float(inputs[f"member{k}:ring_stiffener_spacing"][0]) > 0.0) and
                 (float(inputs[f"member{k}:ring_stiffener_spacing"][0]) < 1.0) ):
                outputs[f"platform_member{k+1}_ring_spacing"] = inputs[f"member{k}:ring_stiffener_spacing"]
                h_web = inputs[f"member{k}:ring_stiffener_web_height"]
                t_web = inputs[f"member{k}:ring_stiffener_web_thickness"]
                t_flange = inputs[f"member{k}:ring_stiffener_flange_thickness"]
                w_flange = inputs[f"member{k}:ring_stiffener_flange_width"]
                outputs[f"platform_member{k+1}_ring_h"] = h_web + t_flange
                outputs[f"platform_member{k+1}_ring_t"] = (t_web*h_web + w_flange*t_flange) / (h_web+t_flange)

            # Ballast discretization conversion
            s_grid = inputs[f"platform_member{k+1}_stations"]
            s_ballast = inputs[f"member{k}:ballast_grid"]
            rho_ballast = inputs[f"member{k}:ballast_density"]
            h_ballast = inputs[f"member{k}:ballast_height"]
            l_fill = np.zeros(s_grid.size-1)
            rho_fill = np.zeros(s_grid.size-1)
            for ii in range(s_ballast.shape[0]):
                iball = np.where(s_ballast[ii,0] >= s_grid)[0][-1]
                rho_fill[iball] = rho_ballast[ii]
                # Initial ballast fill values
                if rho_ballast[ii] > 1100.0: # not variable/water
                    l_fill[iball] = h_ballast[ii]
                else:
                    l_fill[iball] = var_height[k]

            # fix ballast l_fill for RAFT: it does not want l_fill > section size
            # add ballast from last section to next
            remaining_fill = 0
            for i_sec, l in enumerate(l_fill):

                # if there's left over fill, use previous density
                # will use lower ballast if the chamber is too small
                # hopefully solver will bias platform design away from this case eventually
                if remaining_fill:
                    rho_fill[i_sec] = rho_fill[i_sec-1]
                
                # length of current section
                sec_length = s_grid[i_sec+1] - s_grid[i_sec]

                if l and remaining_fill:
                    print('WEIS Warning: there is left over ballast from a previous section and a new ballast being added.  Platform masses may be inconsistent between RAFT and FloatingSE')

                # add fill from previous section
                l += remaining_fill     
                if l > sec_length:  # only fill up to section length
                    l_fill[i_sec] = sec_length
                    remaining_fill = l - sec_length
                else:   # fill with remaining
                    l_fill[i_sec] = l
                    remaining_fill = 0

            if remaining_fill:  # there is ballast still to be filled
                print('WEIS Warning: Not all ballast was assigned to a RAFT section.')
            outputs[f"platform_member{k+1}_l_fill"] = l_fill
            outputs[f"platform_member{k+1}_rho_fill"] = rho_fill

        # Mooring
        for k in range(opt['mooring']['n_nodes']):
            outputs[f'mooring_point{k+1}_location'] = inputs['mooring_nodes'][k,:]

        for k in range(opt['mooring']['n_lines']):
            outputs[f'mooring_line{k+1}_length'] = inputs['unstretched_length'][k]

        for k in range(opt['mooring']['n_line_types']):
            outputs[f'mooring_line_type{k+1}_cost'] = inputs['line_cost_rate'][k]

            for var in ['diameter','mass_density','stiffness','breaking_load',
                        'transverse_added_mass','tangential_added_mass','transverse_drag','tangential_drag']:
                outputs[f'mooring_line_type{k+1}_{var}'] = inputs[f'line_{var}'][k]


        # Cases
        DLCs = opt['DLC_driver']['DLCs']
        # Initialize the DLC generator
        cut_in = inputs['V_cutin']
        cut_out = inputs['V_cutout']
        rated = inputs['Vrated']
        ws_class = discrete_inputs['turbine_class']
        turb_class = discrete_inputs['turbulence_class']
        fix_wind_seeds = opt['DLC_driver']['fix_wind_seeds']
        fix_wave_seeds = opt['DLC_driver']['fix_wave_seeds']
        metocean = opt['DLC_driver']['metocean_conditions']
        dlc_generator = DLCGenerator(cut_in, cut_out, rated, ws_class, turb_class, fix_wind_seeds, fix_wave_seeds, metocean)
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)
        # Build case table for RAFT
        raft_cases = [[]]*dlc_generator.n_cases
        for i, icase in enumerate(dlc_generator.cases):
            if 'EWM' in icase.IEC_WindType:
                wind_type = 'EWM'
            else:
                wind_type = icase.IEC_WindType[-3:]
            turbStr = f'{dlc_generator.wind_speed_class}{turb_class}_{wind_type}'
            opStr = 'operating' if icase.turbine_status.lower() == 'operating' else 'parked'
            waveStr = 'JONSWAP' #icase.wave_spectrum
            raft_cases[i] = [float(icase.URef),
                             float(icase.wind_heading),
                             str(turbStr),
                             str(opStr),
                             float(icase.yaw_misalign),
                             str(waveStr),
                             float(max(1.0, icase.wave_period)),
                             float(max(1.0, icase.wave_height)),
                             float(icase.wave_heading)]
        raft_opt['raft_dlcs'] = raft_cases
        raft_opt['raft_dlcs_keys'] = ['wind_speed', 'wind_heading', 'turbulence',
                                              'turbine_status', 'yaw_misalign', 'wave_spectrum',
                                              'wave_period', 'wave_height', 'wave_heading']
