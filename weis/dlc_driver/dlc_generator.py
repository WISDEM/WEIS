import numpy as np
import os
import logging
import copy
import weis.inputs as sch
from weis.dlc_driver.turbulence_models import IEC_TurbulenceModels
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from openfast_io.FileTools import remove_numpy
from weis.aeroelasticse.utils import OLAFParams

logger = logging.getLogger("wisdem/weis")

# TODO: not sure where this should live, so it's a global for now
# This is a global that we copy into the DLC_Driver class as an attribute
# Could it be an input yaml?
openfast_input_map = {
    # Generic name: OpenFAST input (list if necessary)
    'total_time': ("Fst","TMax"),
    'transient_time': ("Fst","TStart"),
    
    'WindFile_type': ("InflowWind","WindType"),
    'wind_speed': ("InflowWind","HWindSpeed"),
    'WindFile_name': ("InflowWind","FileName_BTS"),
    'WindFile_name': ("InflowWind","FileName_Uni"),
    'rotorD': ("InflowWind","RefLength"),
    'WindHd': ("InflowWind","PropagationDir"),
    'hub_height': ("InflowWind","RefHt_Uni"),
    
    'rot_speed_initial': ("ElastoDyn","RotSpeed"),
    'pitch_initial': [("ElastoDyn","BlPitch1"),("ElastoDyn","BlPitch2"),("ElastoDyn","BlPitch3")],
    'azimuth_init': ("ElastoDyn","Azimuth"),
    'yaw_misalign': ("ElastoDyn","NacYaw"),

    'compute_aerodynamics': ("Fst", "CompAero"),
    'compute_inflow': ("Fst", "CompInflow"),
    'compute_control': ("Fst", "CompServo"),
    
    'generator_dof':            ("ElastoDyn","GenDOF"),
    'initial_platform_surge':   ("ElastoDyn","PtfmSurge"),
    'initial_platform_sway':    ("ElastoDyn","PtfmSway"),
    'initial_platform_heave':   ("ElastoDyn","PtfmHeave"),
    'initial_platform_roll':    ("ElastoDyn","PtfmRoll"),
    'initial_platform_pitch':   ("ElastoDyn","PtfmPitch"),
    'initial_platform_yaw':     ("ElastoDyn","PtfmYaw"),
    
    'wave_height': ("SeaState","WaveHs"),
    'wave_period': ("SeaState","WaveTp"),
    'wave_direction': ("SeaState","WaveDir"),
    'wave_gamma': ("SeaState","WavePkShp"),
    'wave_seed': ("SeaState","WaveSeed1"),

    'wave_model': ("SeaState","WaveMod"),
    
    'shutdown_time': [
        ("ServoDyn","TPitManS1"),
        ("ServoDyn","TPitManS2"),
        ("ServoDyn","TPitManS3"),
        ("ServoDyn","TimGenOf"),
        ],

    'startup_time': [
        ("ServoDyn","TimGenOn"),
        ("ServoDyn","TPCOn"),
    ],
        

    'final_blade_pitch': [
        ("ServoDyn","BlPitchF(1)"),
        ("ServoDyn","BlPitchF(2)"),
        ("ServoDyn","BlPitchF(3)"),
        
    ],
    'pitchfault_time1': ("ServoDyn","TPitManS1"),
    'pitchfault_time2': ("ServoDyn","TPitManS2"),
    'pitchfault_time3': ("ServoDyn","TPitManS3"),
    'pitchfault_blade1pos': ("ServoDyn","BlPitchF(1)"),
    'pitchfault_blade2pos': ("ServoDyn","BlPitchF(2)"),
    'pitchfault_blade3pos': ("ServoDyn","BlPitchF(3)"),
    'genfault_time': ("ServoDyn","TimGenOf"),
    'yawfault_time': ("ServoDyn","TYawManS"),
    'yawfault_yawpos': ("ServoDyn","NacYawF"),
    
    'aero_mod': ("AeroDyn","AFAeroMod"),
    'wake_mod': ("AeroDyn","Wake_Mod"),
    'tau1_const': ("AeroDyn","tau1_const"),


    # 'dlc_label': ("DLC","Label"),
    # 'wind_seed': ("DLC","WindSeed"),
    # 'wind_speed': ("DLC","MeanWS"),

    # TODO: where should turbsim live?
    # These aren't actually used to generate turbsim, the generic inputs are used
    # However, I think it's better to be over-thorough and check that inputs are applied than the uncertainty of not checking any
    'wind_seed': ("TurbSim", "RandSeed1"),
    'direction': ("TurbSim", "direction_pn"),
    'shear': ("TurbSim", "shear_hv")
}

class DLCInstance(object):

    def __init__(self, options=None):
        # Set default DLC with empty properties
        self.URef = 0.0
        self.wind_heading = 0.0
        self.yaw_misalign = 0.0
        self.wave_height = 0.0
        self.wave_period = 0.0
        self.wave_heading = 0.0
        self.wave_gamma = 0.0
        self.probability = 0.0
        self.analysis_time = 600.
        self.transient_time = 120.
        self.shutdown_time = 9999.
        self.IEC_WindType = 'NTM'
        self.turbine_status = 'operating'
        self.wave_spectrum = 'JONSWAP'
        self.turbulent_wind = False
        self.direction_pn = '' # Positive (p) or negative (n), used for ECD
        self.shear_hv = '' # Horizontal (h) or vertical (v), used for EWS
        self.sigma1 = '' # Standard deviation of the wind
        self.RandSeed1 = 0
        self.wave_seed1 = 0
        self.label = '' # For 1.1/Custom
        self.wind_file = ''
        self.PSF = 1.35 # Partial Safety Factor
        self.azimuth_init = 0.0

        if not options is None:
            self.default_turbsim_props(options)

    def default_turbsim_props(self, options):
        for key in options['turbulent_wind'].keys():
            setattr(self, key, options['turbulent_wind'][key])

class DLCGenerator(object):

    dlc_schema = sch.validation.get_modeling_schema()['properties']['DLC_driver']['properties']['DLCs']['items']['properties']    

    def __init__(
            self, 
            ws_cut_in=4.0, 
            ws_cut_out=25.0, 
            ws_rated=10.0, 
            wind_speed_class = 'I',
            wind_turbulence_class = 'B', 
            fix_wind_seeds=True, 
            fix_wave_seeds=True, 
            metocean={},
            dlc_driver_options={},      # note that with this addition, it includes metocean
            initial_condition_table = {},
            default_options = {},
            ):
        self.ws_cut_in = ws_cut_in
        self.ws_cut_out = ws_cut_out
        self.wind_speed_class = wind_speed_class
        self.wind_turbulence_class = wind_turbulence_class
        self.ws_rated = ws_rated
        self.cases = []
        if fix_wind_seeds:
            self.rng_wind = np.random.default_rng(12345)
        else:
            self.rng_wind = np.random.default_rng()
        if fix_wave_seeds:
            self.rng_wave = np.random.default_rng(6789)
        else:
            self.rng_wave = np.random.default_rng()
        self.n_cases = 0
        self.n_ws_dlc11 = 0

        # OpenFAST input map
        self.openfast_input_map = copy.deepcopy(openfast_input_map)

        # Set and update default_options, applied to dlc_options and first group in case_inputs
        self.default_options = {
            'wake_mod': 1,
            'wave_model': 2,
        }
        self.default_options.update(default_options)

        # Init openfast case list
        self.openfast_case_inputs = []

        # Metocean conditions
        self.mo_ws = metocean['wind_speed']
        self.mo_Hs_NSS = metocean['wave_height_NSS']
        self.mo_Tp_NSS = metocean['wave_period_NSS']
        self.mo_Hs_F = metocean['wave_height_fatigue']
        self.mo_Tp_F = metocean['wave_period_fatigue']
        self.mo_Hs_SSS = metocean['wave_height_SSS']
        self.mo_Tp_SSS = metocean['wave_period_SSS']
        if len(self.mo_ws)!=len(self.mo_Hs_NSS):
            raise Exception('The vector of metocean conditions wave_height_NSS in the modeling options must have the same length of the tabulated wind speeds')
        if len(self.mo_ws)!=len(self.mo_Tp_NSS):
            raise Exception('The vector of metocean conditions wave_period_NSS in the modeling options must have the same length of the tabulated wind speeds')
        if len(self.mo_ws)!=len(self.mo_Hs_F):
            raise Exception('The vector of metocean conditions wave_height_fatigue in the modeling options must have the same length of the tabulated wind speeds')
        if len(self.mo_ws)!=len(self.mo_Tp_F):
            raise Exception('The vector of metocean conditions wave_period_fatigue in the modeling options must have the same length of the tabulated wind speeds')
        if len(self.mo_ws)!=len(self.mo_Hs_SSS):
            raise Exception('The vector of metocean conditions wave_height_SSS in the modeling options must have the same length of the tabulated wind speeds')
        if len(self.mo_ws)!=len(self.mo_Tp_SSS):
            raise Exception('The vector of metocean conditions wave_period_SSS in the modeling options must have the same length of the tabulated wind speeds')

        # Load extreme wave heights and periods
        self.wave_height50 = np.array([metocean['wave_height50']])
        self.wave_period50 = np.array([metocean['wave_period50']])
        self.wave_height1 = np.array([metocean['wave_height1']])
        self.wave_period1 = np.array([metocean['wave_period1']])

        self.initial_condition_table = initial_condition_table

        # Add to openfast_input_map
        if ('openfast_input_map' in dlc_driver_options) and dlc_driver_options['openfast_input_map']:
            for key, value in dlc_driver_options['openfast_input_map'].items():

                if key in self.openfast_input_map:
                    raise Exception(f'The user-defined openfast_input_map key {key} is already defined.')

                # If the value is a list
                if is_list_of_lists(value):
                    self.openfast_input_map[key] = [tuple(v) for v in value]
                else:
                    self.openfast_input_map[key] = tuple(value)

    def IECwind(self):
        self.IECturb = IEC_TurbulenceModels()
        self.IECturb.Turbine_Class = self.wind_speed_class
        self.IECturb.Turbulence_Class = self.wind_turbulence_class
        self.IECturb.setup()
        _, self.V_e50, self.V_e1, _, _ = self.IECturb.EWM(0.)
        self.V_ref = self.IECturb.V_ref
        self.wind_speed_class_num = self.IECturb.Turbine_Class_Num

    def to_dict(self):
        return [vars(m) for m in self.cases]

    def get_wind_speeds(self, options):
        if len(options['wind_speed']) > 0:
            wind_speed = np.array( [float(m) for m in options['wind_speed']] )
        else:
            wind_speed = np.arange(self.ws_cut_in, self.ws_cut_out+0.5*options['ws_bin_size'], options['ws_bin_size'])
            if wind_speed[-1] != self.ws_cut_out:
                wind_speed = np.append(wind_speed, self.ws_cut_out)

        return wind_speed

    def get_wind_seeds(self, options, wind_speed):
        
        if len(options['wind_seed']) > 0:
            wind_seed = np.array( [int(m) for m in options['wind_seed']] )
        else:
            wind_seed = self.rng_wind.integers(2147483648, size=options['n_seeds']*len(wind_speed), dtype=int)
            wind_speed = np.repeat(wind_speed, options['n_seeds'])

        return wind_speed, wind_seed

    def get_wave_seeds(self, options, wind_speed):
        if len(options['wave_seed']) > 0:
            wave_seed = np.array( [int(m) for m in options['wave_seed']] )
        else:
            wave_seed = self.rng_wave.integers(2147483648, size=len(wind_speed), dtype=int)

        return wave_seed

    def get_wind_heading(self, options):
        if len(options['wind_heading']) > 0:
            wind_heading = np.array( [float(m) for m in options['wind_heading']] )
        else:
            wind_heading = np.array([])
        return wind_heading

    def get_wave_height(self, options):
        if len(options['wave_height']) > 0:
            wave_height = np.array( [float(m) for m in options['wave_height']] )
        else:
            wave_height = np.array([])
        return wave_height

    def get_wave_period(self, options):
        if len(options['wave_period']) > 0:
            wave_period = np.array( [float(m) for m in options['wave_period']] )
        else:
            wave_period = np.array([])
        return wave_period

    def get_wave_gamma(self, options):
        if len(options['wave_gamma']) > 0:
            wave_gamma = np.array( [float(m) for m in options['wave_gamma']] )
        else:
            wave_gamma = np.array([])
        return wave_gamma

    def get_wave_heading(self, options):
        if len(options['wave_heading']) > 0:
            wave_heading = np.array( [float(m) for m in options['wave_heading']] )
        else:
            wave_heading = np.array([])
        return wave_heading

    def get_probabilities(self, options):
        if len(options['probabilities']) > 0:
            probabilities = np.array( [float(m) for m in options['probabilities']] )
        else:
            probabilities = np.array([])
        return probabilities

    def get_metocean(self, options):
        wind_speeds_indiv = self.get_wind_speeds(options)
        wind_speed, wind_seed = self.get_wind_seeds(options, wind_speeds_indiv)
        wave_seed = self.get_wave_seeds(options, wind_speed)
        wind_heading = self.get_wind_heading(options)
        wave_height = self.get_wave_height(options)
        wave_period = self.get_wave_period(options)
        wave_gamma = self.get_wave_gamma(options)
        wave_heading = self.get_wave_heading(options)
        probabilities = self.get_probabilities(options)

        if len(wind_seed) > 1 and len(wind_seed) != len(wind_speed):
            raise Exception("The vector of wind_seed must have either length=1 or the same length of wind speeds")
        if len(wind_heading) > 1 and len(wind_heading) != len(wind_speed):
            raise Exception("The vector of wind_heading must have either length=1 or the same length of wind speeds")
        if len(wave_seed) > 1 and len(wave_seed) != len(wind_speed):
            raise Exception("The vector of wave seeds must have the same length of wind speeds or not defined")
        if len(wave_height) > 1 and len(wave_height) != len(wind_speed):
            raise Exception("The vector of wave heights must have either length=1 or the same length of wind speeds")
        if len(wave_period) > 1 and len(wave_period) != len(wind_speed):
            raise Exception("The vector of wave periods must have either length=1 or the same length of wind speeds")
        if len(wave_gamma) > 1 and len(wave_gamma) != len(wind_speed):
            raise Exception("The vector of wave_gamma must have either length=1 or the same length of wind speeds")
        if len(wave_heading) > 1 and len(wave_heading) != len(wind_speed):
            raise Exception("The vector of wave heading must have either length=1 or the same length of wind speeds")
        if len(probabilities) > 1 and len(probabilities) != len(wind_speed):
            raise Exception("The vector of probabilities must have either length=1 or the same length of wind speeds")
        if abs(sum(probabilities) - 1.) > 1.e-3:
            raise Exception("The vector of probabilities must sum to 1")
        
        metocean_case_info = {}
        metocean_case_info['wind_speed'] = wind_speed
        metocean_case_info['wind_seed'] = wind_seed
        metocean_case_info['wave_seed'] = wave_seed
        metocean_case_info['wind_heading'] = wind_heading
        metocean_case_info['wave_height'] = wave_height
        metocean_case_info['wave_period'] = wave_period
        # metocean_case_info['current_speeds'] = current_speeds
        metocean_case_info['wave_gamma'] = wave_gamma
        metocean_case_info['wave_heading'] = wave_heading
        metocean_case_info['probabilities'] = probabilities       
        # metocean_case_info['current_std'] = self.mo_current_std       
        
        return metocean_case_info


    def generate(self, label, options):
        # Use schema to determine known_dlcs (weis/inputs/modeling_schema.yaml)
        known_dlcs = self.dlc_schema['DLC']['enum']
        self.OF_dlccaseinputs = {key: None for key in known_dlcs}

        # Get extreme wind speeds
        self.IECwind()

        found = False
        for ilab in known_dlcs:
            func_name = 'generate_'+str(ilab).replace('.','p')

            if label in [ilab, str(ilab)]: # Match either 1.1 or '1.1'
                found = True
                getattr(self, func_name)(options) # calls self.generate_1p1(options)
                break

        if not found:
            raise ValueError(f'DLC {label} is not currently supported')

        self.n_cases = len(self.cases)

    def generate_cases(self,generic_case_inputs,dlc_options):
        '''
        This method will generate the simulation inputs for each design load case

        generic_case_inputs is a list of lists of strings with the options used to create a case matrix
        dlc_options is a dictionary, some of its keys will be in generic_case_inputs and used to generate the cases
            Other keys include:
            sea_state is a string: either severe or normal
            label is the string label used in when the wind inputs are created
        '''

        # Handle default options
        if 'sea_state' not in dlc_options:
            dlc_options['sea_state'] = 'normal'

        if 'PSF' not in dlc_options:
            dlc_options['PSF'] = 1.35

       
        # Generate case list, both generic and OpenFAST specific
        self.set_time_options(dlc_options)
        met_options = self.gen_met_options(dlc_options, sea_state=dlc_options['sea_state'])
        
        # Add met options to dlc_options for output reporting
        dlc_options.update(met_options)
        dlc_options = remove_numpy(dlc_options)

        # Apply initial conditions
        self.apply_initial_conditions(generic_case_inputs,dlc_options, met_options)

        # Add user defined groups
        self.add_user_groups(generic_case_inputs, dlc_options)

        # Generate case list
        generic_case_list = self.gen_case_list(dlc_options,met_options,generic_case_inputs)

        # DLC specific: Make idlc for other parts of WEIS (mostly turbsim generation)
        for _, case in enumerate(generic_case_list):
            idlc = DLCInstance(options=dlc_options)
            if dlc_options['IEC_WindType'] == 'ECD':
                idlc.turbulent_wind = False
                idlc.direction_pn = case['direction']
            elif dlc_options['IEC_WindType'] == 'EOG':
                idlc.turbulent_wind = False
                idlc.sigma1,idlc.V_e1 = self.IECturb.EOG(case['wind_speed'])
            elif dlc_options['IEC_WindType'] == 'EWS':
                idlc.turbulent_wind = False
                idlc.direction_pn = case['direction']
                idlc.shear_hv = case['shear']
                idlc.sigma1 = self.IECturb.NTM(case['wind_speed'])
            else:
                idlc.turbulent_wind = True
                idlc.RandSeed1 = case['wind_seed']
            idlc.URef = case['wind_speed']
            idlc.label = dlc_options['label']
            idlc.total_time = case['total_time']
            idlc.IEC_WindType = dlc_options['IEC_WindType']
            idlc.turbine_status = dlc_options['turbine_status']

            # Apply case_list info to idlc
            for key in case:
                setattr(idlc,key,case[key])

            #if dlc_options['label'] == '1.2':
            #    idlc.probability = probabilities[i_WaH]
            self.cases.append(idlc)

            # AEP DLC: set constant turbulence intensity
            if dlc_options['label'] == 'AEP':
                idlc.IECturbc = self.IECturb.NTM(idlc.URef) * dlc_options['TI_factor'] / idlc.URef * 100

            
    def apply_sea_state(self,met_options,sea_state='normal'):
        '''
        Apply waves based on the expected values provided in the metocean inputs
        Will use met_options as an input and modify that dict
        sea_state can be normal, severe
        '''
        allowed_sea_states = ['normal','severe','50-year','1-year']
        if sea_state not in allowed_sea_states:
            raise Exception(f'Selected sea state of {sea_state} is not in allowed_sea_states: {allowed_sea_states}')
        
        # Select wind speed, sea state lookup
        if sea_state == 'normal':
            wind_speed_table = self.mo_ws
            wave_height_table = self.mo_Hs_NSS
            wave_period_table = self.mo_Tp_NSS
        elif sea_state == 'severe':
            wind_speed_table = self.mo_ws
            wave_height_table = self.mo_Hs_SSS
            wave_period_table = self.mo_Tp_SSS
        elif sea_state == '50-year':
            wind_speed_table = [50.]
            wave_height_table = self.wave_height50
            wave_period_table = self.wave_period50
        elif sea_state == '1-year':
            wind_speed_table = [50.]
            wave_height_table = self.wave_height1
            wave_period_table = self.wave_period1


        # If the user has not defined Hs (wave_height in modopts) and Tp (wave_period in modopts), apply the metocean conditions defined by the table
        if len(met_options['wave_height'])==0:
            met_options['wave_height'] = np.interp(met_options['wind_speed'], wind_speed_table, wave_height_table)
        if len(met_options['wave_period'])==0:
            met_options['wave_period'] = np.interp(met_options['wind_speed'], wind_speed_table, wave_period_table)

    def set_time_options(self, options):
        '''
        Handle time options and add total_time to dict
        Default for analysis and transient_time is 0
        '''
        if options['analysis_time'] > 0:
            options['analysis_time'] = options['analysis_time']
        else:
            options['analysis_time'] = 600.
        if options['transient_time'] >= 0:
            options['transient_time'] = options['transient_time']
        options['total_time'] = options['analysis_time'] + options['transient_time']

    def gen_case_list(self,dlc_options, met_options, generic_case_inputs):
        '''
        Generate case list from generic_case_inputs
        TODO: this whole thing could be moved into generate_cases, thoughts?
        '''

        
        # Combine
        comb_options = combine_options(dlc_options,met_options)

        # Check that all inputs are valid options
        all_inputs = sum(generic_case_inputs, [])
        for input in all_inputs:
            if not input in comb_options:
                raise Exception(f'The desired input {input} is not defined. Options include {comb_options.keys()}')

        # Setup generic cross product of inputs: 
        gen_case_inputs = {}
        for i_group, group in enumerate(generic_case_inputs):
            first_array_len = len(comb_options[group[0]])
            for input in group:
                
                # Check that all inputs are of equal length
                if len(comb_options[input]) != first_array_len:
                    raise Exception(f'The input options in group {i_group} are not equal.  This group contains: {group}')

                gen_case_inputs[input] = {'vals': comb_options[input], 'group': i_group}
            
        # Generate generic case list
        generic_case_list, _ = CaseGen_General(gen_case_inputs,save_matrix=False)

        case_inputs_openfast = self.map_generic_to_openfast(generic_case_inputs, comb_options)
        self.openfast_case_inputs.append(case_inputs_openfast)
        return generic_case_list

    def gen_met_options(self, dlc_options, sea_state='normal'):
        '''
        Determine metocean options based on dlcs and sea state requested
        met_options includes wind, waves, seeds, etc.

        TODO: what input conditions are required of self?
        TODO: what is required in dlc_options?
        '''
        met_options = self.get_metocean(dlc_options)
        
        # Apply wave conditions based on wind speeds
        self.apply_sea_state(met_options,sea_state=sea_state)
        
        make_equal_length(met_options,'wind_speed')
        return met_options

    def apply_initial_conditions(self,generic_case_inputs, dlc_options, met_options):
        '''
        Add available initial conditions to generic_case_inputs and interpolate options based on initial_condition_table
        This is performed within each dlc generator function

        '''
        
        # These allowed_ics should map to input in openfast_input_map
        allowed_ics = ['pitch_initial','rot_speed_initial','tau1_const']

        
        if self.initial_condition_table and dlc_options['turbine_status'] == 'operating': # there is an IC table that's not empty
            dlc_wind_speeds = met_options['wind_speed']  # need to use met_options wind speeds because it accounts for seeds
            # find group with wind_speed
            wind_group = ['wind_speed' in gci for gci in generic_case_inputs].index(True)
            group = generic_case_inputs[wind_group]

            for initial_condition in allowed_ics:
                if initial_condition in self.initial_condition_table:
                    group.append(initial_condition)
                    dlc_options[initial_condition] = np.interp(dlc_wind_speeds,self.initial_condition_table['U'],self.initial_condition_table[initial_condition])
                
            # Apply new group
            generic_case_inputs[wind_group] = group

    def add_user_groups(self, generic_case_inputs, dlc_options):
        if dlc_options['user_group']:
            # If a list of user_groups, adds multiple groups
            if type(dlc_options['user_group']) == list:
                for user_group in dlc_options['user_group']:
                    # Add input to options, make new group
                    new_group = []
                    for key, value in user_group.items():
                        new_group.append(key)
                        dlc_options[key] = value

                    generic_case_inputs.append(new_group)
            else:
                # Add input to options, make new group
                new_group = []
                for key, value in dlc_options['user_group'].items():
                    new_group.append(key)
                    dlc_options[key] = value
                
                generic_case_inputs.append(new_group)
            
    
    def map_generic_to_openfast(self,generic_case_inputs, comb_options):
        case_inputs_openfast = {}
        for i_group, generic_case_group in enumerate(generic_case_inputs):
            for generic_input in generic_case_group:
                
                if generic_input not in self.openfast_input_map.keys():
                    raise Exception(f'The input {generic_input} does not map to an OpenFAST input key in openfast_input_map')

                openfast_input = self.openfast_input_map[generic_input]

                if type(openfast_input) == list:
                    # Apply to all list of openfast_inputs
                    for of_input in openfast_input:
                        case_inputs_openfast[of_input] = {'vals': comb_options[generic_input], 'group': i_group}

                else:
                    case_inputs_openfast[openfast_input] = {'vals': comb_options[generic_input], 'group': i_group}

        return case_inputs_openfast

    def generate_1p1(self, dlc_options):
        # Power production normal turbulence model - normal sea state
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = '1.1'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.35

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed','wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_AEP(self, dlc_options):
        # Same as DLC 1.1, but with a constant TI
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = 'AEP'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.35
        if 'TI_factor' not in dlc_options:
            raise Exception('A TI_factor must be set for the AEP DLC.')
        
        if 'turbulence_class' in dlc_options:
            self.IECturb.Turbulence_Class = dlc_options['turbulence_class']
            self.IECturb.setup()
            

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_1p2(self, dlc_options):
        # Power production normal turbulence model - fatigue loads
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = '1.2'
        dlc_options['sea_state'] = 'normal'

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)


    def generate_1p3(self, dlc_options):
        # Power production extreme turbulence model - ultimate loads
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = '1.3'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = '1ETM'

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_1p4(self, dlc_options):
        # Extreme coherent gust with direction change - ultimate loads
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = '1.4'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'ECD'
        dlc_options['direction'] = ['n', 'p']
        dlc_options['aero_mod'] = 1     # don't use unsteady aero
        
        dlc_options['azimuth_init'] = np.linspace(0.,120.,dlc_options['n_azimuth'],endpoint=False)

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]*len(dlc_options['azimuth_init'])


        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','aero_mod'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign','azimuth_init']) # group 2: 
        generic_case_inputs.append(['direction']) # group 3: 

        self.generate_cases(generic_case_inputs,dlc_options)

        
    def generate_1p5(self, dlc_options):
        # Extreme wind shear - ultimate loads
        
        # Get default options
        dlc_options.update(self.default_options)   
        
        # Handle DLC Specific options:
        dlc_options['label'] = '1.5'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'EWS'
        dlc_options['direction'] = ['p', 'n']
        dlc_options['shear'] = ['h', 'v']
        

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]
        
        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2: 
        generic_case_inputs.append(['direction']) # group 3: 
        generic_case_inputs.append(['shear']) # group 4: 

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_1p6(self, dlc_options):
        # Power production normal turbulence model - severe sea state

        # Get default options
        dlc_options.update(self.default_options)   
        
        # DLC Specific options:
        dlc_options['label'] = '1.6'
        dlc_options['sea_state'] = 'severe'
        dlc_options['IEC_WindType'] = 'NTM'

        # Set yaw_misalign, else default
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_2p1(self, dlc_options):
        # Power production plus loss of electrical network

        # Get default options
        dlc_options.update(self.default_options)   
        
        # DLC Specific options:
        dlc_options['label'] = '2.1'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35  # For fault cases, psf depends on the mean-time between faults
        
        if 'genfault_time' not in dlc_options:
            raise Exception('genfault_time must be set for the DLC 2.1')

        # azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0.,120.,dlc_options['n_azimuth'],endpoint=False)

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []

        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','genfault_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['azimuth_init']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_2p2(self, dlc_options):
        # Power production plus occurrence of fault

        # Get default options
        dlc_options.update(self.default_options)   
        
        # DLC Specific options:
        dlc_options['label'] = '2.2'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35  # For fault cases, psf depends on the mean-time between faults

        # azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0.,120.,dlc_options['n_azimuth'],endpoint=False)


        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        group0 = ['total_time','transient_time','wake_mod','wave_model']

        AnyFault = False
        if 'pitchfault_time1' in dlc_options:
            group0.extend(['pitchfault_time1','pitchfault_blade1pos'])
            AnyFault = True
        if 'pitchfault_time2' in dlc_options:
            group0.extend(['pitchfault_time2','pitchfault_blade2pos'])
            AnyFault = True
        if 'pitchfault_time3' in dlc_options:
            group0.extend(['pitchfault_time3','pitchfault_blade3pos'])
            AnyFault = True
        if 'yawfault_time' in dlc_options:
            group0.extend(['yawfault_time','yawfault_yawpos'])
            AnyFault = True
        
        if not AnyFault:
            raise Exception('yawfault or pitchfault for at least one blade must be set for dlc 2.2')

        generic_case_inputs.append(group0)  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['azimuth_init']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_2p3(self, dlc_options):
        # Power production plus occurrence of fault
        # Normal control system fault

        # Get default options
        dlc_options.update(self.default_options)   
        
        # DLC Specific options:
        dlc_options['label'] = '2.3'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'EOG'
        dlc_options['PSF'] = 1.1  # For fault cases, psf depends on the mean-time between faults
        
        if 'genfault_time' not in dlc_options:
            raise Exception('genfault_time must be set for the DLC 2.3')

        # azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0.,120.,dlc_options['n_azimuth'],endpoint=False)

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','genfault_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['azimuth_init']) # group 2

        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_3p1(self, dlc_options):
        # Start up - normal wind - fatigue
        # 
        
        # Get default options
        dlc_options.update(self.default_options)      
        
        # DLC Specific options:
        dlc_options['label'] = '3.1'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['pitch_initial'] = 90.
        dlc_options['turbine_status'] = 'parked-idling'     # initial turbine status is what matters here

        # Specify startup time for this case
        if dlc_options['startup_time'] > dlc_options['analysis_time']:
            raise Exception(f"DLC 3.1 was selected, but the startup_time ({dlc_options['startup_time']}) option is greater than the analysis_time ({dlc_options['analysis_time']})")
        else:
            dlc_options['startup_time'] = dlc_options['startup_time']

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','startup_time','wake_mod','wave_model','pitch_initial'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        # generic_case_inputs.append(['azimuth_init']) # group 2
      
        self.generate_cases(generic_case_inputs,dlc_options)

    
    def generate_5p1(self, dlc_options):
        # Power production normal turbulence model - shutdown with varous azimuth initial conditions
        # 
        
        # Get default options
        dlc_options.update(self.default_options)      
        
        # DLC Specific options:
        dlc_options['label'] = '5.1'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['final_blade_pitch'] = 90.

        # Time options, set defaults if not provided
        if dlc_options['analysis_time'] == self.dlc_schema['analysis_time']['default']: 
            dlc_options['analysis_time'] = 600

        if dlc_options['shutdown_time'] == self.dlc_schema['shutdown_time']['default']:
            dlc_options['shutdown_time'] = 300

        
        # azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0.,120.,dlc_options['n_azimuth'],endpoint=False)

        # Specify shutdown time for this case
        if dlc_options['shutdown_time'] > dlc_options['analysis_time']:
            raise Exception(f"DLC 5.1 was selected, but the shutdown_time ({dlc_options['shutdown_time']}) option is greater than the analysis_time ({dlc_options['analysis_time']})")
        else:
            dlc_options['shutdown_time'] = dlc_options['shutdown_time']

        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','shutdown_time','wake_mod','wave_model','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['azimuth_init']) # group 2
      
        self.generate_cases(generic_case_inputs,dlc_options)


    def generate_6p1(self, dlc_options):
        # Parked (standing still or idling) - extreme wind model 50-year return period - ultimate loads
        # extra dlc_options: 
        # yaw_misalign: default = [-8,8]

        # Get default options
        dlc_options.update(self.default_options)

        # DLC Specific options:
        dlc_options['label'] = '6.1'
        dlc_options['sea_state'] = '50-year'
        dlc_options['IEC_WindType'] = self.wind_speed_class_num + 'EWM50'

        # yaw_misalign
        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [-8,8]

        if not dlc_options['wind_speed']:
            dlc_options['wind_speed'] = [self.V_e50]

        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.


        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                                    'rot_speed_initial','shutdown_time','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2
      
        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_6p2(self, dlc_options):
        # Parked (standing still or idling) - extreme wind model 50-year return period - ultimate loads
        # This is the same as DLC 6.1 in the 61400-3-1 standards, except there's a loss of electrical network.
        # In DLC 6.1, the generator is disabled already, so the only difference in 6.2 may be that users may want to simulate larger yaw misalignments
        # extra dlc_options: 
        # yaw_misalign: default = [-180 to 180]

        # Get default options
        dlc_options.update(self.default_options)

        # DLC Specific options:
        dlc_options['label'] = '6.2'
        dlc_options['sea_state'] = '50-year'
        dlc_options['IEC_WindType'] = self.wind_speed_class_num + 'EWM50'
        dlc_options['PSF'] = 1.1

        # yaw_misalign
        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = np.arange(-180+15,180+15,15).tolist()     # -180 is not valid in OF

        if not dlc_options['wind_speed']:
            dlc_options['wind_speed'] = [self.V_e50]

        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.


        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                                    'rot_speed_initial','shutdown_time','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2
      
        self.generate_cases(generic_case_inputs,dlc_options)


    def generate_6p3(self, dlc_options):
        # Parked (standing still or idling) - extreme wind model 1-year return period - ultimate loads, usually larger (20 deg) yaw offset

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = '6.3'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = self.wind_speed_class_num + 'EWM1'

        # Set dlc-specific options, like yaw_misalign, initial azimuth
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [-20.,20.]

        if not dlc_options['wind_speed']:
            dlc_options['wind_speed'] = [self.V_e1]
            
        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                                    'rot_speed_initial','shutdown_time','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling        
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)


    def generate_6p4(self, dlc_options):
        # Parked (standing still or idling) - normal turbulence model - fatigue loads

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = '6.4'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        # Set wind speeds to DLC spec if not defined by the user
        if len(dlc_options['wind_speed']) == 0:
            dlc_options['wind_speed'] = np.arange(self.ws_cut_in, 0.7 * self.V_ref, dlc_options['ws_bin_size'])
            # Include V_ref
            if dlc_options['wind_speed'][-1] != self.V_ref:
                dlc_options['wind_speed'] = np.append(dlc_options['wind_speed'], self.V_ref)
            dlc_options['wind_speed'] = dlc_options['wind_speed'].tolist()

        # Set dlc-specific options, like yaw_misalign, initial azimuth
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0.]

        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                                    'rot_speed_initial','shutdown_time','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_7p1(self, dlc_options):
        # Parked (standing still or idling) - extreme wind model 1-year return period - ultimate loads, usually larger (20 deg) yaw offset

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = '7.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = self.wind_speed_class_num + 'EWM1'
        dlc_options['PSF'] = 1.1

        # Set dlc-specific options, like yaw_misalign, initial azimuth
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0.] # default

        if not dlc_options['wind_speed']:
            dlc_options['wind_speed'] = [self.V_e1]
            
        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []

        group0 = ['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                'rot_speed_initial','shutdown_time','final_blade_pitch'] # group 0, (usually constants) turbine variables, DT, aero_modeling
        
        if 'pitchfault_time1' in dlc_options:
            group0.extend(['pitchfault_time1','pitchfault_blade1pos'])
        if 'pitchfault_time2' in dlc_options:
            group0.extend(['pitchfault_time2','pitchfault_blade2pos'])
        if 'pitchfault_time3' in dlc_options:
            group0.extend(['pitchfault_time3','pitchfault_blade3pos'])
        if 'yawfault_time' in dlc_options:
            group0.extend(['yawfault_time','yawfault_yawpos'])
        if 'genfault_time' in dlc_options:
            group0.extend(['genfault_time'])

        generic_case_inputs.append(group0)  
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_7p2(self, dlc_options):
        # Parked (standing still or idling) - normal turbulence model - fatigue loads

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = '7.2'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        
        # Set wind speeds to DLC spec if not defined by the user
        if len(dlc_options['wind_speed']) == 0:
            dlc_options['wind_speed'] = np.arange(0,self.ws_cut_out, dlc_options['ws_bin_size'])
            dlc_options['wind_speed'] = dlc_options['wind_speed'].tolist()

        # Set dlc-specific options, like yaw_misalign, initial azimuth
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0.]

        # parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model','pitch_initial',
                                    'rot_speed_initial','shutdown_time','final_blade_pitch'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_freedecay(self,dlc_options):
        # Describe the new design load case

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = 'freedecay'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'EOG'  # let's make a dummy EOG until we have steady wind input (cheaper than NTM, inflow should be disabled)
        dlc_options['wind_speed'] = [0]
        dlc_options['turbine_status'] = 'parked-still'

        # Disable generator, inflow, and aerodynamics by default
        dlc_options['generator_dof'] = False
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['compute_aerodynamics'] = dlc_options.get('compute_aerodynamics',0)     # Use user input, otherwise disabled
        dlc_options['compute_inflow'] = dlc_options.get('compute_inflow',0) # Use user input, otherwise disabled
        dlc_options['compute_control'] = dlc_options.get('compute_control',0) # Use user input, otherwise disabled
        dlc_options['wave_model'] = 0       

        # Zero platform ICs by default
        platform_ics = [
            'initial_platform_surge',
            'initial_platform_sway',
            'initial_platform_heave',
            'initial_platform_roll',
            'initial_platform_pitch',
            'initial_platform_yaw',
        ]
        for ptfm_ic in platform_ics:
            if ptfm_ic not in dlc_options:
                dlc_options[ptfm_ic] = 0

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []
        generic_case_inputs.append([
            'total_time',
            'transient_time',
            'wake_mod',
            'wave_model',
            'generator_dof',
            'rot_speed_initial',
            'initial_platform_surge',
            'initial_platform_sway',
            'initial_platform_heave',
            'initial_platform_roll',
            'initial_platform_pitch',
            'initial_platform_yaw',
            'compute_aerodynamics',
            'compute_inflow',
            'compute_control',
            ])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        
        # Don't need wind/waves/yaw
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)

    def generate_new_dlc(self,dlc_options):
        # Describe the new design load case

        # Get default options
        dlc_options.update(self.default_options)   
        
        # Set DLC Specific options:
        # These three are required
        dlc_options['label'] = '1.6'
        dlc_options['sea_state'] = 'severe'
        dlc_options['IEC_WindType'] = 'NTM'

        # Set dlc-specific options, like yaw_misalign, initial azimuth
        if 'yaw_misalign' in dlc_options:
            dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
        else: # default
            dlc_options['yaw_misalign'] = [0]

        # DLC-specific: define groups
        # Groups are dependent variables, the cases are a cross product of the independent groups
        # The options in each group should have the same length
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','wake_mod','wave_model'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed', 'wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['yaw_misalign']) # group 2

        # This function does the rest and generates the individual cases for each DLC
        self.generate_cases(generic_case_inputs,dlc_options)

def make_equal_length(option_dict,target_name):
    '''
    This function will set the length of all the option_dicts to that of option_dict[target_name] if it's a scalar
    '''
    target_len = len(option_dict[target_name])
    for key in option_dict:
        if len(option_dict[key]) == 1:
            if isinstance(option_dict[key], np.ndarray):
                option_dict[key] = np.tile(option_dict[key],target_len)
            elif isinstance(option_dict[key], list):
                option_dict[key] = option_dict[key] * target_len
            else:
                raise Exception(f'Cannot coerce {key} into an array with same length as wind_speed')
            
            # re-normalize probabilities
            if key == 'probabilities':
                option_dict['probabilities'] /= target_len
            

def combine_options(*dicts):
    """
    Combine option dictionarys and do standard processing, 
    like removing numpy and turning everything into lists for case_inputs
    
    Args:
        *dicts: Variable number of dictionaries.
        
    Returns:
        dict: Combined dictionary.
    """
    comb_options = {}
    for d in dicts:
        comb_options.update(d)

    comb_options = remove_numpy(comb_options)
    
    # Make all options a list
    for opt in comb_options:
        if not isinstance(comb_options[opt], list):  # if not a list
            comb_options[opt] = [comb_options[opt]]

    return comb_options

def is_list_of_lists(lst):
    """Checks if a given object is a list of lists."""

    if not isinstance(lst, list):
        return False

    for item in lst:
        if not isinstance(item, list):
            return False

    return True

if __name__ == "__main__":

    # Wind turbine inputs that will eventually come in from somewhere
    ws_cut_in = 4.
    ws_cut_out = 25.
    ws_rated = 10.
    wind_speed_class = 'I'
    wind_turbulence_class = 'B'

    # Load modeling options file
    weis_dir                = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
    fname_modeling_options = os.path.join(weis_dir , "examples", "05_IEA-3.4-130-RWT", "modeling_options.yaml")
    modeling_options = sch.load_modeling_yaml(fname_modeling_options)

    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']

    # Initialize the generator
    fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(ws_cut_in, ws_cut_out, ws_rated, wind_speed_class, wind_turbulence_class, fix_wind_seeds, fix_wave_seeds, metocean)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

    # print(dlc_generator.cases[1].URef)
    # print(dlc_generator.n_cases)

    FAST_runDirectory = '/Users/dzalkind/Tools/WEIS-DLC/examples/05_IEA-3.4-130-RWT/outputs/05_DLC15_new_setup/openfast_runs'
    FAST_InputFile = 'weis_job'

    case_list_all = []
    for case_inputs in dlc_generator.openfast_case_inputs:
        case_list, case_name = CaseGen_General(case_inputs, FAST_runDirectory, FAST_InputFile)
        print('here')


