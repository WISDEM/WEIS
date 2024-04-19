import numpy as np
import os
import weis.inputs as sch
from weis.dlc_driver.turbulence_models import IEC_TurbulenceModels
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.FileTools import remove_numpy

# TODO: not sure where this should live, so it's a global for now
# Could it be an input yaml?
openfast_input_map = {
    # Generic name: OpenFAST input (list if necessary)
    'total_time': ("Fst","TMax"),
    'transient_time': ("Fst","TStart"),
    
    'WindFile_type': ("InflowWind","WindType"),
    'wind_speeds': ("InflowWind","HWindSpeed"),
    'WindFile_name': ("InflowWind","FileName_BTS"),
    'WindFile_name': ("InflowWind","Filename_Uni"),
    'rotorD': ("InflowWind","RefLength"),
    'WindHd': ("InflowWind","PropagationDir"),
    'hub_height': ("InflowWind","RefHt_Uni"),
    
    'rot_speed_initial': ("ElastoDyn","RotSpeed"),
    'pitch_initial': [("ElastoDyn","BlPitch1"),("ElastoDyn","BlPitch2"),("ElastoDyn","BlPitch3")],
    'azimuth_init': ("ElastoDyn","Azimuth"),
    'yaw_misalignment': ("ElastoDyn","NacYaw"),
    
    'wave_Hs': ("HydroDyn","WaveHs"),
    'wave_Tp': ("HydroDyn","WaveTp"),
    'WaveHd': ("HydroDyn","WaveDir"),
    'WaveGamma': ("HydroDyn","WavePkShp"),
    'WaveSeed1': ("HydroDyn","WaveSeed1"),
    
    'shutdown_time': [("ServoDyn","TPitManS1"),("ServoDyn","TPitManS2"),("ServoDyn","TPitManS3")],
    
    'aero_mod': ("AeroDyn15","AFAeroMod"),
    'wake_mod': ("AeroDyn15","WakeMod"),
    'tau1_const': ("AeroDyn15","tau1_const"),
    'DTfvw': ("AeroDyn15","OLAF","DTfvw"),
    'nNWPanels': ("AeroDyn15","OLAF","nNWPanels"),
    'nNWPanelsFree': ("AeroDyn15","OLAF","nNWPanelsFree"),
    'nFWPanels': ("AeroDyn15","OLAF","nFWPanels"),
    'nFWPanelsFree': ("AeroDyn15","OLAF","nFWPanelsFree"),

    # 'dlc_label': ("DLC","Label"),
    # 'wind_seed': ("DLC","WindSeed"),
    # 'wind_speeds': ("DLC","MeanWS"),

    # TODO: where should turbsim live?
    # These aren't actually used to generate turbsim, the generic inputs are used
    # However, I think it's better to be over-thorough and check that inputs are applied than the uncertainty of not checking any
    'rand_seeds': ("TurbSim", "RandSeed1"),
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
    OF_dlccaseinputvars = {("InflowWind","WindType"):0,
        ("ElastoDyn","RotSpeed"):0,
        ("ElastoDyn","BlPitch1"):90.0,
        ("ElastoDyn","BlPitch2"):90.0,
        ("ElastoDyn","BlPitch3"):90.0,
        ("ServoDyn","TPitManS1"):9999,
        ("ElastoDyn","Azimuth"):0,
        ("InflowWind","PropagationDir"):0,
        ("HydroDyn","WaveHs"):0,
        ("HydroDyn","WaveTp"):0,
        ("HydroDyn","WaveDir"):0,
        ("HydroDyn","WavePkShp"):0,
        ("HydroDyn","WaveSeed1"):0,
        ("Fst","TMax"):0,
        ("Fst","TStart"):0,
        ("DLC","Label"):0,
        ("DLC","WindSeed"):0,
        ("DLC","MeanWS"):0,
        ("ElastoDyn","NacYaw"):0,
        ("AeroDyn15","tau1_const"):0,
        ("AeroDyn15","OLAF","DTfvw"):0,
        ("AeroDyn15","OLAF","nNWPanels"):0,
        ("AeroDyn15","OLAF","nNWPanelsFree"):0,
        ("AeroDyn15","OLAF","nFWPanels"):0,
        ("AeroDyn15","OLAF","nFWPanelsFree"):0,
    }
    

    def __init__(self, ws_cut_in=4.0, ws_cut_out=25.0, ws_rated=10.0, wind_speed_class = 'I',
                wind_turbulence_class = 'B', fix_wind_seeds=True, fix_wave_seeds=True, metocean={}):
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
        self.wave_Hs50 = np.array([metocean['wave_height50']])
        self.wave_Tp50 = np.array([metocean['wave_period50']])
        self.wave_Hs1 = np.array([metocean['wave_height1']])
        self.wave_Tp1 = np.array([metocean['wave_period1']])

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
            wind_speeds = np.array( [float(m) for m in options['wind_speed']] )
        else:
            wind_speeds = np.arange(self.ws_cut_in, self.ws_cut_out+0.5*options['ws_bin_size'], options['ws_bin_size'])
            if wind_speeds[-1] != self.ws_cut_out:
                wind_speeds = np.append(wind_speeds, self.ws_cut_out)

        return wind_speeds

    def get_wind_seeds(self, options, wind_speeds):
        if 'yaw_misalign' in options:
            n_yaw_ms = len(options['yaw_misalign'])
        else:
            n_yaw_ms = 1
        
        if len(options['wind_seed']) > 0:
            wind_seeds = np.array( [int(m) for m in options['wind_seed']] )
        else:
            wind_seeds = self.rng_wind.integers(2147483648, size=options['n_seeds']*len(wind_speeds) * n_yaw_ms, dtype=int)
            wind_speeds = np.repeat(wind_speeds, options['n_seeds'] * n_yaw_ms)

        return wind_speeds, wind_seeds

    def get_wave_seeds(self, options, wind_speeds):
        if len(options['wave_seeds']) > 0:
            wave_seeds = np.array( [int(m) for m in options['wave_seeds']] )
        else:
            wave_seeds = self.rng_wave.integers(2147483648, size=len(wind_speeds), dtype=int)

        return wave_seeds

    def get_wind_heading(self, options):
        if len(options['wind_heading']) > 0:
            wind_heading = np.array( [float(m) for m in options['wind_heading']] )
        else:
            wind_heading = np.array([])
        return wind_heading

    def get_wave_Hs(self, options):
        if len(options['wave_height']) > 0:
            wave_Hs = np.array( [float(m) for m in options['wave_height']] )
        else:
            wave_Hs = np.array([])
        return wave_Hs

    def get_wave_Tp(self, options):
        if len(options['wave_period']) > 0:
            wave_Tp = np.array( [float(m) for m in options['wave_period']] )
        else:
            wave_Tp = np.array([])
        return wave_Tp

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
        wind_speeds, wind_seeds = self.get_wind_seeds(options, wind_speeds_indiv)
        wave_seeds = self.get_wave_seeds(options, wind_speeds)
        wind_heading = self.get_wind_heading(options)
        wave_Hs = self.get_wave_Hs(options)
        wave_Tp = self.get_wave_Tp(options)
        wave_gamma = self.get_wave_gamma(options)
        wave_heading = self.get_wave_heading(options)
        probabilities = self.get_probabilities(options)

        if len(wind_seeds) > 1 and len(wind_seeds) != len(wind_speeds):
            raise Exception("The vector of wind_seeds must have either length=1 or the same length of wind speeds")
        if len(wind_heading) > 1 and len(wind_heading) != len(wind_speeds):
            raise Exception("The vector of wind_heading must have either length=1 or the same length of wind speeds")
        if len(wave_seeds) > 1 and len(wave_seeds) != len(wind_speeds):
            raise Exception("The vector of wave seeds must have the same length of wind speeds or not defined")
        if len(wave_Hs) > 1 and len(wave_Hs) != len(wind_speeds):
            raise Exception("The vector of wave heights must have either length=1 or the same length of wind speeds")
        if len(wave_Tp) > 1 and len(wave_Tp) != len(wind_speeds):
            raise Exception("The vector of wave periods must have either length=1 or the same length of wind speeds")
        if len(wave_gamma) > 1 and len(wave_gamma) != len(wind_speeds):
            raise Exception("The vector of wave_gamma must have either length=1 or the same length of wind speeds")
        if len(wave_heading) > 1 and len(wave_heading) != len(wind_speeds):
            raise Exception("The vector of wave heading must have either length=1 or the same length of wind speeds")
        if len(probabilities) > 1 and len(probabilities) != len(wind_speeds):
            raise Exception("The vector of probabilities must have either length=1 or the same length of wind speeds")
        if abs(sum(probabilities) - 1.) > 1.e-3:
            raise Exception("The vector of probabilities must sum to 1")
        
        metocean_case_info = {}
        metocean_case_info['wind_speeds'] = wind_speeds
        metocean_case_info['rand_seeds'] = wind_seeds
        metocean_case_info['wave_seeds'] = wave_seeds
        metocean_case_info['wind_heading'] = wind_heading
        metocean_case_info['wave_Hs'] = wave_Hs
        metocean_case_info['wave_Tp'] = wave_Tp
        # metocean_case_info['current_speeds'] = current_speeds
        metocean_case_info['wave_gamma'] = wave_gamma
        metocean_case_info['wave_heading'] = wave_heading
        metocean_case_info['probabilities'] = probabilities       
        # metocean_case_info['current_std'] = self.mo_current_std       
        
        return metocean_case_info


    def generate(self, label, options):
        known_dlcs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 5.1, 6.1, 6.3, 6.4, 6.5, 12.1]
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

    def generate_custom(self, options):
        pass

    def generate_1p1(self, options):
        # Power production normal turbulence model - ultimate loads
        wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(wind_speeds) / len(yaw_misalign)))
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.yaw_misalign = yaw_misalign_deg[i_WiSe]
            idlc.turbulent_wind = True
            idlc.label = '1.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

        self.n_ws_dlc11 = len(np.unique(wind_speeds))

    def generate_1p2(self, options):
        # Power production normal turbulence model - fatigue loads
        wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, probabilities = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the fatigue analysis
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_F)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_F)
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.probability = probabilities[i_WaH]
            idlc.turbulent_wind = True
            idlc.label = '1.2'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def generate_1p3(self, options):
        # Power production extreme turbulence model - ultimate loads
        wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(wind_speeds) / len(yaw_misalign)))
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.yaw_misalign = yaw_misalign_deg[i_WiSe]
            idlc.IEC_WindType = self.wind_speed_class_num + 'ETM'
            idlc.turbulent_wind = True
            idlc.label = '1.3'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def generate_1p4(self, options):
        # Extreme coherent gust with direction change - ultimate loads
        wind_speeds, _, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        directions = ['n', 'p']
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(wind_speeds) / len(yaw_misalign)))
        # Set azimuth start positions, tile so length is same as wind_seeds
        azimuth_inits = np.tile(
            np.linspace(0.,120.,options['n_azimuth'],endpoint=False),
            int(len(wind_speeds)/options['n_azimuth'])
            )
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            for direction in directions:
                idlc = DLCInstance(options=options)
                idlc.URef = ws
                idlc.wave_seed1 = wave_seeds[i_WaSe]
                idlc.wind_heading = wind_heading[i_WiH]
                idlc.wave_height = wave_Hs[i_Hs]
                idlc.wave_period = wave_Tp[i_Tp]
                idlc.wave_gamma = wave_gamma[i_WG]
                idlc.wave_heading = wave_heading[i_WaH]
                idlc.yaw_misalign = yaw_misalign_deg[i_WaSe]
                idlc.azimuth_init = azimuth_inits[i_WaSe]
                idlc.IEC_WindType = 'ECD'
                idlc.turbulent_wind = False
                idlc.label = '1.4'
                if options['analysis_time'] > 0:
                    idlc.analysis_time = options['analysis_time']
                if options['transient_time'] >= 0:
                    idlc.transient_time = options['transient_time']
                idlc.direction_pn = direction
                idlc.OF_dlccaseinputs = {("Fst","TMax"):idlc.analysis_time + idlc.transient_time,
                                      ("Fst","TStart"):idlc.transient_time,
                                      ("InflowWind","HWindSpeed"):idlc.URef,
                                      ("InflowWind","PropagationDir"):idlc.wind_heading,
                                      ("ElastoDyn","Azimuth"):idlc.azimuth_init,
                                      ("ElastoDyn","NacYaw"):idlc.yaw_misalign,
                                      ("HydroDyn","WaveHs"):idlc.wave_height,
                                      ("HydroDyn","WaveTp"):idlc.wave_period,
                                      ("HydroDyn","WaveDir"):idlc.wave_heading,
                                      ("HydroDyn","WavePkShp"):idlc.wave_gamma,
                                      ("HydroDyn","WaveSeed1"):idlc.wave_seed1,
                                      ("DLC","Label"):idlc.label,
                                      #("DLC","WindSeed"):idlc.RandSeed1, # TODO AG: Check why windseed is not used for this DLC
                                      ("DLC","MeanWS"):idlc.URef,
                                      }
                self.cases.append(idlc)
                if len(wind_heading)>1:
                    i_WiH+=1
                if len(wave_gamma)>1:
                    i_WG+=1
                if len(wave_heading)>1:
                    i_WaH+=1
            # Same wave height, period, and seed per direction, check whether this is allowed or change seed sampling
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_seeds)>1:
                i_WaSe+=1

    def generate_1p5(self, options):
        # Extreme wind shear - ultimate loads
        wind_speeds, _, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(wind_speeds) / len(yaw_misalign)))
        directions = ['p', 'n']
        shears=['h', 'v']
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            for direction in directions:
                for shear in shears:
                    idlc = DLCInstance(options=options)
                    idlc.URef = ws
                    idlc.wave_seed1 = wave_seeds[i_WaSe]
                    idlc.wind_heading = wind_heading[i_WiH]
                    idlc.wave_height = wave_Hs[i_Hs]
                    idlc.wave_period = wave_Tp[i_Tp]
                    idlc.wave_gamma = wave_gamma[i_WG]
                    idlc.wave_heading = wave_heading[i_WaH]
                    idlc.yaw_misalign = yaw_misalign_deg[i_WaSe]
                    idlc.IEC_WindType = 'EWS'
                    idlc.turbulent_wind = False
                    idlc.label = '1.5'
                    if options['analysis_time'] > 0:
                        idlc.analysis_time = options['analysis_time']
                    if options['transient_time'] >= 0:
                        idlc.transient_time = options['transient_time']
                    idlc.sigma1 = self.IECturb.NTM(ws)
                    idlc.direction_pn = direction
                    idlc.shear_hv = shear
                    self.cases.append(idlc)
                    if len(wind_heading)>1:
                        i_WiH+=1
                    if len(wave_gamma)>1:
                        i_WG+=1
                    if len(wave_heading)>1:
                        i_WaH+=1
            # Same wave height, period, and seed per direction, check whether this is allowed or change seed sampling
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1

    def generate_1p6(self, options):
        # Power production normal turbulence model - severe sea state
        wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the severe sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_SSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_SSS)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(wind_speeds) / len(yaw_misalign)))
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.yaw_misalign = yaw_misalign_deg[i_WiSe]
            idlc.turbulent_wind = True
            idlc.label = '1.6'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def apply_wave_conditions(self,met_options):
        '''
        Apply waves based on the expected values provided in the metocean inputs
        Will use met_options as an input and modify that dict
        '''

        # TODO: separate normal and severe?
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(met_options['wave_Hs'])==0:
            met_options['wave_Hs'] = np.interp(met_options['wind_speeds'], self.mo_ws, self.mo_Hs_NSS)
        if len(met_options['wave_Tp'])==0:
            met_options['wave_Tp'] = np.interp(met_options['wind_speeds'], self.mo_ws, self.mo_Tp_NSS)

    def set_time_options(self, options):
        '''
        Handle time options and add total_time to dict
        Default for analysis and transient_time is 0
        '''
        if options['analysis_time'] > 0:
            options['analysis_time'] = options['analysis_time']
        if options['transient_time'] >= 0:
            options['transient_time'] = options['transient_time']
        options['total_time'] = options['analysis_time'] + options['transient_time']


    def generate_5p1(self, options):
        # Power production normal turbulence model - shutdown with varous azimuth initial conditions
        
        # These should always happen
        met_options = self.get_metocean(options)
        self.set_time_options(options)

        # Apply normal wave conditions based on wind speeds
        self.apply_wave_conditions(met_options)
        
        # Handle DLC Specific options:
        # azimuth starting positions
        options['azimuth_init'] = np.linspace(0.,120.,options['n_azimuth'],endpoint=False)

        # Specify shutdown time for this case
        if options['shutdown_time'] > options['analysis_time']:
            raise Exception(f"DLC 5.1 was selected, but the shutdown_time ({options['shutdown_time']}) option is greater than the analysis_time ({options['analysis_time']})")
        else:
            options['shutdown_time'] = options['shutdown_time']

        # All DLCs: Option processing 
        # TODO: figure out how to handle input options vs. options that are looped over, which need to be in a particular form
        make_equal_length(met_options,'wind_speeds')
        comb_options = combine_options(options,met_options)


        # DLC-specific: define groups
        # These options should be the same length and we will generate a matrix of all cases
        generic_case_inputs = []
        generic_case_inputs.append(['total_time','transient_time','shutdown_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
        generic_case_inputs.append(['wind_speeds','wave_Hs','wave_Tp', 'rand_seeds']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
        generic_case_inputs.append(['azimuth_init']) # group 2
      
        # All DLCs: Generate case list, both generic and OpenFAST specific
        case_list = gen_case_list(generic_case_inputs,comb_options)
        case_inputs_openfast = map_generic_to_openfast(generic_case_inputs, comb_options)
        self.openfast_case_inputs.append(case_inputs_openfast)

        # DLC specific: Make idlc for other parts of WEIS
        for i_case, case in enumerate(case_list):
            idlc = DLCInstance(options=options)

            # TODO: Figure out if there's a way to make the things below into generic_case_inputs
            idlc.turbulent_wind = True
            idlc.URef = case['wind_speeds']
            idlc.label = '5.1'
            idlc.RandSeed1 = case['rand_seeds']  # TODO: need this!!
            self.cases.append(idlc)


        # TODO: the majority of this method can be automated across DLCs.  What needs to be in each dlc generator function?
        # A few special options, like shutdown_time here
        # Maybe we need everything before the group setup to be an function, then everything after to start
        # We likely won't know until setting up more DLCs


    def generate_6p1(self, options):
        # Parked (standing still or idling) - extreme wind model 50-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        _, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-8., 8.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(wave_Hs)==0:
            wave_Hs = self.wave_Hs50
        if len(wave_Tp)==0:
            wave_Tp = self.wave_Tp50
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e50
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM50'
            idlc.turbulent_wind = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def generate_6p3(self, options):
        # Parked (standing still or idling) - extreme wind model 1-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        _, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-20., 20.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(wave_Hs)==0:
            wave_Hs = self.wave_Hs1
        if len(wave_Tp)==0:
            wave_Tp = self.wave_Tp1
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e1
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM1'
            idlc.turbulent_wind = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.3'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def generate_6p4(self, options):
        # Parked (standing still or idling) - normal turbulence model - fatigue loads
        wind_speeds = np.arange(self.ws_cut_in, 0.7 * self.V_ref, options['ws_bin_size'])
        if wind_speeds[-1] != self.V_ref:
            wind_speeds = np.append(wind_speeds, self.V_ref)
        wind_speeds, wind_seeds = self.get_wind_seeds(options, wind_speeds)
        wind_speeds = np.repeat(wind_speeds, options['n_seeds'])
        wave_seeds = self.get_wave_seeds(options, wind_speeds)
        _, _, _, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(wave_Hs)==0:
            wave_Hs = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(wave_Tp)==0:
            wave_Tp = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.turbulent_wind = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.4'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1

    def generate_6p5(self, options):
        # Parked (standing still or idling) - extreme wind model 500-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        _, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-8., 8.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(wave_Hs)==0:
            wave_Hs = self.wave_Hs1
        if len(wave_Tp)==0:
            wave_Tp = self.wave_Tp1
        # Counter for wind seed
        i_WiSe=0
        # Counters for wave conditions
        i_WaSe=0
        i_Hs=0
        i_Tp=0
        i_WiH=0
        i_WG=0
        i_WaH=0
        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e50 * 1.125
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = wind_seeds[i_WiSe]
            idlc.wave_seed1 = wave_seeds[i_WaSe]
            idlc.wind_heading = wind_heading[i_WiH]
            idlc.wave_height = wave_Hs[i_Hs]
            idlc.wave_period = wave_Tp[i_Tp]
            idlc.wave_gamma = wave_gamma[i_WG]
            idlc.wave_heading = wave_heading[i_WaH]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM1'
            idlc.turbulent_wind = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.5'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            if len(wind_seeds)>1:
                i_WiSe+=1
            if len(wave_seeds)>1:
                i_WaSe+=1
            if len(wind_heading)>1:
                i_WiH+=1
            if len(wave_Hs)>1:
                i_Hs+=1
            if len(wave_Tp)>1:
                i_Tp+=1
            if len(wave_gamma)>1:
                i_WG+=1
            if len(wave_heading)>1:
                i_WaH+=1
    
    def generate_12p1(self, options):
        # Pass through uniform wind input
        wind_speeds, _, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.label = '12.1'
            idlc.IEC_WindType = 'Custom'
            idlc.URef = wind_speeds
            idlc.turbulent_wind = False
            # idlc.wind_file = options['wind_file']
            if options['analysis_time'] >= 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']

            self.cases.append(idlc)

    # TODO: set up these methods with input information from openmdao_openfast
    def assign_initial_conditions(self):
        pass

    def assign_olaf_parameters(self):
        pass

    def generate_wind_inputs(self):
        pass


def make_equal_length(option_dict,target_name):
    '''
    This function will set the length of all the option_dicts to that of option_dict[target_name] if it's a scalar
    '''
    target_len = len(option_dict[target_name])
    for key in option_dict:
        if len(option_dict[key]) == 1:
            option_dict[key] = option_dict[key] * target_len

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

def gen_case_list(generic_case_inputs,comb_options):
    """
    Create generic case list from list of generic_case_inputs and comb_options
    Will apply elements of comb_options to generic_case_inputs, then create list of cases
    
    Args:
        *dicts: Variable number of dictionaries.
        
    Returns:
        dict: Combined dictionary.
    """
    # Setup generic cross product of inputs: TODO: make name of gen_case_inputs better
    gen_case_inputs = {}
    for i_group, group in enumerate(generic_case_inputs):
        first_array_len = len(comb_options[group[0]])
        for input in group:
            # Check that input is a valid option
            if not input in comb_options:
                raise Exception(f'The desired input {input} is not a valid option.  option includes {comb_options.keys()}')
            
            # Check that all inputs are of equal length
            if len(comb_options[input]) != first_array_len:
                raise Exception(f'The input options in group {i_group} are not equal.  This group contains: {group}')

            gen_case_inputs[input] = {'vals': comb_options[input], 'group': i_group}
        
    # Generate generic case list
    case_list, _ = CaseGen_General(gen_case_inputs)
    return case_list

def map_generic_to_openfast(generic_case_inputs, comb_options):
    case_inputs_openfast = {}
    for i_group, generic_case_group in enumerate(generic_case_inputs):
        for generic_input in generic_case_group:
            
            if generic_input not in openfast_input_map.keys():
                raise Exception(f'The input {generic_input} does not map to an OpenFAST input key in openfast_input_map')

            openfast_input = openfast_input_map[generic_input]

            if type(openfast_input) == list:
                # Apply to all list of openfast_inputs
                for of_input in openfast_input:
                    case_inputs_openfast[of_input] = {'vals': comb_options[generic_input], 'group': i_group}

            else:
                case_inputs_openfast[openfast_input] = {'vals': comb_options[generic_input], 'group': i_group}

    return case_inputs_openfast


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


