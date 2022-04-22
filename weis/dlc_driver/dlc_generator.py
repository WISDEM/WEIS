import numpy as np
import os
import weis.inputs as sch
from weis.dlc_driver.turbulence_models import IEC_TurbulenceModels


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

        if not options is None:
            self.default_turbsim_props(options)

    def default_turbsim_props(self, options):
        for key in options['turbulent_wind'].keys():
            setattr(self, key, options['turbulent_wind'][key])

class DLCGenerator(object):

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
        if len(options['wind_seed']) > 0:
            wind_seeds = np.array( [int(m) for m in options['wind_seed']] )
        else:
            wind_seeds = self.rng_wind.integers(2147483648, size=options['n_seeds']*len(wind_speeds), dtype=int)
            wind_speeds = np.repeat(wind_speeds, options['n_seeds'])

        return wind_speeds, wind_seeds

    def get_wave_seeds(self, options, wind_speeds):
        if len(options['wave_seed']) > 0:
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

        return wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, probabilities

    def generate(self, label, options):
        known_dlcs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 6.1, 6.2, 6.3, 6.4, 12.1]

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
                idlc.IEC_WindType = 'ECD'
                idlc.turbulent_wind = False
                idlc.label = '1.4'
                if options['analysis_time'] > 0:
                    idlc.analysis_time = options['analysis_time']
                if options['transient_time'] >= 0:
                    idlc.transient_time = options['transient_time']
                idlc.direction_pn = direction
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

    def generate_6p1(self, options):
        # Parked (standing still or idling) - extreme wind model 50-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        _, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        yaw_misalign_deg = np.array([-8., 8.] * options['n_seeds'])
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
        yaw_misalign_deg = np.array([-20., 20.] * options['n_seeds'])
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

    def generate_12p1(self, options):
        # Pass through uniform wind input
        wind_speeds, _, wave_seeds, wind_heading, wave_Hs, wave_Tp, wave_gamma, wave_heading, _ = self.get_metocean(options)
        idlc = DLCInstance(options=options)
        idlc.label = '12.1'
        idlc.IEC_WindType = 'Custom'
        idlc.wind_file = options['wind_file']
        if options['analysis_time'] >= 0:
            idlc.analysis_time = options['analysis_time']
        if options['transient_time'] >= 0:
            idlc.transient_time = options['transient_time']

        self.cases.append(idlc)


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

    print(dlc_generator.cases[5].URef)
    print(dlc_generator.n_cases)
