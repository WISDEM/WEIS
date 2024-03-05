import numpy as np
import os
import weis.inputs as sch
from weis.dlc_driver.turbulence_models import IEC_TurbulenceModels
import itertools

class DLCInstance(object):

    def __init__(self, options=None):
        # Set default DLC with empty properties
        self.URef = 0.0
        self.wind_heading = 0.0
        self.yaw_misalign = 0.0
        self.wave_height = 0.0
        self.wave_period = 0.0
        self.wave_heading = 0.0
        self.current = 0.0
        self.sea_level_offset = 0.0
        self.wave_gamma = 0.0
        self.probability = 0.0
        self.analysis_time = 600.
        self.transient_time = 120.
        self.shutdown_time = 9999.
        self.IEC_WindType = 'NTM'
        self.IECturbc = -1   # Default in weis modeling options
        self.turbine_status = 'operating'
        self.wave_spectrum = 'JONSWAP'
        self.turbulent = False
        self.direction_pn = '' # Positive (p) or negative (n), used for ECD
        self.shear_hv = '' # Horizontal (h) or vertical (v), used for EWS
        self.sigma1 = '' # Standard deviation of the wind
        self.RandSeed1 = 0
        self.wave_seed1 = 0
        self.constrained_wave = options['constrained_wave']
        self.label = '' # For 1.1/Custom
        self.wind_file = ''
        self.PSF = 1.35 # Partial Safety Factor
        self.azimuth_init = 0.0
        self.marine_turbine = False

        if not options is None:
            self.default_turbsim_props(options)

    def default_turbsim_props(self, options):
        for key in options['turbsim_inputs'].keys():
            setattr(self, key, options['turbsim_inputs'][key])

class MetoceanCounters(object):
    def __init__(self, metocean):
        # Counter for wind seed
        self.i_wind_speed = 0
        self.i_seed = 0
        self.i_Hs = 0
        self.i_Tp = 0
        self.i_heading = 0
        self.i_gamma = 0
        self.i_wave_heading = 0
        self.i_wave_seed = 0
        self.i_current = 0

        self.metocean = metocean

    def increment(self):
        if len(self.metocean['wind_speeds'])>1:
            self.i_wind_speed+=1
        if len(self.metocean['rand_seeds'])>1:
            self.i_seed+=1
        if len(self.metocean['wave_seeds'])>1:
            self.i_wave_seed+=1
        if len(self.metocean['wind_heading'])>1:
            self.i_heading+=1
        if len(self.metocean['wave_Hs'])>1:
            self.i_Hs+=1
        if len(self.metocean['wave_Tp'])>1:
            self.i_Tp+=1
        if len(self.metocean['wave_gamma'])>1:
            self.i_gamma+=1
        if len(self.metocean['wave_heading'])>1:
            self.i_wave_heading+=1
        if len(self.metocean['current_speeds'])>1:
            self.i_current+=1

class DLCGenerator(object):

    def __init__(self, metocean={}, **kwargs):

        # Default parameters
        self.ws_cut_in              = 4.0
        self.ws_cut_out             = 25.0
        self.ws_rated               = 10.0
        self.wind_speed_class       = "I"
        self.wind_turbulence_class  = "B"
        self.fix_wind_seeds         = True
        self.fix_wave_seeds         = True
        self.MHK                    = False

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass


        self.cases = []
        if self.fix_wind_seeds:
            self.rng_wind = np.random.default_rng(12345)
        else:
            self.rng_wind = np.random.default_rng()
        if self.fix_wave_seeds:
            self.rng_wave = np.random.default_rng(6789)
        else:
            self.rng_wave = np.random.default_rng()
        self.n_cases = 0
        self.n_ws_dlc11 = 0

        # Metocean conditions
        if self.MHK:
            self.mo_ws = metocean['current_speed']
        else:
            self.mo_ws = metocean['wind_speed']
        self.mo_Hs_NSS = metocean['wave_height_NSS']
        self.mo_Tp_NSS = metocean['wave_period_NSS']
        self.mo_Hs_F = metocean['wave_height_fatigue']
        self.mo_Tp_F = metocean['wave_period_fatigue']
        self.mo_Cu_NSS = metocean['current_NSS']
        self.mo_Cu_SSS = metocean['current_SSS']
        self.mo_Cu_F = metocean['current_fatigue']
        self.mo_Hs_SSS = metocean['wave_height_SSS']
        self.mo_Tp_SSS = metocean['wave_period_SSS']
        self.mo_current_std = metocean['current_std']
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
        self.current_50 = np.array([metocean['current_50']])
        self.wave_Hs1 = np.array([metocean['wave_height1']])
        self.wave_Tp1 = np.array([metocean['wave_period1']])
        self.current_1 = np.array([metocean['current_1']])


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

    def get_rand_seeds(self, options, wind_speeds):
        if 'yaw_misalign' in options:
            n_yaw_ms = len(options['yaw_misalign'])
        else:
            n_yaw_ms = 1
        
        if len(options['wind_seed']) > 0:
            wind_seeds = np.array( [int(m) for m in options['wind_seed']] )
        else:
            wind_seeds = self.rng_wind.integers(2147483648, size=options['n_seeds']*len(wind_speeds) * n_yaw_ms * options['n_azimuth'], dtype=int)
            wind_speeds = np.repeat(wind_speeds, options['n_seeds'] * n_yaw_ms * options['n_azimuth'])

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

    def get_current(self, options):
        if len(options['current_speed']) > 0:
            current_speeds = np.array( [float(m) for m in options['current_speed']] )
        else:
            if self.MHK:
                current_speeds = np.arange(self.ws_cut_in, self.ws_cut_out+0.5*options['ws_bin_size'], options['ws_bin_size'])
                if current_speeds[-1] != self.ws_cut_out:
                    current_speeds = np.append(current_speeds, self.ws_cut_out)
            else:
                current_speeds = np.array([0])

        return current_speeds

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
        current_speeds = self.get_current(options)
        
        # Generic speeds is index for waves
        if self.MHK:
            speeds = current_speeds
        else:
            speeds = wind_speeds_indiv

        speeds, rand_seeds = self.get_rand_seeds(options, speeds)
        wave_seeds = self.get_wave_seeds(options, speeds)
        wind_heading = self.get_wind_heading(options)
        wave_Hs = self.get_wave_Hs(options)
        wave_Tp = self.get_wave_Tp(options)
        wave_gamma = self.get_wave_gamma(options)
        wave_heading = self.get_wave_heading(options)
        probabilities = self.get_probabilities(options)

        if len(rand_seeds) > 1 and len(rand_seeds) != len(speeds):
            raise Exception("The vector of rand_seeds must have either length=1 or the same length of speeds")
        if len(wind_heading) > 1 and len(wind_heading) != len(speeds):
            raise Exception("The vector of wind_heading must have either length=1 or the same length of speeds")
        if len(wave_seeds) > 1 and len(wave_seeds) != len(speeds):
            raise Exception("The vector of wave seeds must have the same length of speeds or not defined")
        if len(wave_Hs) > 1 and len(wave_Hs) != len(speeds):
            raise Exception("The vector of wave heights must have either length=1 or the same length of speeds")
        if len(wave_Tp) > 1 and len(wave_Tp) != len(speeds):
            raise Exception("The vector of wave periods must have either length=1 or the same length of speeds")

        if not self.MHK:
            if len(current_speeds) > 1 and len(current_speeds) != len(speeds):
                raise Exception("The vector of currents must have either length=1 or the same length of speeds")
        if len(wave_gamma) > 1 and len(wave_gamma) != len(speeds):
            raise Exception("The vector of wave_gamma must have either length=1 or the same length of speeds")
        if len(wave_heading) > 1 and len(wave_heading) != len(speeds):
            print("DLCGenerator WARNING: The vector of wave heading must have either length=1 or the same length of speeds")
        if len(probabilities) > 1 and len(probabilities) != len(speeds):
            raise Exception("The vector of probabilities must have either length=1 or the same length of speeds")
        if abs(sum(probabilities) - 1.) > 1.e-3:
            raise Exception("The vector of probabilities must sum to 1")

        if self.MHK:
            current_speeds = speeds
        else:
            wind_speeds = speeds

        metocean_case_info = {}
        metocean_case_info['wind_speeds'] = speeds
        metocean_case_info['rand_seeds'] = rand_seeds
        metocean_case_info['wave_seeds'] = wave_seeds
        metocean_case_info['wind_heading'] = wind_heading
        metocean_case_info['wave_Hs'] = wave_Hs
        metocean_case_info['wave_Tp'] = wave_Tp
        metocean_case_info['current_speeds'] = current_speeds
        metocean_case_info['wave_gamma'] = wave_gamma
        metocean_case_info['wave_heading'] = wave_heading
        metocean_case_info['probabilities'] = probabilities       
        metocean_case_info['current_std'] = self.mo_current_std       
        
        return metocean_case_info

    def generate(self, label, options):
        universal_dlcs = [12.1]  # apply to both WT, MHK in same way

        if self.MHK:
            known_dlcs = [1.1, 1.2, 1.3, 6.1]
            gen_func_ext = '_mhk'
        else:
            known_dlcs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 5.1, 6.1, 6.3, 6.4, 6.5]
            gen_func_ext = ''

        # Get extreme wind speeds, TODO: make MHK replacement
        self.IECwind()

        found = False
        for ilab in known_dlcs + universal_dlcs:

            if float(label) in universal_dlcs: # drop extension
                gen_func_ext = ''

            func_name = f'generate_{ilab:.1f}{gen_func_ext}'.replace('.','p')

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
        metocean = self.get_metocean(options)
        # wind_speeds, wind_seeds, wave_seeds, wind_heading, wave_Hs, wave_Tp, current, wave_gamma, wave_heading, _, _
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_NSS)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_NSS)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)
            
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(metocean['wind_speeds']) / len(yaw_misalign)))
        
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.yaw_misalign = yaw_misalign_deg[mc.i_seed]
            idlc.turbulent = True
            idlc.label = '1.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)
            mc.increment()

        self.n_ws_dlc11 = len(np.unique(metocean['wind_speeds']))

    def generate_1p1_mhk(self, options):
        # Power production normal turbulence model - ultimate loads
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['current_speeds'], self.mo_ws, self.mo_Hs_NSS)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['current_speeds'], self.mo_ws, self.mo_Tp_NSS)

        # Resample/interpolate current_std (u_prime) to wind speeds in cases
        current_std = np.interp(metocean['current_speeds'], self.mo_ws, self.mo_current_std)

        # Turbulence intensity in percent
        TIs = 100 * current_std / metocean['current_speeds']

        mc = MetoceanCounters(metocean)

        for ws in metocean['current_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.turbulent = True
            idlc.IECturbc  = TIs[mc.i_seed]
            idlc.label = '1.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)

            mc.increment()
            

        self.n_ws_dlc11 = len(np.unique(metocean['current_speeds']))

    def generate_1p2(self, options):
        # Power production normal turbulence model - fatigue loads
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the fatigue analysis
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)
        
        # Counters for wave conditions
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.probability = metocean['probabilities'][mc.i_wave_heading]
            idlc.turbulent = True
            idlc.label = '1.2'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.
            self.cases.append(idlc)
            mc.increment()

    def generate_1p2_mhk(self, options):
        # Section 7.3.7.2, Table 8 in IEC docs

        # Get initial options
        metocean = self.get_metocean(options)

        # use Hs and Tp from cut-out
        metocean['wave_Tp'] = [self.mo_Tp_NSS[-1]]
        metocean['wave_Hs'] = [self.mo_Hs_NSS[-1]]

        # Wave headings from 0 to 360 in 30 deg increments
        if list(options['wave_heading']) == [0]:  # this is default, if it were selected, could be confusing
            options['wave_heading'] = np.arange(0,360,30)

        # Make cartesian product of current speeds (x number of seeds) with wave heading
        speed_heading_product = list(itertools.product(*[options['wave_heading'],metocean['current_speeds']]))
        metocean['wave_heading'] = [sh[0] for sh in speed_heading_product]
        metocean['current_speeds'] = [sh[1] for sh in speed_heading_product]


        options['n_seeds'] = 1  # Trick next function into giving us 1 seed for each current_speed, we generated the proper amount of current_speeds already
        metocean['current_speeds'], metocean['rand_seeds'] = self.get_rand_seeds(options, metocean['current_speeds'])
        metocean['wave_seeds'] = self.get_wave_seeds(options, metocean['current_speeds'])

        mc = MetoceanCounters(metocean)

        for ws in metocean['current_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.turbulent = True
            idlc.label = '1.2'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)

            mc.increment()

    def generate_1p3(self, options):
        # Power production extreme turbulence model - ultimate loads
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the fatigue analysis
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [0]
        yaw_misalign_deg = np.array(yaw_misalign * int(len(metocean['wind_speeds']) / len(yaw_misalign)))
        
        # Counters
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.yaw_misalign = yaw_misalign_deg[mc.i_seed]
            idlc.IEC_WindType = self.wind_speed_class_num + 'ETM'
            idlc.turbulent = True
            idlc.label = '1.3'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()

    def generate_1p3_mhk(self, options):
        # Power production extreme turbulence model - ultimate loads
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['current_speeds'], self.mo_ws, self.mo_Hs_NSS)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['current_speeds'], self.mo_ws, self.mo_Tp_NSS)

        mc = MetoceanCounters(metocean)

        for ws in metocean['current_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.turbulent = True
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.IEC_WindType = self.wind_speed_class_num + 'ETM'
            idlc.label = '1.3'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)

            mc.increment()

    def generate_1p4(self, options):
        # Extreme coherent gust with direction change - ultimate loads
        metocean = self.get_metocean(options)
        directions = ['n', 'p']
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)

        # Set azimuth start positions, tile so length is same as wind_seeds
        azimuth_inits = np.tile(
            np.linspace(0.,120.,options['n_azimuth'],endpoint=False),
            int(len(metocean['wind_speeds'])/options['n_azimuth']) * len(directions)
            )

        # Tile wave_seeds, Hs, Tp for each direction
        metocean['wave_seeds'] = np.tile(metocean['wave_seeds'],len(directions))
        metocean['wave_Hs'] = np.tile(metocean['wave_Hs'],len(directions))
        metocean['wave_Tp'] = np.tile(metocean['wave_Tp'],len(directions))
        metocean['current_speeds'] = np.tile(metocean['current_speeds'],len(directions))

        # Counters for wave conditions
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            for direction in directions:
                idlc = DLCInstance(options=options)
                idlc.URef = ws
                idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
                idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
                idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
                idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
                idlc.current = metocean['current_speeds'][mc.i_current]
                idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
                idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
                idlc.azimuth_init = azimuth_inits[mc.i_wave_seed]
                idlc.IEC_WindType = 'ECD'
                idlc.turbulent = False
                idlc.label = '1.4'
                if options['analysis_time'] > 0:
                    idlc.analysis_time = options['analysis_time']
                if options['transient_time'] >= 0:
                    idlc.transient_time = options['transient_time']
                idlc.direction_pn = direction
                self.cases.append(idlc)
                mc.increment()  # check this one

    def generate_1p5(self, options):
        # Extreme wind shear - ultimate loads
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)

        directions = ['p', 'n']
        shears=['h', 'v']
        # tile seeds and waves info for direction, shears
        metocean['wave_seeds'] = np.tile(metocean['wave_seeds'],4)
        metocean['wave_Hs'] = np.tile(metocean['wave_Hs'],4)
        metocean['wave_Tp'] = np.tile(metocean['wave_Tp'],4)
        metocean['current_speeds'] = np.tile(metocean['current_speeds'],4)

        # Counters for wave conditions
        mc = MetoceanCounters(metocean)
        for ws in metocean['wind_speeds']:
            for direction in directions:
                for shear in shears:
                    idlc = DLCInstance(options=options)
                    idlc.URef = ws
                    idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
                    idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
                    idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
                    idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
                    idlc.current = metocean['current_speeds'][mc.i_current]
                    idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
                    idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
                    idlc.IEC_WindType = 'EWS'
                    idlc.turbulent = False
                    idlc.label = '1.5'
                    if options['analysis_time'] > 0:
                        idlc.analysis_time = options['analysis_time']
                    if options['transient_time'] >= 0:
                        idlc.transient_time = options['transient_time']
                    idlc.sigma1 = self.IECturb.NTM(ws)
                    idlc.direction_pn = direction
                    idlc.shear_hv = shear
                    self.cases.append(idlc)
                    mc.increment()

    def generate_1p6(self, options):
        # Power production normal turbulence model - severe sea state
        metocean = self.get_metocean(options)        # If the user has not defined Hs and Tp, apply the metocean conditions for the severe sea state
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)
        # Counters for wave conditions
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.turbulent = True
            idlc.label = '1.6'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()

    def generate_5p1(self, options):
        # Power production normal turbulence model - severe sea state
        metocean = self.get_metocean(options)        # If the user has not defined Hs and Tp, apply the metocean conditions for the severe sea state
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Hs_F)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Tp_F)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(metocean['wind_speeds'], self.mo_ws, self.mo_Cu_F)
        

        # Set azimuth start positions, tile so length is same as wind_seeds
        azimuth_inits = np.tile(
            np.linspace(0.,120.,options['n_azimuth'],endpoint=False),
            int(len(metocean['wind_speeds'])/options['n_azimuth'])
            )

        # Counters for wave conditions
        mc = MetoceanCounters(metocean)

        for ws in metocean['wind_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.azimuth_init = azimuth_inits[mc.i_wave_seed]
            idlc.turbulent = True
            idlc.label = '5.1'
            idlc.turbine_status = 'operating-shutdown'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            if options['shutdown_time'] > options['analysis_time']:
                raise Exception(f"DLC 5.1 was selected, but the shutdown_time ({options['shutdown_time']}) option is greater than the analysis_time ({options['analysis_time']})")
            else:
                idlc.shutdown_time = options['shutdown_time']
            self.cases.append(idlc)
            mc.increment()


    def generate_6p1(self, options):
        # Parked (standing still or idling) - extreme wind model 50-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        metocean = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-8., 8.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = self.wave_Hs50
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = self.wave_Tp50
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = self.current_50
        # Counters for wave conditions
        mc = MetoceanCounters(metocean)

        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e50
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM50'
            idlc.turbulent = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()

    def generate_6p1_mhk(self, options):
        # Section 7.3.7.2, Table 8 in IEC docs

        # Get initial options
        metocean = self.get_metocean(options)

        # Use 50 year waves, current if not specified
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = self.wave_Hs50
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = self.wave_Tp50
            
            
        metocean['current_speeds'] = np.tile(self.current_50,options['n_seeds'])

        # Make cartesian product of current speeds (x number of seeds) with wave heading
        speed_heading_product = list(itertools.product(*[options['wave_heading'],metocean['current_speeds']]))
        metocean['wave_heading'] = [sh[0] for sh in speed_heading_product]
        metocean['current_speeds'] = [sh[1] for sh in speed_heading_product]

        options['n_seeds'] = 1  # Trick next function into giving us 1 seed for each current_speed, we generated the proper amount of current_speeds already
        metocean['current_speeds'], metocean['rand_seeds'] = self.get_rand_seeds(options, metocean['current_speeds'])
        metocean['wave_seeds'] = self.get_wave_seeds(options, metocean['current_speeds'])

        mc = MetoceanCounters(metocean)

        for ws in metocean['current_speeds']:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.turbulent = True
            idlc.label = '6.1'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            idlc.PSF = 1.2 * 1.25
            self.cases.append(idlc)

            mc.increment()

    def generate_6p3(self, options):
        # Parked (standing still or idling) - extreme wind model 1-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        metocean = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-20., 20.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = self.wave_Hs1
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = self.wave_Tp1
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = self.current_1
        # Counters for wave conditions
        mc = MetoceanCounters(metocean)
        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e1
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM1'
            idlc.turbulent = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.3'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()

    def generate_6p4(self, options):
        # Parked (standing still or idling) - normal turbulence model - fatigue loads
        wind_speeds = np.arange(self.ws_cut_in, 0.7 * self.V_ref, options['ws_bin_size'])
        if wind_speeds[-1] != self.V_ref:
            wind_speeds = np.append(wind_speeds, self.V_ref)
        wind_speeds, wind_seeds = self.get_rand_seeds(options, wind_speeds)
        wind_speeds = np.repeat(wind_speeds, options['n_seeds'])
        wave_seeds = self.get_wave_seeds(options, wind_speeds)
        metocean = self.get_metocean(options)
        # If the user has not defined Hs and Tp, apply the metocean conditions for the normal sea state
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = np.interp(wind_speeds, self.mo_ws, self.mo_Hs_NSS)
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = np.interp(wind_speeds, self.mo_ws, self.mo_Tp_NSS)
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = np.interp(wind_speeds, self.mo_ws, self.mo_Cu_F)
        else:
            metocean['current_speeds'] = metocean['current_speeds']

        mc = MetoceanCounters(metocean)

        for ws in wind_speeds:
            idlc = DLCInstance(options=options)
            idlc.URef = ws
            idlc.RandSeed1 = wind_seeds[mc.i_seed]
            idlc.wave_seed1 = wave_seeds[mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.turbulent = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.4'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()

    def generate_6p5(self, options):
        # Parked (standing still or idling) - extreme wind model 500-year return period - ultimate loads
        options['wind_speed'] = [50,50]  # set dummy, so wind seeds are correct
        metocean = self.get_metocean(options)
        # Set yaw_misalign, else default
        if 'yaw_misalign' in options:
            yaw_misalign = options['yaw_misalign']
        else: # default
            yaw_misalign = [-8., 8.]
        yaw_misalign_deg = np.array(yaw_misalign * options['n_seeds'])
        if len(metocean['wave_Hs'])==0:
            metocean['wave_Hs'] = self.wave_Hs1
        if len(metocean['wave_Tp'])==0:
            metocean['wave_Tp'] = self.wave_Tp1
        if metocean['current_speeds']==[0]:
            metocean['current_speeds'] = self.current_1
        else:
            metocean['current_speeds'] = metocean['current_speeds']
        # Counters for wave conditions
        mc = MetoceanCounters(metocean)
        for yaw_ms in yaw_misalign_deg:
            idlc = DLCInstance(options=options)
            if idlc.URef < 0:   # default is -1, this allows us to set custom V_50
                idlc.URef = self.V_e50 * 1.125
            idlc.yaw_misalign = yaw_ms
            idlc.RandSeed1 = metocean['rand_seeds'][mc.i_seed]
            idlc.wave_seed1 = metocean['wave_seeds'][mc.i_wave_seed]
            idlc.wind_heading = metocean['wind_heading'][mc.i_heading]
            idlc.wave_height = metocean['wave_Hs'][mc.i_Hs]
            idlc.wave_period = metocean['wave_Tp'][mc.i_Tp]
            idlc.current = metocean['current_speeds'][mc.i_current]
            idlc.wave_gamma = metocean['wave_gamma'][mc.i_gamma]
            idlc.wave_heading = metocean['wave_heading'][mc.i_wave_heading]
            idlc.IEC_WindType = self.wind_speed_class_num + 'EWM1'
            idlc.turbulent = True
            if idlc.turbine_status == 'operating':
                idlc.turbine_status = 'parked-still'
            idlc.label = '6.5'
            if options['analysis_time'] > 0:
                idlc.analysis_time = options['analysis_time']
            if options['transient_time'] >= 0:
                idlc.transient_time = options['transient_time']
            self.cases.append(idlc)
            mc.increment()
    
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
