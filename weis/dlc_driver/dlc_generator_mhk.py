"""
MHK DLC Generator for Tidal Energy Converters (TECs)
per IEC TS 62600-2:2019, Table 8

Subclass of DLCGenerator that overrides generate_* methods for MHK-specific
current models (NTM, ETM, OCM, ECM), flood/ebb/orientation-error directional
cases, and TEC wave conditions.
"""
import numpy as np
import logging

from weis.dlc_driver.dlc_generator import DLCGenerator

logger = logging.getLogger("wisdem/weis")


class MHKDLCGenerator(DLCGenerator):
    """DLC generator for MHK tidal-energy converters per IEC TS 62600-2."""

    def __init__(
            self,
            ws_cut_in=0.5,
            ws_cut_out=4.0,
            ws_rated=2.0,
            **kwargs
    ):
        # Set flow_key before calling super().__init__() so metocean validation uses 'current_speed'
        self.flow_key = 'current_speed'

        super().__init__(
            ws_cut_in=ws_cut_in,
            ws_cut_out=ws_cut_out,
            ws_rated=ws_rated,
            **kwargs
        )
        self.MHK = True

        # Add current_speed to openfast_input_map (if not already present from global)
        if 'current_speed' not in self.openfast_input_map:
            self.openfast_input_map['current_speed'] = [
                ("SeaState", "CurrDIV"),
                ("InflowWind", "HWindSpeed"),
            ]

        # Load MHK-specific metocean fields
        metocean = self.metocean
        self.wave_height5 = np.array([metocean.get('wave_height5', 0.)])
        self.wave_period5 = np.array([metocean.get('wave_period5', 0.)])
        self.wave_height_cutout = metocean.get('wave_height_cutout', 0.)
        self.current_peak_spring = metocean.get('current_peak_spring', 0.)
        self.current_mean_spring = metocean.get('current_mean_spring', 0.)

        # Tidal turbines have no IEC turbulence class (A/B/C): TI is tabulated
        # against current_speed in the metocean inputs instead
        self.current_TI_NTM = np.array(metocean.get('current_TI_NTM') or metocean.get('current_TI') or [], dtype=float)
        self.current_TI_ETM = np.array(metocean.get('current_TI_ETM') or [], dtype=float)
        if self.current_TI_ETM.size == 0:
            self.current_TI_ETM = self.current_TI_NTM
        for name, table in [('current_TI_NTM', self.current_TI_NTM), ('current_TI_ETM', self.current_TI_ETM)]:
            if table.size and table.size != len(self.metocean[self.flow_key]):
                raise Exception(f'The vector of metocean conditions {name} in the modeling options must have the same length of the tabulated {self.flow_key}s')

    def generate_cases(self, generic_case_inputs, dlc_options):
        '''Generate cases, then set turbulence from the metocean TI tables (TECs have no turbulence class)'''
        n_before = len(self.cases)
        super().generate_cases(generic_case_inputs, dlc_options)

        TI_table = self.current_TI_ETM if 'ETM' in dlc_options.get('IEC_WindType', '') else self.current_TI_NTM
        for idlc in self.cases[n_before:]:
            if not idlc.turbulent_wind:
                continue
            if dlc_options.get('TI'):
                TI = dlc_options['TI'] / 100     # user override is given in percent
            elif TI_table.size:
                TI = float(np.interp(idlc.URef, self.metocean[self.flow_key], TI_table))
            else:
                raise Exception(
                    'MHK turbines have no IEC turbulence class: provide current_TI_NTM '
                    '(and current_TI_ETM for ETM DLCs) in metocean_conditions, or TI in the DLC options.'
                )
            idlc.IECturbc = TI * 100
            # TurbSim's TIDAL/RIVER models are driven by UStar, not IECturbc
            # Realized TI will not match this request exactly: short records lose
            # low-frequency variance, and TurbSim's coherence factorization leaks
            # variance upward across a height-varying spectrum. Use
            # analysis_time >= 600 s. See docs/known_issues.rst.
            idlc.UStar = 1.2814 * TI * idlc.URef

    # ──────────────────────────────────────────────────────────────────────
    # DLC 1.1 — Normal operation, NTM current, OSS waves
    # IEC TS 62600-2, Table 8: NTM Uin≤U≤Uout, flood/ebb/OE, OSS H=Hm0
    # γf = 1.35 (ULS), 1.00 (FLS, SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_1p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '1.1'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # User specifies yaw_misalign in YAML for flood/ebb/OE cases
        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([])  # group 0
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])  # group 1
        generic_case_inputs.append(['yaw_misalign'])  # group 2

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 1.2 — Normal operation, NTM current, wave direction sweep
    # IEC TS 62600-2, Table 8: NTM Uin≤U≤Uout, flood/ebb/OE,
    #   OSS H=Hm0,out, wave dir 0°–360° in 30° steps
    # γf = 1.35 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_1p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '1.2'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Use cutout wave height Hm0,out for all current speeds
        met_options = self.gen_met_options(dlc_options, sea_state=dlc_options['sea_state'])
        if self.wave_height_cutout > 0:
            dlc_options['wave_height'] = self.wave_height_cutout * np.ones_like(met_options[self.flow_key])
        else:
            # Fallback: use max operational wave height
            max_wave_height = np.max(met_options['wave_height'])
            dlc_options['wave_height'] = max_wave_height * np.ones_like(met_options[self.flow_key])

        # Wave direction sweep 0°–360° in 30° steps
        default_wave_directions = np.arange(0, 360, 30)
        if len(dlc_options.get('wave_direction', [])) == 0:
            dlc_options['wave_direction'] = default_wave_directions

        # User specifies yaw_misalign in YAML for flood/ebb/OE cases
        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([])  # group 0
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])  # group 1
        generic_case_inputs.append(['wave_direction'])  # group 2
        generic_case_inputs.append(['yaw_misalign'])  # group 3

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 1.3 — Normal operation, ETM current, OSS waves
    # IEC TS 62600-2, Table 8: ETM Uin≤U≤Uout, flood/ebb/OE, OSS H=Hm0
    # γf = 1.35 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_1p3(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '1.3'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.35
        dlc_options['IEC_WindType'] = '1ETM'  # Extreme turbulence model
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # User specifies yaw_misalign in YAML for flood/ebb/OE cases
        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([])  # group 0
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])  # group 1
        generic_case_inputs.append(['yaw_misalign'])  # group 2

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 2.1 — Normal op + fault (grid loss / controller fault)
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.35 (ULS), 1.00 (FLS, SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_2p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '2.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        if 'genfault_time' not in dlc_options:
            raise Exception('genfault_time must be set for DLC 2.1')

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0., 120., dlc_options['n_azimuth'], endpoint=False)

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(['wake_mod', 'wave_model', 'genfault_time'])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['azimuth_init'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 2.2 — Normal op + safety-system / PTO fault
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.10 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_2p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '2.2'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.1
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0., 120., dlc_options['n_azimuth'], endpoint=False)

        group0 = ['wake_mod', 'wave_model']

        AnyFault = False
        if 'pitchfault_time1' in dlc_options:
            group0.extend(['pitchfault_time1', 'pitchfault_blade1pos'])
            AnyFault = True
        if 'pitchfault_time2' in dlc_options:
            group0.extend(['pitchfault_time2', 'pitchfault_blade2pos'])
            AnyFault = True
        if 'pitchfault_time3' in dlc_options:
            group0.extend(['pitchfault_time3', 'pitchfault_blade3pos'])
            AnyFault = True
        if 'yawfault_time' in dlc_options:
            group0.extend(['yawfault_time', 'yawfault_yawpos'])
            AnyFault = True

        if not AnyFault:
            raise Exception('yawfault or pitchfault for at least one blade must be set for DLC 2.2')

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(group0)
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['azimuth_init'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 2.3 — Normal op + accidental fault
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.10 (ALS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_2p3(self, dlc_options):

        raise NotImplementedError("DLC 2.3 (Normal op + accidental fault) requires additional set up for the model. Implement any accidental faults in the following template and then call self.generate_cases() with the appropriate generic_case_inputs and dlc_options.")

        dlc_options.update(self.default_options)

        dlc_options['label'] = '2.3'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.1
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        if 'genfault_time' not in dlc_options:
            raise Exception('genfault_time must be set for DLC 2.3')

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0., 120., dlc_options['n_azimuth'], endpoint=False)

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(['wake_mod', 'wave_model', 'genfault_time'])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['azimuth_init'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 3.1 — Start-up, NTM current, 1-yr waves
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.35 (ULS), 1.00 (FLS, SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_3p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '3.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['pitch_initial'] = 90.
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Startup options
        dlc_options['startup_mode'] = 1
        dlc_options['SU_FW_MinDuration'] = dlc_options.get('SU_FW_MinDuration', 40)
        dlc_options['SU_RotorSpeedThresh'] = dlc_options.get('SU_RotorSpeedThresh', 0.02)
        dlc_options['SU_RotorSpeedCornerFreq'] = dlc_options.get('SU_RotorSpeedCornerFreq', 0.51888)
        dlc_options['SU_LoadStages_N'] = dlc_options.get('SU_LoadStages_N', 2)
        dlc_options['SU_LoadStages'] = dlc_options.get('SU_LoadStages', "[0.4,0.8]")
        dlc_options['SU_LoadRampDuration'] = dlc_options.get('SU_LoadRampDuration', "[20,20]")
        dlc_options['SU_LoadHoldDuration'] = dlc_options.get('SU_LoadHoldDuration', "[20,20]")

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([
            "wake_mod", "wave_model", "pitch_initial",
            "startup_mode", "SU_FW_MinDuration", "SU_RotorSpeedThresh",
            "SU_RotorSpeedCornerFreq", "SU_LoadStages_N", "SU_LoadStages",
            "SU_LoadRampDuration", "SU_LoadHoldDuration",
        ])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 3.2 — Start-up, NTM current, cutout wave height
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm0,out
    # γf = 1.35 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_3p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '3.2'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['pitch_initial'] = 90.
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Use cutout wave height
        met_options = self.gen_met_options(dlc_options, sea_state=dlc_options['sea_state'])
        if self.wave_height_cutout > 0:
            dlc_options['wave_height'] = self.wave_height_cutout * np.ones_like(met_options[self.flow_key])
        else:
            max_wave_height = np.max(met_options['wave_height'])
            dlc_options['wave_height'] = max_wave_height * np.ones_like(met_options[self.flow_key])

        # Startup options
        dlc_options['startup_mode'] = 1
        dlc_options['SU_FW_MinDuration'] = dlc_options.get('SU_FW_MinDuration', 0)
        dlc_options['SU_RotorSpeedThresh'] = dlc_options.get('SU_RotorSpeedThresh', 0.02)
        dlc_options['SU_RotorSpeedCornerFreq'] = dlc_options.get('SU_RotorSpeedCornerFreq', 0.51888)
        dlc_options['SU_LoadStages_N'] = dlc_options.get('SU_LoadStages_N', 1)
        dlc_options['SU_LoadStages'] = dlc_options.get('SU_LoadStages', 1)
        dlc_options['SU_LoadRampDuration'] = dlc_options.get('SU_LoadRampDuration', 20)
        dlc_options['SU_LoadHoldDuration'] = dlc_options.get('SU_LoadHoldDuration', 20)

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([
            "wake_mod", "wave_model", "pitch_initial",
            "startup_mode", "SU_FW_MinDuration", "SU_RotorSpeedThresh",
            "SU_RotorSpeedCornerFreq", "SU_LoadStages_N", "SU_LoadStages",
            "SU_LoadRampDuration", "SU_LoadHoldDuration",
        ])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 4.1 — Normal shutdown, NTM current, 1-yr waves
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.35 (ULS), 1.00 (FLS, SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_4p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '4.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Shutdown options
        dlc_options['shutdown_mode'] = 1
        dlc_options['SD_EnableTime'] = 1

        if 'normal_shutdown_time' not in dlc_options:
            raise Exception('normal_shutdown_time must be set for DLC 4.1')
        elif dlc_options['normal_shutdown_time'] > dlc_options['analysis_time']:
            raise Exception(f"DLC 4.1: normal_shutdown_time ({dlc_options['normal_shutdown_time']}) > analysis_time ({dlc_options['analysis_time']})")

        group0 = ["wake_mod", "wave_model", "shutdown_mode", "SD_EnableTime", "normal_shutdown_time"]
        if 'SD_MaxTorqueRate' in dlc_options:
            group0.append('SD_MaxTorqueRate')
        if 'SD_MaxPitchRate' in dlc_options:
            group0.append('SD_MaxPitchRate')

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(group0)
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 4.2 — Normal shutdown, NTM current, cutout wave height
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm0,out
    # γf = 1.35 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_4p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '4.2'
        dlc_options['sea_state'] = 'normal'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        # Use cutout wave height
        met_options = self.gen_met_options(dlc_options, sea_state=dlc_options['sea_state'])
        if self.wave_height_cutout > 0:
            dlc_options['wave_height'] = self.wave_height_cutout * np.ones_like(met_options[self.flow_key])
        else:
            max_wave_height = np.max(met_options['wave_height'])
            dlc_options['wave_height'] = max_wave_height * np.ones_like(met_options[self.flow_key])

        # Shutdown options
        dlc_options['shutdown_mode'] = 1
        dlc_options['SD_EnableTime'] = 1

        if 'normal_shutdown_time' not in dlc_options:
            raise Exception('normal_shutdown_time must be set for DLC 4.2')

        group0 = ["total_time", "transient_time", "wake_mod", "wave_model",
                   "shutdown_mode", "SD_EnableTime", "normal_shutdown_time"]
        if 'SD_MaxTorqueRate' in dlc_options:
            group0.append('SD_MaxTorqueRate')
        if 'SD_MaxPitchRate' in dlc_options:
            group0.append('SD_MaxPitchRate')

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(group0)
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 5.1 — Emergency shutdown, NTM current, 1-yr waves
    # IEC TS 62600-2, Table 8: OCM/NTM, U_rated≤U≤U_out, NWH H=Hm1
    # γf = 1.35 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_5p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '5.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['final_blade_pitch'] = 90.
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Time options
        if dlc_options['analysis_time'] == self.dlc_schema['analysis_time']['default']:
            dlc_options['analysis_time'] = 600
        if dlc_options['shutdown_time'] == self.dlc_schema['shutdown_time']['default']:
            dlc_options['shutdown_time'] = 300

        # Azimuth starting positions
        dlc_options['azimuth_init'] = np.linspace(0., 120., dlc_options['n_azimuth'], endpoint=False)

        if dlc_options['shutdown_time'] > dlc_options['analysis_time']:
            raise Exception(f"DLC 5.1: shutdown_time ({dlc_options['shutdown_time']}) > analysis_time ({dlc_options['analysis_time']})")

        # Default current speeds: rated to cut-out
        dlc_options[self.flow_key] = dlc_options.get(self.flow_key) or [self.ws_rated, self.ws_cut_out]

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(['shutdown_time', 'wake_mod', 'wave_model', 'final_blade_pitch'])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['azimuth_init'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 6.1 — Parked/survival, ECM at peak spring current
    # IEC TS 62600-2, Table 8:
    #   6.1a: ECM U=peak spring, ESS H=Hm5, γf=1.35 (Extreme)
    #   6.1b: ECM U=peak spring, ESS H=Hm50, γf=1.35 (Extreme)
    # User selects sub-case via dlc_options['subcase'] = 'a' or 'b'
    # Default runs both as separate labels '6.1a' and '6.1b'
    # ──────────────────────────────────────────────────────────────────────
    def generate_6p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # User must specify peak spring current speed, or use metocean
        if not dlc_options.get(self.flow_key):
            dlc_options[self.flow_key] = [self.current_peak_spring]

        # Parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        subcases = dlc_options.get('subcase', ['a', 'b'])
        if isinstance(subcases, str):
            subcases = [subcases]

        for sc in subcases:
            dlc_opts_sc = dict(dlc_options)
            if sc == 'a':
                dlc_opts_sc['label'] = '6.1a'
                dlc_opts_sc['sea_state'] = '5-year'
            elif sc == 'b':
                dlc_opts_sc['label'] = '6.1b'
                dlc_opts_sc['sea_state'] = '50-year'
            else:
                raise Exception(f"DLC 6.1 subcase must be 'a' or 'b', got '{sc}'")

            generic_case_inputs = []
            generic_case_inputs.append(['wake_mod', 'wave_model', 'pitch_initial',
                                        'rot_speed_initial', 'shutdown_time', 'final_blade_pitch'])
            generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
            generic_case_inputs.append(['yaw_misalign'])

            self.generate_cases(generic_case_inputs, dlc_opts_sc)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 6.2 — Parked/survival + grid loss
    # IEC TS 62600-2, Table 8: worst combo from 6.1a/6.1b,
    #   U≤Uin or Uout≤U, γf=1.10 (Abnormal)
    # DZ note: we're not doing much about grid loss here
    # ──────────────────────────────────────────────────────────────────────
    def generate_6p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '6.2'
        dlc_options['sea_state'] = '50-year'
        dlc_options['PSF'] = 1.1
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # User must specify current speed; default to peak spring
        if not dlc_options.get(self.flow_key):
            dlc_options[self.flow_key] = [self.current_peak_spring]

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        # Parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        generic_case_inputs = []
        generic_case_inputs.append(['wake_mod', 'wave_model', 'pitch_initial',
                                    'rot_speed_initial', 'shutdown_time', 'final_blade_pitch'])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 7.1 — Parked + fault, ETM at mean spring current
    # IEC TS 62600-2, Table 8: ETM U=mean spring, ESS H=Hm1
    # γf = 1.10 (ULS), 1.00 (SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_7p1(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '7.1'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = '1ETM'
        dlc_options['PSF'] = 1.1
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # User must specify mean spring current speed, or use metocean
        if not dlc_options.get(self.flow_key):
            dlc_options[self.flow_key] = [self.current_mean_spring]

        # Parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        group0 = ['wake_mod', 'wave_model', 'pitch_initial',
                  'rot_speed_initial', 'shutdown_time', 'final_blade_pitch']

        if 'pitchfault_time1' in dlc_options:
            group0.extend(['pitchfault_time1', 'pitchfault_blade1pos'])
        if 'pitchfault_time2' in dlc_options:
            group0.extend(['pitchfault_time2', 'pitchfault_blade2pos'])
        if 'pitchfault_time3' in dlc_options:
            group0.extend(['pitchfault_time3', 'pitchfault_blade3pos'])
        if 'yawfault_time' in dlc_options:
            group0.extend(['yawfault_time', 'yawfault_yawpos'])

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(group0)
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # DLC 7.2 — Parked + grid loss, NTM operational current
    # IEC TS 62600-2, Table 8: NTM Uin≤U≤Uout, NWH H=Hm1
    # γf = 1.35 (ULS), 1.00 (FLS, SLS)
    # ──────────────────────────────────────────────────────────────────────
    def generate_7p2(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = '7.2'
        dlc_options['sea_state'] = '1-year'
        dlc_options['IEC_WindType'] = 'NTM'
        dlc_options['PSF'] = 1.35
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        # Parked options
        dlc_options['turbine_status'] = 'parked-idling'
        dlc_options['wake_mod'] = 0
        dlc_options['pitch_initial'] = 90.
        dlc_options['rot_speed_initial'] = 0.
        dlc_options['shutdown_time'] = 0.
        dlc_options['final_blade_pitch'] = 90.

        group0 = ['wake_mod', 'wave_model', 'pitch_initial',
                  'rot_speed_initial', 'shutdown_time', 'final_blade_pitch']

        if 'pitchfault_time1' in dlc_options:
            group0.extend(['pitchfault_time1', 'pitchfault_blade1pos'])
        if 'pitchfault_time2' in dlc_options:
            group0.extend(['pitchfault_time2', 'pitchfault_blade2pos'])
        if 'pitchfault_time3' in dlc_options:
            group0.extend(['pitchfault_time3', 'pitchfault_blade3pos'])
        if 'yawfault_time' in dlc_options:
            group0.extend(['yawfault_time', 'yawfault_yawpos'])

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append(group0)
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed', 'wave_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # AEP — Annual Energy Production; TI comes from the metocean table via
    # generate_cases, like every other MHK DLC
    # ──────────────────────────────────────────────────────────────────────
    def generate_AEP(self, dlc_options):
        dlc_options.update(self.default_options)

        dlc_options['label'] = 'AEP'
        dlc_options['sea_state'] = 'normal'
        dlc_options['PSF'] = 1.0
        dlc_options['wave_model'] = dlc_options.get('wave_model', 2)

        if 'yaw_misalign' not in dlc_options:
            dlc_options['yaw_misalign'] = [0]

        generic_case_inputs = []
        generic_case_inputs.append([])
        generic_case_inputs.append([self.flow_key, 'wave_height', 'wave_period', 'wind_seed'])
        generic_case_inputs.append(['yaw_misalign'])

        self.generate_cases(generic_case_inputs, dlc_options)

    # ──────────────────────────────────────────────────────────────────────
    # Wind-only DLCs — blocked for MHK TECs
    # ──────────────────────────────────────────────────────────────────────
    def generate_1p4(self, dlc_options):
        raise NotImplementedError("DLC 1.4 (ECD) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_1p5(self, dlc_options):
        raise NotImplementedError("DLC 1.5 (EWS) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_1p6(self, dlc_options):
        raise NotImplementedError("DLC 1.6 (SSS+NTM) is wind-dominated and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_2p4(self, dlc_options):
        raise NotImplementedError("DLC 2.4 (EOG + grid loss) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_3p3(self, dlc_options):
        raise NotImplementedError("DLC 3.3 (EDC + startup) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_6p3(self, dlc_options):
        raise NotImplementedError("DLC 6.3 (parked in normal wind) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_6p4(self, dlc_options):
        raise NotImplementedError("DLC 6.4 (50-yr wind + reduced waves) is wind-dominated and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_6p5(self, dlc_options):
        raise NotImplementedError("DLC 6.5 (EWM + loss of grid) is wind-dominated and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_9p1(self, dlc_options):
        raise NotImplementedError("DLC 9.1 is wind-only (standstill) and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_9p2(self, dlc_options):
        raise NotImplementedError("DLC 9.2 is wind-only (standstill) and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_10p1(self, dlc_options):
        raise NotImplementedError("DLC 10.1 is wind-only (blade inspection) and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_10p2(self, dlc_options):
        raise NotImplementedError("DLC 10.2 is wind-only (blade inspection) and does not apply to MHK TECs per IEC TS 62600-2.")

    def generate_12p1(self, dlc_options):
        raise NotImplementedError("DLC 12.1 (power production + 50-yr wind + grid loss) is wind-only and does not apply to MHK TECs per IEC TS 62600-2.")
