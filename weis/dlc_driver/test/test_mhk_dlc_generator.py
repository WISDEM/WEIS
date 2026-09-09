"""
Unit tests for MHKDLCGenerator
Tests case generation for all MHK TEC DLCs per IEC TS 62600-2.
"""
import unittest
import os
import numpy as np
import weis.inputs as sch
from weis.dlc_driver.dlc_generator import DLCGenerator
from weis.dlc_driver.dlc_generator_mhk import MHKDLCGenerator


class TestMHKDLCGenerator(unittest.TestCase):
    """Test MHK DLC generator case generation (no OpenFAST runs)."""

    @classmethod
    def setUpClass(cls):
        """Load modeling options and create generator once for all tests."""
        this_dir = os.path.dirname(__file__)
        fname = os.path.join(this_dir, "weis_inputs", "modeling_options_mhk_dlcs.yaml")
        cls.modeling_options = sch.load_modeling_yaml(fname)

        cls.cs_cut_in = 0.5
        cls.cs_cut_out = 3.5
        cls.cs_rated = 2.0

        cls.metocean = cls.modeling_options['DLC_driver']['metocean_conditions']
        cls.dlc_driver_options = cls.modeling_options['DLC_driver']

    def _make_generator(self):
        """Create a fresh MHK generator."""
        return MHKDLCGenerator(
            ws_cut_in=self.cs_cut_in,
            ws_cut_out=self.cs_cut_out,
            ws_rated=self.cs_rated,
            fix_wind_seeds=True,
            fix_wave_seeds=True,
            metocean=self.metocean,
            dlc_driver_options=self.dlc_driver_options,
        )

    def _generate_single_dlc(self, dlc_label, extra_opts=None):
        """Generate a single DLC and return (generator, cases)."""
        gen = self._make_generator()
        dlc_opt = {'DLC': dlc_label}
        # Apply schema defaults
        for key, val in gen.dlc_schema.items():
            if key not in dlc_opt and 'default' in val:
                dlc_opt[key] = val['default']
        if extra_opts:
            dlc_opt.update(extra_opts)
        gen.generate(dlc_label, dlc_opt)
        return gen, gen.cases

    # ── Init tests ───────────────────────────────────────────────────────

    def test_init_flow_key(self):
        gen = self._make_generator()
        self.assertEqual(gen.flow_key, 'current_speed')
        self.assertTrue(gen.MHK)

    def test_init_metocean_loaded(self):
        gen = self._make_generator()
        self.assertAlmostEqual(gen.current_peak_spring, 4.0)
        self.assertAlmostEqual(gen.current_mean_spring, 3.0)
        self.assertAlmostEqual(gen.wave_height_cutout, 1.5)

    def test_init_openfast_input_map(self):
        gen = self._make_generator()
        self.assertIn('current_speed', gen.openfast_input_map)
        cs_map = gen.openfast_input_map['current_speed']
        modules = [t[0] for t in cs_map]
        self.assertIn('InflowWind', modules)

    # ── DLC 1.1 ──────────────────────────────────────────────────────────

    def test_1p1_generates_cases(self):
        gen, cases = self._generate_single_dlc('1.1')
        self.assertGreater(len(cases), 0)
        for c in cases:
            self.assertEqual(c.label, '1.1')
            self.assertGreaterEqual(c.URef, self.cs_cut_in)
            self.assertLessEqual(c.URef, self.cs_cut_out)

    def test_1p1_psf(self):
        _, cases = self._generate_single_dlc('1.1')
        for c in cases:
            self.assertAlmostEqual(c.PSF, 1.35)

    # ── DLC 1.2 ──────────────────────────────────────────────────────────

    def test_1p2_wave_heading_sweep(self):
        _, cases = self._generate_single_dlc('1.2')
        headings = set(c.wave_direction for c in cases)
        # Should have 12 headings (0, 30, ..., 330)
        self.assertEqual(len(headings), 12)

    def test_1p2_cutout_wave_height(self):
        _, cases = self._generate_single_dlc('1.2')
        for c in cases:
            # All cases should use the cutout wave height
            self.assertAlmostEqual(c.wave_height, 1.5)

    # ── DLC 1.3 ──────────────────────────────────────────────────────────

    def test_1p3_etm(self):
        _, cases = self._generate_single_dlc('1.3')
        for c in cases:
            self.assertEqual(c.label, '1.3')
            self.assertEqual(c.IEC_WindType, '1ETM')
            self.assertAlmostEqual(c.PSF, 1.35)

    # ── DLC 2.1 ──────────────────────────────────────────────────────────

    def test_2p1_requires_genfault(self):
        with self.assertRaises(Exception):
            self._generate_single_dlc('2.1')

    def test_2p1_rated_to_cutout(self):
        _, cases = self._generate_single_dlc('2.1', {'genfault_time': 100})
        speeds = set(c.URef for c in cases)
        self.assertIn(self.cs_rated, speeds)
        self.assertIn(self.cs_cut_out, speeds)

    # ── DLC 2.2 ──────────────────────────────────────────────────────────

    def test_2p2_requires_fault(self):
        with self.assertRaises(Exception):
            self._generate_single_dlc('2.2')

    def test_2p2_with_pitch_fault(self):
        _, cases = self._generate_single_dlc('2.2', {
            'pitchfault_time1': 100,
            'pitchfault_blade1pos': 20,
        })
        self.assertGreater(len(cases), 0)
        for c in cases:
            self.assertAlmostEqual(c.PSF, 1.1)

    # ── DLC 2.3 ──────────────────────────────────────────────────────────

    def test_2p3_not_implemented(self):
        """DLC 2.3 is a template stub until accidental-fault setup is defined."""
        with self.assertRaises(NotImplementedError):
            self._generate_single_dlc('2.3', {'genfault_time': 100})

    # ── DLC 3.1 ──────────────────────────────────────────────────────────

    def test_3p1_startup(self):
        _, cases = self._generate_single_dlc('3.1')
        for c in cases:
            self.assertEqual(c.label, '3.1')
            self.assertEqual(c.turbine_status, 'parked-idling')

    # ── DLC 3.2 ──────────────────────────────────────────────────────────

    def test_3p2_cutout_waves(self):
        _, cases = self._generate_single_dlc('3.2')
        for c in cases:
            self.assertAlmostEqual(c.wave_height, 1.5)

    # ── DLC 4.1 ──────────────────────────────────────────────────────────

    def test_4p1_requires_shutdown_time(self):
        with self.assertRaises(Exception):
            self._generate_single_dlc('4.1')

    def test_4p1_shutdown(self):
        _, cases = self._generate_single_dlc('4.1', {
            'analysis_time': 700,
            'normal_shutdown_time': 400,
        })
        self.assertGreater(len(cases), 0)
        for c in cases:
            self.assertEqual(c.label, '4.1')

    # ── DLC 4.2 ──────────────────────────────────────────────────────────

    def test_4p2_cutout_waves(self):
        _, cases = self._generate_single_dlc('4.2', {
            'normal_shutdown_time': 400,
        })
        for c in cases:
            self.assertAlmostEqual(c.wave_height, 1.5)

    # ── DLC 5.1 ──────────────────────────────────────────────────────────

    def test_5p1_emergency(self):
        _, cases = self._generate_single_dlc('5.1', {
            'shutdown_time': 300,
            'transient_time': 0.0,
            'n_azimuth': 1,
        })
        for c in cases:
            self.assertEqual(c.label, '5.1')
            self.assertAlmostEqual(c.PSF, 1.35)

    # ── DLC 6.1 ──────────────────────────────────────────────────────────

    def test_6p1_subcases(self):
        _, cases = self._generate_single_dlc('6.1')
        labels = set(c.label for c in cases)
        self.assertIn('6.1a', labels)
        self.assertIn('6.1b', labels)

    def test_6p1_parked(self):
        _, cases = self._generate_single_dlc('6.1')
        for c in cases:
            self.assertEqual(c.turbine_status, 'parked-idling')
            self.assertAlmostEqual(c.URef, 4.0)  # peak spring

    # ── DLC 6.2 ──────────────────────────────────────────────────────────

    def test_6p2_abnormal(self):
        _, cases = self._generate_single_dlc('6.2')
        for c in cases:
            self.assertAlmostEqual(c.PSF, 1.1)
            self.assertEqual(c.turbine_status, 'parked-idling')

    # ── DLC 7.1 ──────────────────────────────────────────────────────────

    def test_7p1_mean_spring(self):
        _, cases = self._generate_single_dlc('7.1')
        for c in cases:
            self.assertAlmostEqual(c.URef, 3.0)  # mean spring
            self.assertAlmostEqual(c.PSF, 1.1)
            self.assertEqual(c.IEC_WindType, '1ETM')

    # ── DLC 7.2 ──────────────────────────────────────────────────────────

    def test_7p2_operational_range(self):
        _, cases = self._generate_single_dlc('7.2')
        for c in cases:
            self.assertEqual(c.label, '7.2')
            self.assertAlmostEqual(c.PSF, 1.35)
            self.assertEqual(c.turbine_status, 'parked-idling')

    # ── AEP ──────────────────────────────────────────────────────────────

    def test_aep_ti_interpolation(self):
        _, cases = self._generate_single_dlc('AEP')
        for c in cases:
            self.assertEqual(c.label, 'AEP')
            # IECturbc should be TI percentage (> 0)
            self.assertGreater(c.IECturbc, 0)
            # TI at 1.0 m/s is 0.75 (default AEP scale) x the 10% NTM value
            if abs(c.URef - 1.0) < 0.01:
                self.assertAlmostEqual(c.IECturbc, 7.5, places=0)

    def test_aep_constant_ti(self):
        _, cases = self._generate_single_dlc('AEP', {'TI': 15.0})
        for c in cases:
            self.assertAlmostEqual(c.IECturbc, 15.0)

    # ── Turbulence from metocean tables (no IEC turbulence class) ─────────

    def test_ti_and_ustar_on_all_turbulent_dlcs(self):
        for dlc in ['1.1', '1.2', '1.3', '7.1', 'AEP']:
            gen, cases = self._generate_single_dlc(dlc)
            turb = [c for c in cases if c.turbulent_wind]
            self.assertGreater(len(turb), 0, msg=f'DLC {dlc} has no turbulent cases')
            if 'ETM' in turb[0].IEC_WindType:
                table = gen.current_TI_ETM
            elif dlc == 'AEP':
                table = gen.current_TI_AEP
            else:
                table = gen.current_TI_NTM
            for c in turb:
                TI = np.interp(c.URef, self.metocean['current_speed'], table)
                self.assertAlmostEqual(c.IECturbc, TI * 100)
                self.assertAlmostEqual(c.UStar, 1.2814 * TI * c.URef)

    def test_scalar_ti_scales_ntm(self):
        metocean = dict(self.metocean, current_TI_ETM=1.25, current_TI_AEP=0.75)
        gen = MHKDLCGenerator(
            ws_cut_in=self.cs_cut_in, ws_cut_out=self.cs_cut_out, ws_rated=self.cs_rated,
            metocean=metocean, dlc_driver_options=self.dlc_driver_options,
        )
        np.testing.assert_allclose(gen.current_TI_ETM, 1.25 * np.array(self.metocean['current_TI_NTM']))
        np.testing.assert_allclose(gen.current_TI_AEP, 0.75 * np.array(self.metocean['current_TI_NTM']))

    def test_etm_ti_exceeds_ntm(self):
        _, ntm = self._generate_single_dlc('1.1')
        _, etm = self._generate_single_dlc('1.3')
        self.assertGreater(max(c.IECturbc for c in etm), max(c.IECturbc for c in ntm))

    # ── Wind-only DLCs blocked ───────────────────────────────────────────

    def test_wind_only_dlcs_blocked(self):
        blocked = ['1.4', '1.5', '1.6', '6.3', '6.4']
        for dlc in blocked:
            with self.assertRaises(NotImplementedError, msg=f"DLC {dlc} should be blocked"):
                self._generate_single_dlc(dlc)

    # ── Full matrix integration ──────────────────────────────────────────

    def test_full_matrix(self):
        """Generate all MHK DLCs from the test YAML and verify total case count."""
        gen = self._make_generator()
        DLCs = self.modeling_options['DLC_driver']['DLCs']

        for dlc_entry in DLCs:
            dlc_opt = dict(dlc_entry)
            label = dlc_opt.pop('DLC')
            # Apply schema defaults
            for key, val in gen.dlc_schema.items():
                if key not in dlc_opt and 'default' in val:
                    dlc_opt[key] = val['default']
            gen.generate(label, dlc_opt)

        self.assertGreater(gen.n_cases, 0)
        # Check we have cases for expected labels
        labels = set(c.label for c in gen.cases)
        expected = {'1.1', '1.2', '1.3', '2.1', '2.2', '3.1', '3.2',
                    '4.1', '4.2', '5.1', '6.1a', '6.1b', '6.2', '7.1', '7.2', 'AEP'}
        self.assertTrue(expected.issubset(labels), f"Missing labels: {expected - labels}")
        print(f"\nTotal MHK cases generated: {gen.n_cases}")
        for lab in sorted(labels):
            n = sum(1 for c in gen.cases if c.label == lab)
            print(f"  {lab}: {n} cases")


class TestWindDLCRegression(unittest.TestCase):
    """Verify the base class wind DLC generator still works after refactoring."""

    def test_wind_dlc_generator(self):
        """Regression: base class should produce the same case count as before."""
        this_dir = os.path.dirname(__file__)
        fname = os.path.join(this_dir, "weis_inputs", "modeling_options_all_dlcs.yaml")
        modeling_options = sch.load_modeling_yaml(fname)

        ws_cut_in = 4.
        ws_cut_out = 25.
        ws_rated = 10.
        wind_speed_class = 'I'
        wind_turbulence_class = 'B'

        metocean = modeling_options['DLC_driver']['metocean_conditions']
        dlc_generator = DLCGenerator(
            ws_cut_in, ws_cut_out, ws_rated,
            wind_speed_class, wind_turbulence_class,
            modeling_options['DLC_driver']['fix_wind_seeds'],
            modeling_options['DLC_driver']['fix_wave_seeds'],
            metocean,
            modeling_options['DLC_driver'],
        )

        DLCs = modeling_options['DLC_driver']['DLCs']
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)

        # Original expected value from test_DLC_generator.py
        np.testing.assert_equal(dlc_generator.n_cases, 481)
        np.testing.assert_equal(dlc_generator.cases[11].URef, ws_cut_out)

        # AEP wind speed count
        dlc_aep_ws = [c.URef for c in dlc_generator.cases if c.label == '1.1']
        n_ws_aep = len(np.unique(dlc_aep_ws))
        np.testing.assert_equal(n_ws_aep, 12)


if __name__ == "__main__":
    unittest.main()
