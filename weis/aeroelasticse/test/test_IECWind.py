import unittest
from weis.aeroelasticse.pyIECWind import pyIECWind_extreme
from weis.aeroelasticse.Turbsim_mdao.turbulence_spectrum import IECKaimal
from weis.aeroelasticse.Turbsim_mdao.turbsim_writer import TurbsimBuilder
from weis.aeroelasticse.Turbsim_mdao.turbsim_reader import turbsimReader
import numpy as np
import os

class TestIECWind(unittest.TestCase):
    def setUp(self):
        self.extreme_wind = pyIECWind_extreme()
        self.extreme_wind.setup()
        
        self.tsb = TurbsimBuilder()
        self.tsb.turbsim_vt.metboundconds.UserFile = 'tsim_user_turbulence_default.inp'
        self.tsb.turbsim_vt.metboundconds.ProfileFile = 'default.profile'
        self.tsb.run_dir = 'wind'
        self.tsb.tsim_input_file = 'test.inp'

    def test_NTM(self):
        sigma_1 = self.extreme_wind.NTM(10.0)
        np.testing.assert_almost_equal(sigma_1, 1.834)

    def test_ETM(self):
        sigma_1 = self.extreme_wind.ETM(10.0)
        np.testing.assert_almost_equal(sigma_1, 2.96128)

    def test_EWM(self):
        sigma_1, V_e50, V_e1, V_50, V_1 = self.extreme_wind.EWM(10.0)
        np.testing.assert_almost_equal(sigma_1, 1.1)
        np.testing.assert_almost_equal(V_e50, 70.0)
        np.testing.assert_almost_equal(V_e1, 56.0)
        np.testing.assert_almost_equal(V_50, 50.0)
        np.testing.assert_almost_equal(V_1, 40.0)

    def testIECKaimal(self):
        # Average wind speed at hub height
        V_hub = 10.
        # Wind turbine hub height
        HH = 100.
        # IEC Turbine Wind Speed Class, can be I, II, or III
        Class = 'I'
        # IEC Turbine Wind Turbulence Category, can be A, B, or C
        Categ = 'A'
        # Wind turbulence model, it can be NTM = normal turbulence, ETM = extreme turbulence, or EWM = extreme wind
        TurbMod = 'NTM'
        # Rotor radius, used to calculate rotor average wind spectrum
        R = 60.
        # Frequency range
        f=np.linspace(0.001, 20., 10)


        U, V, W, Rot = IECKaimal(f, V_hub, HH, Class, Categ, TurbMod, R)
        np.testing.assert_array_almost_equal(U , np.array([4.38659198e+02, 2.22282686e-02, 7.01694129e-03, 
                                                           3.57258290e-03, 2.21264170e-03, 1.52577475e-03, 
                                                           1.12612267e-03, 8.71070791e-04, 6.97323831e-04,
                                                           5.73069089e-04]), 5)
        np.testing.assert_array_almost_equal(V , np.array([1.14285126e+02, 2.93758372e-02, 9.30714989e-03,
                                                           4.74439644e-03, 2.94018708e-03, 2.02821285e-03, 
                                                           1.49732131e-03, 1.15840022e-03, 9.27463030e-04, 
                                                           7.62277955e-04]), 5)
        np.testing.assert_array_almost_equal(W , np.array([1.18477575e+01, 2.83846534e-02, 9.14368382e-03,
                                                           4.68723385e-03, 2.91293916e-03, 2.01281628e-03,
                                                           1.48763283e-03, 1.15183382e-03, 9.22764383e-04,
                                                           7.58773687e-04]), 5)
        np.testing.assert_array_almost_equal(Rot , np.array([3.47567168e+02, 1.72828935e-06, 1.36729085e-07,
                                                            0,           0,            0,    
                                                            0,           0,            0,
                                                            0]), 5)
                                                           
    def testTurbSimwriter(self):
        self.tsb.execute()

    def testTurbSimreader(self):
        self.tsb.execute()
        reader = turbsimReader()
        reader.read_input_file(os.path.join(self.tsb.run_dir, self.tsb.tsim_input_file))


if __name__ == "__main__":
    unittest.main()
