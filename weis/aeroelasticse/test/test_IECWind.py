import unittest

from openmdao.utils.assert_utils import assert_near_equal
from weis.aeroelasticse.pyIECWind import pyIECWind_extreme


class TestIECWind(unittest.TestCase):
    def setUp(self):
        self.extreme_wind = pyIECWind_extreme()
        self.extreme_wind.setup()

    def test_NTM(self):
        sigma_1 = self.extreme_wind.NTM(10.0)
        assert_near_equal(sigma_1, 1.834)

    def test_ETM(self):
        sigma_1 = self.extreme_wind.ETM(10.0)
        assert_near_equal(sigma_1, 2.96128)

    def test_EWM(self):
        sigma_1, V_e50, V_e1, V_50, V_1 = self.extreme_wind.EWM(10.0)
        assert_near_equal(sigma_1, 1.1)
        assert_near_equal(V_e50, 70.0)
        assert_near_equal(V_e1, 56.0)
        assert_near_equal(V_50, 50.0)
        assert_near_equal(V_1, 40.0)


if __name__ == "__main__":
    unittest.main()
