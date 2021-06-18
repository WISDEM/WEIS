import unittest
from weis.test.utils import run_all_scripts




# 02_ref turbines are regression tested in test_gluecode, no need to duplicate runtime
all_scripts = [
    "01_aeroelasticse/run_DLC",
    "01_aeroelasticse/run_general",
    # "01_aeroelasticse/run_mass_sweep",
    "01_aeroelasticse/run_OLAF",
    "01_aeroelasticse/run_iec_turbsim",
    # "01_aeroelasticse/run_PC_sweep",
    # "01_aeroelasticse/run_stability_analysis",
    
    # "02_control_opt/run_lin_turbine",
    # "02_control_opt/runOptimization",
    
    "03_NREL5MW_OC3_spar/weis_driver",
    # "03_NREL5MW_OC3_spar/weis_freq_driver",
    
    "04_NREL5MW_OC4_semi/weis_driver",
    "04_NREL5MW_OC4_semi/weis_freq_driver",
    
    "05_IEA-3.4-130-RWT/weis_driver",
    
    "06_IEA-15-240-RWT/weis_driver",
    
    # "07_te_flaps/dac_driver",

    "08_OLAF/weis_driver",
    
    "09_design_of_experiments/weis_driver",
    
    "10_override_example/weis_driver",
    
    
    
]

class TestExamples(unittest.TestCase):
    
    def test_aeroelasticse(self):
        run_all_scripts("01_", all_scripts)
                    
    def test_control_opt(self):
        run_all_scripts("02_", all_scripts)
                    
    def test_OC3(self):
        run_all_scripts("03_", all_scripts)
                    
    def test_IEA_3_4(self):
        run_all_scripts("05_", all_scripts)
                    
    def test_IEA_15(self):
        run_all_scripts("06_", all_scripts)

    def test_OLAF(self):
        run_all_scripts("08_", all_scripts)

    def test_DOE(self):
        run_all_scripts("09_", all_scripts)

    def test_override(self):
        run_all_scripts("10_", all_scripts)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExamples))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
