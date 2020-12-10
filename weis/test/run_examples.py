import os
import unittest
from time import time
from weis.test.utils import execute_script




# 02_ref turbines are regression tested in test_gluecode, no need to duplicate runtime
all_scripts = [
    "01_aeroelasticse/run_DLC",
    "01_aeroelasticse/run_general",
    # "01_aeroelasticse/run_mass_sweep",
    "01_aeroelasticse/run_OLAF",
    # "01_aeroelasticse/run_PC_sweep",
    "01_aeroelasticse/run_stability_analysis",
    
    "02_control_opt/run_lin_turbine",
    "02_control_opt/runOptimization",
    
    "03_NREL5MW_OC3_spar/weis_driver",
    
    # "04_NREL5MW_OC4_semi/",  # there appears to be no python file within this folder
    
    "05_IEA-3.4-130-RWT/weis_driver",
    
    "06_IEA-15-240-RWT/run_control_opt",
    "06_IEA-15-240-RWT/weis_driver",
    
    # "07_te_flaps/dac_driver",
]





class TestExamples(unittest.TestCase):
    
    def test_aeroelasticse(self):
        scripts = [m for m in all_scripts if m.find("01_") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExamples))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())