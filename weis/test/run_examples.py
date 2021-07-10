import unittest
from weis.test.utils import execute_script




# 02_ref turbines are regression tested in test_gluecode, no need to duplicate runtime
all_scripts = [
    "01_aeroelasticse/run_general",
    "01_aeroelasticse/run_mass_sweep",
    "01_aeroelasticse/run_PC_sweep",
    "01_aeroelasticse/run_OLAF",
    
    # "02_control_opt/run_lin_turbine",
    # "02_control_opt/runOptimization",
    
    #"03_NREL5MW_OC3_spar/weis_driver", # executed in the test_OC3.py
    "03_NREL5MW_OC3_spar/weis_freq_driver",
    
    "04_NREL5MW_OC4_semi/weis_driver",
    "04_NREL5MW_OC4_semi/weis_freq_driver",
    
    "05_IEA-3.4-130-RWT/weis_driver",
    
    #"06_IEA-15-240-RWT/weis_driver", # executed in the test_IEA15.py
    
    # "07_te_flaps/dac_driver",

    "08_OLAF/weis_driver",
    
    #"09_design_of_experiments/weis_driver", # executed in the test_DOE.py
    
    "10_override_example/weis_driver",

    "12_linearization/doe_driver",

    "12_linearization/weis_driver"
]

class TestExamples(unittest.TestCase):

    def test_all_scripts(self):
        for ks,s in enumerate(all_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s)
                    self.assertTrue(True)
                except:
                    self.assertEqual(s, "Failed")

    '''
    def test_aeroelasticse(self):
        try:
            run_all_scripts("01_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)
                    
    def test_control_opt(self):
        try:
            run_all_scripts("02_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)
                    
    def test_OC3(self):
        try:
            run_all_scripts("03_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)
                    
    def test_IEA_3_4(self):
        try:
            run_all_scripts("05_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)
                    
    def test_IEA_15(self):
        try:
            run_all_scripts("06_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_OLAF(self):
        try:
            run_all_scripts("08_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_DOE(self):
        try:
            run_all_scripts("09_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_override(self):
        try:
            run_all_scripts("10_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_linearization(self):
        try:
            run_all_scripts("12_", all_scripts)
            self.assertTrue(True)
        except:
            self.assertTrue(False)


    def test_linearization(self):
        run_all_scripts("12_", all_scripts)
    '''


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
