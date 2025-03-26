import os
import unittest
from weis.test.utils import execute_script

skinny_scripts = [
    "02_run_openfast_cases/weis_driver_rosco_opt",       #It's fast, I promise (49 sec. locally)
    "02_run_openfast_cases/weis_driver_sm",    #Not as fast as weis_driver, but not too bad (120 sec. locally)
    "03_NREL5MW_OC3_spar/weis_freq_driver",
    "05_IEA-3.4-130-RWT/weis_driver_model_only", 
    "06_IEA-15-240-RWT/weis_driver_monopile",
    "06_IEA-15-240-RWT/weis_driver_TMDs",
    "09_design_of_experiments/DOE_openfast",
    # "13_DTQP/gen_oloc",
    "17_IEA22_Optimization/driver_weis_openfast_opt",   # Moved from all_scripts to test viz app
    "17_IEA22_Optimization/driver_weis_raft_opt",       # Moved from all_scripts to test viz app
]


all_scripts = [
    "01_aeroelasticse/run_general",
    "01_aeroelasticse/run_OLAF",
    #"02_run_openfast_cases/weis_driver_rosco_opt",  # executed in examples_skinny
    #"02_run_openfast_cases/weis_driver_sm", # executed in examples_skinny
    "02_run_openfast_cases/weis_driver_loads",
    "03_NREL5MW_OC3_spar/weis_driver",
    # "03_NREL5MW_OC3_spar/weis_freq_driver", # executed in examples_skinny
    # "04_NREL5MW_OC4_semi/weis_driver",  # skipping until we resolve multiple variable ballasts
    "04_NREL5MW_OC4_semi/weis_freq_driver",
    "05_IEA-3.4-130-RWT/weis_driver", # also executed via mpi in the gitthub workflow
    #"05_IEA-3.4-130-RWT/weis_driver_model_only",  # executed in examples_skinny
    #"06_IEA-15-240-RWT/weis_driver", # executed in the test_IEA15.py
    #"06_IEA-15-240-RWT/weis_driver_monopile", # executed in examples_skinny
    #"06_IEA-15-240-RWT/weis_driver_TMDs", # executed in examples_skinny
    # "07_te_flaps/dac_driver",
    "08_OLAF/weis_driver",
    #"09_design_of_experiments/weis_driver", # executed in the test_DOE.py
    #"09_design_of_experiments/DOE_openfast", # executed in examples_skinny
    "10_override_example/weis_driver",
    #"12_linearization/doe_driver", # Soul crushingly long
    "12_linearization/weis_driver",
    "15_RAFT_Studies/weis_driver_raft_opt",
    # "17_IEA22_Optimization/driver_weis_openfast_opt",     # executed in examples_skinny
    # "17_IEA22_Optimization/driver_weis_raft_opt",     # executed in examples_skinny
]

class TestExamples(unittest.TestCase):

    def test_skinny(self):
        for ks,s in enumerate(skinny_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s)
                    self.assertTrue(True)
                except:
                    self.assertEqual(s, "Success")

    @unittest.skipUnless("RUN_EXHAUSTIVE" in os.environ, "exhaustive on pull request only")
    def test_all_scripts(self):
        for ks,s in enumerate(all_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s)
                    self.assertTrue(True)
                except:
                    self.assertEqual(s, "Success")

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
