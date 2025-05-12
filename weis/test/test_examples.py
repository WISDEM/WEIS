import os
import unittest
from weis.test.utils import execute_script

# Run for each push on all platforms (choose one from each example directory)
skinny_scripts = [
    "01_simulate_own_openfast_model/dlc_sim_driver",
    "02_generate_openfast_model_for_dlcs/iea15_semi_driver", 
    "03_design_with_openfast/iea22_ptfm_opt_driver",                # these are used to test visualization
    "04_frequency_domain_analysis_design/iea22_raft_opt_driver",    # these are used to test visualization
    "05_control_optimization/rosco_opt_driver", 
    "08_potential_flow_modeling/raft_potmod_driver",
]

# Only run on PR on Ubuntu
extra_scripts = [
    "01_simulate_own_openfast_model/run_openfast_cases",
    "02_generate_openfast_model_for_dlcs/iea15_monopile_driver",
    "02_generate_openfast_model_for_dlcs/iea34_driver",
    "02_generate_openfast_model_for_dlcs/oc3_driver",
    "02_generate_openfast_model_for_dlcs/olaf_driver",
    "03_design_with_openfast/tower_design_driver",
    "04_frequency_domain_analysis_design/oc3_raft_driver",
    "04_frequency_domain_analysis_design/umaine_semi_raft_opt_driver",
    "05_control_optimization/tmd_opt_driver",
    # "08_potential_flow_modeling/openfast_potmod_driver",   #skip this one for now
]

# Notes:
# 06_parameteric_analysis is run when testing MPI configuration
# 07_postprocessing_notebooks are also tested separately

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
    def test_extra_scripts(self):
        for ks,s in enumerate(extra_scripts):
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
