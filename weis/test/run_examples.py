import unittest
from weis.test.utils import execute_script

all_scripts = [
    "01_aeroelasticse/run_general",
    "01_aeroelasticse/run_OLAF",
    #"03_NREL5MW_OC3_spar/weis_driver", # executed in the test_OC3.py
    "03_NREL5MW_OC3_spar/weis_freq_driver",
    "04_NREL5MW_OC4_semi/weis_driver",
    "04_NREL5MW_OC4_semi/weis_freq_driver",
    "05_IEA-3.4-130-RWT/weis_driver", # also executed via mpi in the gitthub workflow
    #"06_IEA-15-240-RWT/weis_driver", # executed in the test_IEA15.py
    # "07_te_flaps/dac_driver",
    "08_OLAF/weis_driver",
    #"09_design_of_experiments/weis_driver", # executed in the test_DOE.py
    "10_override_example/weis_driver",
    #"12_linearization/doe_driver", # Soul crushingly long
    "12_linearization/weis_driver",
    "15_RAFT_Studies/weis_driver_umaine_semi"
]

class TestExamples(unittest.TestCase):

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
