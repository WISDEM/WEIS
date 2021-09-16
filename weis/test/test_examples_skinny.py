import unittest
from weis.test.utils import execute_script

skinny_scripts = [
    "02_control_opt/weis_driver",       #It's fast, I promise (49 sec. locally)
    "03_NREL5MW_OC3_spar/weis_driver",
    "06_IEA-15-240-RWT/weis_driver",
    "09_design_of_experiments/weis_driver",
]

class TestSkinnyExamples(unittest.TestCase):

    def test_all_scripts(self):
        for ks,s in enumerate(skinny_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s)
                    self.assertTrue(True)
                except:
                    self.assertEqual(s, "Success")

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSkinnyExamples))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
