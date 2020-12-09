import os
import unittest
import importlib
from pathlib import Path
from time import time


thisdir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(os.path.dirname(thisdir))
examples_dir = os.path.join(root_dir, "examples")

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


def execute_script(fscript):
    # Go to location due to relative path use for airfoil files
    print("\n\n")
    print("NOW RUNNING:", fscript)
    print()
    fullpath = os.path.join(examples_dir, fscript + ".py")
    basepath = os.path.join(examples_dir, fscript.split("/")[0])
    os.chdir(basepath)

    # Get script/module name
    froot = fscript.split("/")[-1]

    # Use dynamic import capabilities
    # https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
    print(froot, os.path.realpath(fullpath))
    spec = importlib.util.spec_from_file_location(froot, os.path.realpath(fullpath))
    mod = importlib.util.module_from_spec(spec)
    s = time()
    spec.loader.exec_module(mod)
    print(time() - s, "seconds to run")


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