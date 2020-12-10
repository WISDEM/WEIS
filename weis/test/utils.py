import os
import importlib
from pathlib import Path
from time import time


def execute_script(fscript):
    thisdir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(os.path.dirname(thisdir))
    examples_dir = os.path.join(root_dir, "examples")
    
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
    
def run_all_scripts(folder_string):
    scripts = [m for m in all_scripts if m.find(folder_string) >= 0]
    for k in scripts:
        try:
            execute_script(k)
        except:
            print("Failed to run,", k)
            self.assertTrue(False)