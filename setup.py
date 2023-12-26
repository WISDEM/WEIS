import os
import sys
from distutils.core import run_setup
import setuptools

# Install the python sub-packages
os.chdir('dtqpy')
run_setup('setup.py', script_args=sys.argv[1:], stop_after='run')
os.chdir('..')

setuptools.setup()

