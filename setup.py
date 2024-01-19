import os
import sys
import subprocess
import setuptools

# Install the python sub-packages
os.chdir('dtqpy')

edit_str = ''
for k in sys.argv:
    if k in ['-e','--editable','develop']:
        edit_str = '-e'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', edit_str , '.'])

os.chdir('..')

setuptools.setup()

