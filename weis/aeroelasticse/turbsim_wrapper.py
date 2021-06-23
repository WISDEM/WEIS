import os
import subprocess
class Turbsim_wrapper(object):
    def __init__(self):
        self.turbsim_exe = 'turbsim'
        self.turbsim_input = ""
        self.run_dir = '.'

    def execute(self):
        exec_string = [self.turbsim_exe, self.turbsim_input]
        olddir = os.getcwd()
        os.chdir(self.run_dir)
        subprocess.call(exec_string)
        os.chdir(olddir)