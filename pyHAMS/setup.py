#!/usr/bin/env python
# encoding: utf-8

import setuptools #import setup
from numpy.distutils.core import setup, Extension
import os
import platform
import sys
import sysconfig

os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'

# Source order is important for dependencies
f90src = ['WavDynMods.f90',
          'PatclVelct.f90',
          'BodyIntgr.f90',
          'BodyIntgr_irr.f90',
          'AssbMatx.f90',    
          'AssbMatx_irr.f90',
          'SingularIntgr.f90',
          'InfGreen_Appr.f90',
          'FinGrnExtSubs.f90',
          'FinGreen3D.f90',
          'CalGreenFunc.f90',
          'HydroStatic.f90',
          'ImplementSubs.f90',
          'InputFiles.f90',
          'NormalProcess.f90',
          'ReadPanelMesh.f90',
          'PotentWavForce.f90',
          'PressureElevation.f90',
          'PrintOutput.f90',
          'SolveMotion.f90',
          'WavDynSubs.f90',
          'HAMS_Prog.f90',
          'HAMS_Prog.pyf',
          ]
root_dir = os.path.join('pyhams','src')

intel_flag = sysconfig.get_config_var('FC') == 'ifort'

if not intel_flag:
    for a in sys.argv:
        intel_flag = intel_flag or a.find('intel')>=0

if not intel_flag:
    try:
        if os.environ['FC'] == 'ifort':
            intel_flag = True
    except KeyError:
        pass

myargs = ['-O3','-m64','-fPIC','-g']
mycargs = ['-std=c11']
if intel_flag:
    myfargs = ['-mkl']
    mylib = ['mkl_rt']
    mylink = []
else:
    myfargs = ['-fno-align-commons','-fdec-math']
    mylib = ['lapack']
    mylink = []

pyhamsExt = Extension('pyhams.libhams', sources=[os.path.join(root_dir,m) for m in f90src],
                      extra_compile_args=mycargs+myargs,
                      extra_f90_compile_args=myfargs+myargs,
                      libraries=mylib,
                      extra_link_args=['-fopenmp']+mylink)
extlist = [] if platform.system() == 'Windows' else [pyhamsExt]

setup(
    name='pyHAMS',
    version='1.0.0',
    description='Python module wrapping around HAMS',
    author='NREL WISDEM Team',
    author_email='systems.engineering@nrel.gov',
    license='Apache License, Version 2.0',
    package_data={'pyhams': []},
    packages=['pyhams'],
    ext_modules=extlist,
)


