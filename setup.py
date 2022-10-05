import os
import sys
import platform
import multiprocessing
from distutils.core import run_setup
from setuptools import find_packages
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import setup, Extension
from io import open
import subprocess

# Global constants
ncpus = multiprocessing.cpu_count()
this_directory = os.path.abspath(os.path.dirname(__file__))

# Eagle environment
eagle_nodes = ['el'+str(m) for m in range(10)] + ['ed'+str(m) for m in range(10)]
eagle_flag = platform.node() in eagle_nodes
ci_flag    = platform.node().find('fv-az') >= 0
if eagle_flag:
    os.environ["FC"] = "ifort"
    os.environ["CC"] = "icc"
    os.environ["CXX"] = "icpc"
    os.environ["LDSHARED"] = "icc -pthread -shared"
    
# For the CMake Extensions
class CMakeExtension(Extension):

    def __init__(self, name, sourcedir='', **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):

    def copy_extensions_to_source(self):
        newext = []
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension): continue
            newext.append( ext )
        self.extensions = newext
        super().copy_extensions_to_source()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            # Ensure that CMake is present and working
            try:
                self.spawn(['cmake', '--version'])
            except OSError:
                raise RuntimeError('Cannot find CMake executable')

            localdir = os.path.join(this_directory, 'local')

            # CMAKE profiles default for all
            buildtype = 'RelWithDebInfo' # Hydrodyn has issues with Debug
            cmake_args = ['-DBUILD_SHARED_LIBS=ON',
                          '-DDOUBLE_PRECISION:BOOL=OFF',
                          '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                          '-DCMAKE_INSTALL_PREFIX='+localdir,
                          '-DCMAKE_BUILD_TYPE='+buildtype]
            buildtype = buildtype.upper()

            # Custom tuning
            mycompiler = self.compiler.compiler[0]
            if ci_flag:
                tune = '-g'  #-O0 -ffpe-trap=invalid,zero,overflow,underflow
                
            elif eagle_flag:
                tune = '-xSKYLAKE-AVX512'
                cmake_args += ['-DOPENMP=ON']
                try:
                    self.spawn(['ifort', '--version'])
                except OSError:
                    raise RuntimeError('Recommend loading intel compiler modules on Eagle (comp-intel, intel-mpi, mkl)')

            elif (mycompiler.find('ifort') >= 0 or mycompiler.find('icc') >= 0 or
                  mycompiler.find('icpc') >= 0):
                tune = '-xHost'
                
            else:
                tune = '-march=native -mtune=native'

            cmake_args += ['-DCMAKE_Fortran_FLAGS_'+buildtype+'='+tune,
                           '-DCMAKE_C_FLAGS_'+buildtype+'='+tune,
                           '-DCMAKE_CXX_FLAGS_'+buildtype+'='+tune]

            if platform.system() == 'Windows':
                cmake_args += ['-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE']

                if self.compiler.compiler_type == 'msvc':
                    cmake_args += ['-DCMAKE_GENERATOR_PLATFORM=x64']
                else:
                    cmake_args += ['-G', 'MinGW Makefiles']

            self.build_temp += '_'+ext.name
            os.makedirs(localdir, exist_ok=True)
            # Need fresh build directory for CMake
            os.makedirs(self.build_temp, exist_ok=True)

            self.spawn(['cmake','-S', ext.sourcedir, '-B', self.build_temp] + cmake_args)
            self.spawn(['cmake', '--build', self.build_temp, '-j', str(ncpus), '--target', 'install', '--config', buildtype])

        else:
            super().build_extension(ext)


# All of the extensions
fastExt    = CMakeExtension('openfast','OpenFAST')
roscoExt   = CMakeExtension('rosco','ROSCO/ROSCO')
extList = [roscoExt] if platform.system() == "Windows" else  [roscoExt, fastExt]

# Setup content
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

CLASSIFIERS = '''
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
'''

weis_pkgs       = find_packages()

# Install the python sub-packages
print(sys.argv)
for pkg in ['WISDEM','ROSCO','pCrunch','pyHAMS','MoorPy','RAFT','dtqpy']:
    os.chdir(pkg)
    run_setup('setup.py', script_args=sys.argv[1:], stop_after='run')
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])  # This option runs `pip install -e .` on each package
    os.chdir('..')

# Now install WEIS and the Fortran packages
metadata = dict(
    name                          = 'WEIS',
    version                       = '1.1',
    description                   = 'Wind Energy with Integrated Servo-control',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = 'NREL',
    url                           = 'https://github.com/WISDEM/WEIS',
    install_requires              = ['openmdao>=3.4','numpy','scipy','nlopt','dill','smt','control','jsonmerge','fatpack'],
    classifiers                   = [_f for _f in CLASSIFIERS.split('\n') if _f],
    packages                      = weis_pkgs,
    package_data                  =  {'':['*.yaml','*.xlsx']},
    python_requires               = '>=3.6',
    license                       = 'Apache License, Version 2.0',
    ext_modules                   = extList,
    cmdclass                      = {'build_ext': CMakeBuildExt},
    zip_safe                      = False,
)

setup(**metadata)
