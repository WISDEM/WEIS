#!/usr/bin/env python
# encoding: utf-8

# setup.py
# only if building in place: ``python setup.py build_ext --inplace``
import os
import re
import platform
import shutil
import setuptools
import subprocess


def run_meson_build(staging_dir):
    prefix = os.path.join(os.getcwd(), staging_dir)
    purelibdir = "."

    # check if meson extra args are specified
    meson_args = ""
    if "MESON_ARGS" in os.environ:
        meson_args = os.environ["MESON_ARGS"]

    if platform.system() == "Windows":
        if not "FC" in os.environ:
            os.environ["FC"] = "gfortran"
        if not "CC" in os.environ:
            os.environ["CC"] = "gcc"

    # configure
    meson_path = shutil.which("meson")
    meson_call = (
        f"{meson_path} setup {staging_dir} --prefix={prefix} "
        + f"-Dpython.purelibdir={purelibdir} -Dpython.platlibdir={purelibdir} {meson_args}"
    )
    sysargs = meson_call.split(" ")
    sysargs = [arg for arg in sysargs if arg != ""]
    print(sysargs)
    p1 = subprocess.run(sysargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.makedirs(staging_dir, exist_ok=True)
    setup_log = os.path.join(staging_dir, "setup.log")
    with open(setup_log, "wb") as f:
        f.write(p1.stdout)
    if p1.returncode != 0:
        with open(setup_log, "r") as f:
            print(f.read())
        raise OSError(sysargs, f"The meson setup command failed! Check the log at {setup_log} for more information.")

    # build
    meson_call = f"{meson_path} compile -vC {staging_dir}"
    sysargs = meson_call.split(" ")
    sysargs = [arg for arg in sysargs if arg != ""]
    print(sysargs)
    p2 = subprocess.run(sysargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    compile_log = os.path.join(staging_dir, "compile.log")
    with open(compile_log, "wb") as f:
        f.write(p2.stdout)
    if p2.returncode != 0:
        with open(compile_log, "r") as f:
            print(f.read())
        raise OSError(
            sysargs, f"The meson compile command failed! Check the log at {compile_log} for more information."
        )


def copy_shared_libraries():
    build_path = os.path.join(staging_dir, "pyhams")
    for root, _dirs, files in os.walk(build_path):
        for file in files:
            # move pyhams to just under staging_dir
            if file.endswith((".so", ".lib", ".pyd", ".pdb", ".dylib", ".dll", ".mod")):
                if ".so.p" in root or ".pyd.p" in root:  # excludes intermediate object files
                    continue
                file_path = os.path.join(root, file)
                new_path = str(file_path)
                match = re.search(staging_dir, new_path)
                new_path = new_path[match.span()[1] + 1 :]
                print(f"Copying build file {file_path} -> {new_path}")
                shutil.copy(file_path, new_path)


if __name__ == "__main__":
    # This is where the meson build system will install to, it is then
    # used as the sources for setuptools
    staging_dir = "meson_build"

    # If on Windows, use the compiled exe from Yingyi
    if ( (not platform.system() == "Windows") and
         ("dist" not in str(os.path.abspath(__file__))) ):
        cwd = os.getcwd()
        run_meson_build(staging_dir)
        os.chdir(cwd)
        copy_shared_libraries()

    #docs_require = ""
    #req_txt = os.path.join("doc", "requirements.txt")
    #if os.path.isfile(req_txt):
    #    with open(req_txt) as f:
    #        docs_require = f.read().splitlines()

    init_file = os.path.join("pyhams", "__init__.py")
    #__version__ = re.findall(
    #    r"""__version__ = ["']+([0-9\.]*)["']+""",
    #    open(init_file).read(),
    #)[0]

    setuptools.setup(
        name='pyHAMS',
        version='1.0.0',
        description='Python module wrapping around HAMS',
        long_description="pyHAM is a Python interface to the HAMS boundary-element solver for underwater potential flow solutions",
        author='NREL WISDEM Team',
        author_email='systems.engineering@nrel.gov',
        license='Apache License, Version 2.0',
        install_requires=[
            "numpy",
            "meson",
            "ninja",
        ],
        extras_require={
            "testing": ["pytest"],
        },
        python_requires=">=3.8",
        package_data={"": ["*.yaml", "*.so", "*.lib", "*.pyd", "*.pdb", "*.dylib", "*.dll"]},
        packages=['pyhams'],
        zip_safe=False,
    )

#os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'

# Source order is important for dependencies
#f90src = ['WavDynMods.f90',
#          'PatclVelct.f90',
#          'BodyIntgr.f90',
#          'BodyIntgr_irr.f90',
#          'AssbMatx.f90',    
#          'AssbMatx_irr.f90',
#          'SingularIntgr.f90',
#          'InfGreen_Appr.f90',
#          'FinGrnExtSubs.f90',
#          'FinGreen3D.f90',
#          'CalGreenFunc.f90',
#          'HydroStatic.f90',
#          'ImplementSubs.f90',
#          'InputFiles.f90',
#          'NormalProcess.f90',
#          'ReadPanelMesh.f90',
#          'PotentWavForce.f90',
#          'PressureElevation.f90',
#          'PrintOutput.f90',
#          'SolveMotion.f90',
#          'WavDynSubs.f90',
#          'HAMS_Prog.f90',
#          'HAMS_Prog.pyf',
#          ]
#root_dir = os.path.join('pyhams','src')

#intel_flag = sysconfig.get_config_var('FC') == 'ifort'

#if not intel_flag:
#    for a in sys.argv:
#        intel_flag = intel_flag or a.find('intel')>=0

#if not intel_flag:
#    try:
#        if os.environ['FC'] == 'ifort':
#            intel_flag = True
#    except KeyError:
#        pass

#myargs = ['-O3','-m64','-fPIC','-g']
#mycargs = ['-std=c11']
#if intel_flag:
#    myfargs = ['-mkl']
#    mylib = ['mkl_rt']
#    mylink = []
#else:
#    myfargs = ['-fno-align-commons','-fdec-math']
#    mylib = ['lapack']
#    mylink = []


#pyhamsExt = Extension('pyhams.libhams', sources=[os.path.join(root_dir,m) for m in f90src],
#                      extra_compile_args=mycargs+myargs,
#                      extra_f90_compile_args=myfargs+myargs,
#                      libraries=mylib,
#                      extra_link_args=['-fopenmp']+mylink)
#extlist = [] if platform.system() == 'Windows' else [pyhamsExt]
