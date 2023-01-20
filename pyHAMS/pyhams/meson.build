# NumPy include directory - needed in all submodules
incdir_numpy = get_option('incdir_numpy')
if incdir_numpy == ''
    incdir_numpy = run_command(py3,
      [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
      ],
      check: true
    ).stdout().strip()
endif
# this creates a raw string which is useful for Windows use of '\' for paths
incdir_numpy = '''@0@'''.format(incdir_numpy)

#incdir_f2py = run_command(py3,
#    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
#    check : true
#).stdout().strip()

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)

hams_connect = custom_target('_hamsmodule.c',
                             input : ['src/HAMS_Prog.f90'],
                             output : ['_hamsmodule.c', '_hams-f2pywrappers2.f90'],
                             command: [py3, '-m', 'numpy.f2py',
			               '@INPUT@', '-m', '_hams', '--lower', '--build-dir', 'pyhams']
                            )

# Source order is important for dependencies
py3.extension_module('_hams',
                     'src/WavDynMods.f90',
                     'src/PatclVelct.f90',
                     'src/BodyIntgr.f90',
                     'src/BodyIntgr_irr.f90',
                     'src/AssbMatx.f90',    
                     'src/AssbMatx_irr.f90',
                     'src/SingularIntgr.f90',
                     'src/InfGreen_Appr.f90',
                     'src/FinGrnExtSubs.f90',
                     'src/FinGreen3D.f90',
                     'src/CalGreenFunc.f90',
                     'src/HydroStatic.f90',
                     'src/ImplementSubs.f90',
                     'src/InputFiles.f90',
                     'src/NormalProcess.f90',
                     'src/ReadPanelMesh.f90',
                     'src/PotentWavForce.f90',
                     'src/PressureElevation.f90',
                     'src/PrintOutput.f90',
                     'src/SolveMotion.f90',
                     'src/WavDynSubs.f90',
                     'src/HAMS_Prog.f90',
                     hams_connect,
                     fortranobject_c,
                     include_directories: inc_np,
                     dependencies : [py3_dep, lapack, omp],
                     subdir: 'pyhams',
		     link_language: 'fortran',
                     install : true)


