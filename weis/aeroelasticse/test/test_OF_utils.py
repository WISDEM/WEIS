import unittest
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.aeroelasticse.FAST_writer import InputWriter_OpenFAST
from weis.aeroelasticse.FAST_wrapper import FAST_wrapper
from weis.aeroelasticse.LinearFAST import LinearFAST
import os.path as osp
import platform

examples_dir    = osp.join( osp.dirname( osp.dirname( osp.dirname( osp.dirname( osp.realpath(__file__) ) ) ) ), 'examples')
weis_dir        = osp.dirname( osp.dirname( osp.dirname( osp.dirname(osp.realpath(__file__) ) ) ) ) # get path to this file


mactype = platform.system().lower()
if mactype == "linux" or mactype == "linux2":
    libext = ".so"
elif mactype == "darwin":
    libext = '.dylib'
elif mactype == "win32" or mactype == "windows": #NOTE: platform.system()='Windows', sys.platform='win32'
    libext = '.dll'
elif mactype == "cygwin":
    libext = ".dll"
else:
    raise ValueError('Unknown platform type: '+mactype)

class TestOFutils(unittest.TestCase):
    
    def setUp(self):
        # Initialize OF reader
        self.fast = InputReader_OpenFAST()
        self.fast.FAST_InputFile = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
        self.fast.FAST_directory = osp.join(examples_dir, '01_aeroelasticse',
                                                     'OpenFAST_models', 'IEA-15-240-RWT',
                                                     'IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
        self.fast.execute()

        self.fast.FAST_runDirectory = 'temp/OpenFAST'
        self.fast.FAST_namingOut    = 'iea15'

    def testOFreader(self):
        # Test the OF reader
        self.fast.execute()
                                                           
    def testOFwriter(self):
        # Test the OF writer
        fst_update = {}
        fst_update['Fst', 'TMax'] = 20.
        fst_update['AeroDyn15', 'TwrAero'] = False
        fst_update['ServoDyn', 'DLL_FileName'] = osp.join(weis_dir, 'local', 'lib', f'libdiscon{libext}')
        fastout = InputWriter_OpenFAST()
        fastout.fst_vt = self.fast.fst_vt
        fastout.update(fst_update=fst_update)
        fastout.FAST_runDirectory = self.fast.FAST_runDirectory
        fastout.FAST_namingOut    = self.fast.FAST_namingOut
        fastout.execute()

    def testWrapper(self):

        fast_wr = FAST_wrapper(debug_level=2)
        fast_wr.FAST_exe = osp.join(weis_dir,'local','bin','openfast')   # Path to executable
        fast_wr.FAST_InputFile = 'iea15.fst'   # FAST input file (ext=.fst)
        fast_wr.FAST_directory = self.fast.FAST_runDirectory   # Path to fst directory files
        fast_wr.execute()

    def testLinearFAST(self):

        lin_fast = LinearFAST(debug_level=2)
        lin_fast.FAST_exe = osp.join(weis_dir,'local','bin','openfast')   # Path to executable
        lin_fast.FAST_InputFile = 'iea15.fst'   # FAST input file (ext=.fst)
        lin_fast.FAST_directory = self.fast.FAST_runDirectory   # Path to fst directory files

        lin_fast.FAST_InputFile           = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
        lin_fast.FAST_directory           = osp.join(examples_dir, '01_aeroelasticse','OpenFAST_models','IEA-15-240-RWT','IEA-15-240-RWT-Monopile')   # Path to fst directory files
        lin_fast.FAST_runDirectory        = osp.join(weis_dir,'outputs','iea_mono_lin')
        
        lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
        lin_fast.wind_speeds                 = [16]
        lin_fast.DOFs                       = ['GenDOF','TwFADOF1'] #,'PtfmPDOF']  # enable with 
        lin_fast.TMax                       = 6   # should be 1000-2000 sec or more with hydrodynamic states

        lin_fast.gen_linear_model()


if __name__ == "__main__":
    unittest.main()
