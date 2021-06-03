import unittest
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.aeroelasticse.FAST_writer import InputWriter_OpenFAST
import numpy as np
import os

examples_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep

class TestOFutils(unittest.TestCase):
    
    def setUp(self):
        # Initialize OF reader
        self.fast = InputReader_OpenFAST()
        self.fast.FAST_InputFile = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
        self.fast.FAST_directory = os.path.join(examples_dir, 'examples', '01_aeroelasticse',
                                                     'OpenFAST_models', 'IEA-15-240-RWT',
                                                     'IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
        self.fast.execute()

    def testOFreader(self):
        # Test the OF reader
        self.fast.execute()
                                                           
    def testOFwriter(self):
        # Test the OF writer
        fst_update = {}
        fst_update['Fst', 'TMax'] = 20.
        fst_update['AeroDyn15', 'TwrAero'] = False
        fastout = InputWriter_OpenFAST()
        fastout.fst_vt = self.fast.fst_vt
        fastout.FAST_runDirectory = 'temp/OpenFAST'
        fastout.FAST_namingOut = 'iea15'
        fastout.update(fst_update=fst_update)
        fastout.execute()


if __name__ == "__main__":
    unittest.main()
