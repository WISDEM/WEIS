"""

Example script to test MBC transform and related scripts

"""

import os
import platform
import unittest

import numpy as np

from weis.control.mbc.mbc3 import runMBC, IdentifyModes, IdentifiedModesDict

this_file_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_file_dir))

class TestMBC(unittest.TestCase):
    def test_run(self):
        filenames = [
            'lin_0.fst',
            'lin_1.fst',
        ]

        filenames = [os.path.join(this_file_dir,'linfiles',f) for f in filenames]

        CampbellData = runMBC(filenames)

        modeID_table,modesDesc=IdentifyModes(CampbellData)

        modesInfoPerOP = IdentifiedModesDict(CampbellData,modeID_table,modesDesc)

if __name__ == "__main__":
    unittest.main()
