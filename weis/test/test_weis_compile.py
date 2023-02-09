"""

Example script to compute the steady-state performance in OpenFAST

"""

import os
import platform
import unittest

import numpy as np


this_file_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_file_dir))

class TestGeneral(unittest.TestCase):
    def test_pyhams(self):
        import pyhams._hams

    def test_wisdem(self):
        import wisdem.ccblade


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGeneral))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)