import os
import unittest
import numpy as np
import numpy.testing as npt

import pyhams.pyhams as pyhams


class CertTest(unittest.TestCase):
    def test_cylinder(self):

        cog = np.zeros(3)
        mass = np.array([[0.16040E+04,   0.00000E+00,   0.00000E+00,   0.00000E+00,  -0.60123E+03,  -0.37043E-13],
                         [ 0.00000E+00,   0.16040E+04,   0.00000E+00,   0.60123E+03,   0.00000E+00,   0.51477E-13],
                         [ 0.00000E+00,   0.00000E+00,   0.16040E+04,   0.37043E-13,  -0.51477E-13,   0.00000E+00],
                         [ 0.00000E+00,   0.60123E+03,   0.37043E-13,   0.86185E+03,  -0.14322E-13,   0.24425E-14],
                         [ -0.60123E+03,   0.00000E+00,  -0.51477E-13,  -0.14322E-13,   0.86185E+03,  -0.24980E-14],
                         [ -0.37043E-13,   0.51477E-13,   0.00000E+00,   0.24425E-14,  -0.24980E-14,   0.11931E+04]])
        dampingLin = np.zeros((6,6))
        dampingQuad = np.zeros((6,6))
        kHydro = np.array([[0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00],
                           [ 0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00],
                           [ 0.00000E+00,   0.00000E+00,   0.31462E+05,   0.45338E-12,   0.67571E-12,   0.00000E+00],
                           [ 0.00000E+00,   0.00000E+00,   0.45338E-12,   0.97116E+04,  -0.15694E-12,   0.52228E-12],
                           [ 0.00000E+00,   0.00000E+00,   0.67571E-12,  -0.15694E-12,   0.97116E+04,   0.53473E-12],
                           [ 0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00,   0.00000E+00]])
        kExt = np.zeros((6,6))

        mypath = 'Cylinder'
        pyhams.create_hams_dirs(baseDir=mypath)
        pyhams.write_hydrostatic_file(projectDir=mypath, cog=cog, mass=mass,
                                      dampingLin=dampingLin, dampingQuad=dampingQuad,
                                      kHydro=kHydro, kExt=kExt)
        pyhams.write_control_file(projectDir=mypath, waterDepth=10.0, incFLim=1, iFType=1, oFType=4, numFreqs=-30,
                                  minFreq=0.1, dFreq=0.1, numHeadings=36,
                                  minHeading=0.0, dHeading=10.0,
                                  refBodyCenter=[0.0, 0.0, 0.0], refBodyLen=1.0, irr=1,
                                  numThreads=6)
        pyhams.run_hams(mypath)

        # Ensure reading goes smoothly
        pathWamit1 = os.path.join(mypath, 'Output', 'Wamit_format', 'Buoy.1')
        pathWamit3 = os.path.join(mypath, 'Output', 'Wamit_format', 'Buoy.3')
        addedMass, damping, w = pyhams.read_wamit1(pathWamit1)
        mod, phase, real, imag, w, headings = pyhams.read_wamit3(pathWamit3)

        actual1 = np.loadtxt(pathWamit1, skiprows=72)
        actual3 = np.loadtxt(pathWamit3)
        truth1 = np.loadtxt(os.path.join(mypath, 'truth', 'Buoy.1'))
        truth3 = np.loadtxt(os.path.join(mypath, 'truth', 'Buoy.3'))

        # Test freqs
        npt.assert_array_almost_equal(np.unique(truth1[:,0]), np.unique(actual1[:,0]))
        npt.assert_array_almost_equal(np.unique(truth1[:,0]), w)

        # Added mass and damping
        npt.assert_array_almost_equal(truth1, actual1)
        
        #npt.assert_array_almost_equal(truth3, actual3)



def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CertTest))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
