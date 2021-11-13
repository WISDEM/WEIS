# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:38:05 2021

@author: shousner
"""


import unittest

import numpy as np
import numpy.testing as npt
import raft
import yaml
import os





class TestOC3Spar(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        
        raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        input_file = os.path.join(raft_dir,'designs/OC3spar.yaml')
        with open(input_file) as file:
            design = yaml.load(file, Loader=yaml.FullLoader)
        
        self.model = raft.Model(design)
        self.model.analyzeUnloaded()
        self.fowt = self.model.fowtList[0]
        self.pdiff = 0.01
    
    def test_tower_mass(self):
        actual_tower_mass = 249718
        tower_mass = self.fowt.mtower
        delta = actual_tower_mass*self.pdiff
        self.assertAlmostEqual(tower_mass, actual_tower_mass, delta=abs(delta))
        
    def test_tower_zCG(self):
        actual_tower_zCG = 43.4
        tower_zCG = self.fowt.rCG_tow[2]
        delta = actual_tower_zCG*self.pdiff
        self.assertAlmostEqual(tower_zCG, actual_tower_zCG, delta=abs(delta))
    
    def test_substruc_mass(self):
        actual_substruc_mass = 7466330
        substruc_mass = self.fowt.msubstruc
        delta = actual_substruc_mass*self.pdiff
        self.assertAlmostEqual(substruc_mass, actual_substruc_mass, delta=abs(delta))
        
    def test_substruc_zCG(self):
        actual_substruc_zCG = -89.9155
        substruc_zCG = self.fowt.rCG_sub[2]
        delta = actual_substruc_zCG*self.pdiff
        self.assertAlmostEqual(substruc_zCG, actual_substruc_zCG, delta=abs(delta))
    """
    def test_substruc_RI_CM(self):
        actual_substruc_RI_CM = 4229230000
        substruc_RI_CM = self.fowt.M_struc_subCM[3,3]
        delta = actual_substruc_RI_CM*self.pdiff
        self.assertAlmostEqual(substruc_RI_CM, actual_substruc_RI_CM, delta=abs(delta))
        
    def test_substruc_PI_CM(self):
        actual_substruc_PI_CM = 4229230000
        substruc_PI_CM = self.fowt.M_struc_subCM[4,4]
        delta = actual_substruc_PI_CM*self.pdiff
        self.assertAlmostEqual(substruc_PI_CM, actual_substruc_PI_CM, delta=abs(delta))
        
    def test_substruc_YI_CM(self):
        actual_substruc_YI_CM = 164230000
        substruc_YI_CM = self.fowt.M_struc_subCM[5,5]
        delta = actual_substruc_YI_CM*self.pdiff
        self.assertAlmostEqual(substruc_YI_CM, actual_substruc_YI_CM, delta=abs(delta))
    """
    def test_total_mass(self):
        actual_total_mass = 8066048
        total_mass = self.fowt.M_struc[0,0]
        delta = actual_total_mass*self.pdiff
        self.assertAlmostEqual(total_mass, actual_total_mass, delta=abs(delta))
        
    def test_total_zCG(self):
        actual_total_zCG = -77.97
        total_zCG = self.fowt.rCG_TOT[2]
        delta = actual_total_zCG*self.pdiff
        self.assertAlmostEqual(total_zCG, actual_total_zCG, delta=abs(delta))
        
    def test_buoyancy(self):
        actual_buoyancy = 80708100
        buoyancy = self.fowt.rho_water*self.fowt.g*self.fowt.V
        delta = actual_buoyancy*self.pdiff
        self.assertAlmostEqual(buoyancy, actual_buoyancy, delta=abs(delta))
        
    def test_C33(self):
        actual_C33 = 332941
        C33 = self.fowt.C_hydro[2,2]
        delta = actual_C33*self.pdiff
        self.assertAlmostEqual(C33, actual_C33, delta=abs(delta))
        
    def test_C44(self):
        actual_C44 = -4999180000
        C44 = self.fowt.C_hydro[3,3]
        delta = actual_C44*self.pdiff
        self.assertAlmostEqual(C44, actual_C44, delta=abs(delta))
        
    def test_C55(self):
        actual_C55 = -4999180000
        C55 = self.fowt.C_hydro[4,4]
        delta = actual_C55*self.pdiff
        self.assertAlmostEqual(C55, actual_C55, delta=abs(delta))
        
    def test_F_moor0(self):
        actual_F_moor0 = np.array([0,0,-1607000,0,0,0])
        F_moor0 = self.model.F_moor0
        delta = actual_F_moor0*self.pdiff
        #npt.assert_array_equal(F_moor0, actual_F_moor0, delta=delta)
        #npt.assert_allclose(F_moor0, actual_F_moor0, rtol=1000)
        #npt.assert_array_almost_equal(F_moor0, actual_F_moor0, decimal=-3)
        npt.assert_allclose(F_moor0, actual_F_moor0, atol=716)
    
    def test_C_moor0(self):
        actual_C_moor0 = np.array([[41180,0,0,0,-2821000,0],[0,41180,0,2821000,0,0],[0,0,11940,0,0,0],\
                                  [0,2816000,0,311100000,0,0],[-2816000,0,0,0,311100000,0],[0,0,0,0,0,11560000]])
        C_moor0 = self.model.C_moor0
        delta = actual_C_moor0*self.pdiff
        #self.assertAlmostEqual(C_moor0, actual_C_moor0, delta=delta)
        #npt.assert_array_almost_equal(C_moor0, actual_C_moor0, decimal=-7)
        npt.assert_allclose(C_moor0, actual_C_moor0, rtol=0.1, atol=1e5)
        # allclose works by ensuring (abs(C_moor0-actual_C_moor0) < atol+rtol*abs(actual_C_moor0))
        # unfortunately, our actual_C_moor0 has hard 0's, so no matter what the rtol is, the condition will never satisfy without an atol
        # therefore, since there's a term in C_moor that is 4e4, the atol needs to be greater than 4e4 for the given rtol
    
        
        
        
        
# NOTE: The moments of inertia are supposed to fail, since we don't have enough actual data to compare to
        
        
        
        


#self.assertEqual(sum([1,2,3]), 6, "Should be 6")
#assertTrue
#assertFalse
#assertIs
#assertIsNone
#assertIn
#assertIsInstance
# all can do nots too
#npt.assert_array_equal
#assertAlmostEqual
        

unittest.main()



