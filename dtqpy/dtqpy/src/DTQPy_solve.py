# -*- coding: utf-8 -*-
"""
DTQPy_solve
Solve the dynamic optimztion problem

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from dtqpy.src.mesh.DTQPy_MESH import DTQPy_MESH

def DTQPy_solve(setup,opts):
    # Check matrix structure
    setup.Check_Struct()
    setup.Check_Matrix_shape()
    
    # DTQP mesh
    T,U,Y,P,F,internal,opts = DTQPy_MESH(setup,opts)
    
    # return results
    return T,U,Y,P,F,internal,opts