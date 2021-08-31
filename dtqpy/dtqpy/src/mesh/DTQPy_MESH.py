# -*- coding: utf-8 -*-
"""
DTQPy_MESH
Solve the problem using different mesh refiniment schemes

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
from dtqpy.src.DTQPy_multiphase import DTQPy_multiphase

def DTQPy_MESH(setup,opts):
    if opts.dt.meshr.method == 'None':
        T,U,Y,P,F,internal,opts = DTQPy_multiphase(setup,opts)
        
        
    return T,U,Y,P,F,internal,opts