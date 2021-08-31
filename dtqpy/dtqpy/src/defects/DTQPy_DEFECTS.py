# -*- coding: utf-8 -*-
"""
DTQPy_DEFECTS
Create sparse matrix for the defect constraints

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

from dtqpy.src.defects.DTQPy_DEFECTS_TR import DTQPy_DEFECTS_TR

def DTQPy_DEFECTS(A,B,G,d,internal,opts):
    
    # Trapezoidal rule
    if opts.dt.defects == 'TR':
        [Aeq,beq] = DTQPy_DEFECTS_TR(A,B,G,d,internal,opts)
    
    return Aeq,beq