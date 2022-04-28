#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Jan 22 23:56 2014
I'm not really the author, but...
@author: Sammy Pfeiffer
@email: sammypfeiffer@gmail.com
File found at:
http://projects.scipy.org/scipy/attachment/ticket/393/invfreq.py
Cleaned up a bit, removed unused imports.
"""


import numpy
from numpy import atleast_1d, real
from scipy.linalg import solve


def invfreqs(g, worN, nB, nA, wf=None, nk=0):
    """Compute frequency response of a digital filter.
    Description:
       Computes the numerator (b) and denominator (a) of a digital filter compute
       its frequency response, given a frequency response at frequencies given in worN.
                             nB      nB-1                             nk
               B(s)    (b[0]s + b[1]s   + .... + b[nB-1]s + b[nB])s
        H(s) = ---- = -----------------------------------------------
                            nA      nA-1
               A(s)    a[0]s + a[1]s + .... + a[nA-1]s+a[nA]
        with a[0]=1.
       Coefficients are determined by minimizing sum(wf |B-HA|**2).
       If opt is not None, minimization of sum(wf |H-B/A|**2) is done in at most
       MaxIter iterations until norm of gradient being less than Tol,
       with A constrained to be stable.
    Inputs:
       worN -- The frequencies at which h was computed.
       h -- The frequency response.
    Outputs: (w,h)
       b, a --- the numerator and denominator of a linear filter.
    """
    g = atleast_1d(g)
    worN = atleast_1d(worN)
    if any(wf==None):
        wf = numpy.ones_like(worN)
    if len(g)!=len(worN) or len(worN)!=len(wf):
        raise ValueError("The lengths of g, worN and wf must coincide.")
    if numpy.any(worN<0):
        raise ValueError("worN has negative values.")
    s = 1j*worN

    # Constraining B(s) with nk trailing zeros
    nm = numpy.maximum(nA, nB+nk)
    mD = numpy.vander(1j*worN, nm+1)
    mH = numpy.mat(numpy.diag(g))
    mM = numpy.mat(numpy.hstack(( mH*numpy.mat(mD[:,-nA:]),\
            -numpy.mat(mD[:,-nk-nB-1:][:,:nB+1]))))
    mW = numpy.mat(numpy.diag(wf))
    Y = solve(real(mM.H*mW*mM), -real(mM.H*mW*mH*numpy.mat(mD)[:,-nA-1]))
    a = numpy.ones(nA+1)
    a[1:] = Y[:nA].flatten()
    b = numpy.zeros(nB+nk+1)
    b[:nB+1] = Y[nA:].flatten()

    return b,a