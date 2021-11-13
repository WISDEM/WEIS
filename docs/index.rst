

RAFT
====

Welcome to RAFT's online documentation!

RAFT (**R**\ esponse **A**\ mplitudes of **F**\ loating **T**\ urbines) is a 
Python code for frequency-domain analysis of floating wind turbines. RAFT works in 
the frequency domain to provide efficient computations of a floating wind 
system's response spectra accounting for platform hydrodynamics, quasi-static
mooring reactions, rotor aerodynamics, and linearized control. 
It can be used to calculate response amplitude
operators (RAO's), power spectral densities, mean/static properties, and 
response metrics based on mean and root-mean-square values of each output.

RAFT serves as the lowest fidelity "Level 1" model in the WEIS 
(Wind Energy with Integrated Servo-control) 
toolset for multifidelity co-design of wind turbines. WEIS is a framework that 
combines multiple NREL-developed tools to enable design optimization of floating 
offshore wind turbines. See the `WEIS documentation <https://weis.readthedocs.io>`_ 
and `GitHub repository <https://github.com/WISDEM/WEIS>`_ for more information.

Following a modular philosophy, RAFT can be used independently for various
frequency-domain modeling needs. Or, it can be used as an integrated part of
WEIS through its OpenMDAO wrapper. 

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Home <self>
   starting
   structure
   usage
   theory


The pages available in the menu on the left provide more information about RAFT, and
the RAFT source code is available on GitHub `here <https://github.com/WISDEM/RAFT>`_.

