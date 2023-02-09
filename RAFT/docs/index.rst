
.. toctree::
   :maxdepth: 2
   :hidden:
   
   Home <self>
   starting
   structure
   usage
   theory


RAFT Documentation
==================

RAFT (**R**\ esponse **A**\ mplitudes of **F**\ loating **T**\ urbines) is a 
Python code for frequency-domain analysis of floating wind turbines. RAFT uses 
quasi-static and frequency-domain models to provide efficient computations of 
a floating wind system's response spectra accounting for platform hydrodynamics, 
mooring reactions, rotor aerodynamics, and turbine control. 
It can be used to calculate response amplitude
operators (RAO's), power spectral densities, mean/static properties, and 
response metrics based on mean and root-mean-square values of each output.

.. figure:: /images/illustrations.png
    :align: center
	
    Three reference designs represented in RAFT

RAFT serves as the lowest fidelity "Level 1" dynamics model in the WEIS 
(Wind Energy with Integrated Servo-control) 
toolset for control co-design of floating wind turbines. WEIS is a framework that 
combines multiple NREL-developed tools to enable design optimization of floating 
offshore wind turbines. See the `WEIS documentation <https://weis.readthedocs.io>`_ 
and `GitHub repository <https://github.com/WISDEM/WEIS>`_ for more information.

Following a modular philosophy, RAFT can be used independently for various
frequency-domain modeling needs. Or, it can be used as an integrated part of
WEIS through its OpenMDAO wrapper. The pages in this documentation site provide
some guidance on setting up and using RAFT, as well as understanding how RAFT
works. 
RAFT is available from the `RAFT GitHub Repository <https://github.com/WISDEM/RAFT>`_
and is released under the Apache 2.0 open-source license. 

The following paper provides an overview of RAFT's theory:
 
`M. Hall, S. Housner, D. Zalkind, P. Bortolotti, D. Ogden, and G. Barter, 
“An Open-Source Frequency-Domain Model for Floating Wind Turbine Design Optimization,” 
Journal of Physics: Conference Series, vol. 2265, no. 4, p. 042020, May 2022, DOI: 10.1088/1742-6596/2265/4/042020. <https://iopscience.iop.org/article/10.1088/1742-6596/2265/4/042020>`_
