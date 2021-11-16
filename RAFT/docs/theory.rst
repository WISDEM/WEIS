Theory
=====================

The theory behind RAFT is in the process of being documented. This page will be incrementally updated.


Frequency Domain Equations of Motion
------------------------------------

The overall process of RAFT is to use the input design data of a FOWT to fill out the 6x6 matrices of the equations of motion.

.. math::

   [M+A]\ddot{X}(t) + [B+B_{visc}]\dot{X}(t) + [C_{struc}+C_{hydro}+C_{moor}]X(t) = F_{env}(t)

   F(t) = Re\{\tilde{F}e^{iwt}\}

   \tilde{F} = \|F\|e^{i\phi}

   X = Xe^{iwt}

   \dot{X} = iwXe^{iwt}

   \ddot{X} = -w^2Xe^{iwt}

   X(\omega) = [-\omega^2A(\omega) + i \omega B(\omega) + C]^{-1} F(\omega)

   A(\omega) = M + A_{BEM}(\omega) + A_{morison}(\omega) + A_{aero}(\omega)
   
   B(\omega) = B_{BEM}(\omega) + B_{aero}(\omega) + B_{nonlinear-hydro-drag}(X)

   C = C_{struc} + C_{hydro} + C_{moor}

   F = F_{BEM}(\omega) + F_{hydro}(\omega) + F_{aero}(\omega) + F_{nonlinear-hydro-drag}(X)

Notice that the nonlinear-hydro-drag damping and forcing terms are a function of the platform positions, so these are iterated
until the positions (X) converge.


Member Theory
-------------

.. image:: /images/members1.JPG
    :align: center

.. image:: /images/members2.JPG
    :align: center
    :width: 350px
    :height: 350px

Mass & Inertia
^^^^^^^^^^^^^^
The 6x6 mass matrix is calculated for each member using the properties given in the input design yaml.
Members can be cylindrical or rectangular

Hydrostatics
^^^^^^^^^^^^
The 6x6 hydrostatic stiffness matrix is calculated for each member using the properties given in the input design yaml.

Hydrodynamics
^^^^^^^^^^^^^
The 6x6 hydrodynamic matrices for each member are calculated in one of two ways: a BEM solver, pyHAMS, or a Morison equation approximation



Rotor Theory
------------

The Rotor uses aerodynamic theory from WISDEM's CCBlade. Check out the `CCBlade documentation <https://wisdem.readthedocs.io/en/latest/wisdem/ccblade/index.html>`_ 
for more infomation.

Mooring Theory
--------------

RAFT uses the quasi-static mooring system modeler, MoorPy. Check out `the MoorPy documentation <https://moorpy.readthedocs.io/en/latest/>`_ 
for more information.
