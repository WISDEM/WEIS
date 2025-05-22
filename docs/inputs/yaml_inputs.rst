.. _inputs-documentation:

WEIS Inputs
====================

The WEIS input schemas are a merged version of the `WEIS <https://github.com/WISDEM/WEIS/tree/develop/weis/inputs>`_ and `WISDEM <https://github.com/WISDEM/WISDEM/tree/master/wisdem/inputs>`_ input schemas.

Geometry schemas are shared across fidelity levels. 
Most of the analysis options are from the WISDEM schema, but there are a few additional options in the WEIS schema.

In general, it's a good idea to start with working examples and change inputs within that framework.  If additional inputs are required, then consult these input schemas.


.. only:: html

    Inputs are divided into three different files:

    - *Geometry file* describes the physical turbine
    - *Modeling file* describes the modeling equations applied and the discretization
    - *Analysis file* describes the how the model is executed in an analysis

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   geometry_schema
   modeling_schema
   analysis_schema
