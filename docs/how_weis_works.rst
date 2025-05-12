How WEIS works
--------------

WEIS is a stack of software that can be used for floating wind turbine design and analysis.  The turbine’s components (blade, tower, platform, mooring, and more) are parameterized in a geometry input. The system engineering tool `WISDEM <https://github.com/WISDEM/WISDEM>`_ determines component masses, costs, and engineering parameters that can be used to determine modeling inputs.  A variety of pre-processing tools are used to convert these system engineering models into simulation models.  

The modeling_options.yaml determines how the turbine is simulated.  We currently support `OpenFAST <https://github.com/OpenFAST/openfast>`_ and `RAFT <https://github.com/WISDEM/RAFT>`_.  

The analysis_options.yaml determine how to run simulations, either a single run for analysis, a sweep or design of experiments, or an optimization.  

A full description of the inputs can be found in the :ref:`inputs-documentation`.

.. image:: images/WEIS_Stack.png
  :width: 500
  :alt: Stack of software tools contained in WEIS

WEIS is `"glued" <https://github.com/WISDEM/WEIS/blob/master/weis/glue_code/glue_code.py>`_ together using `openmdao <https://openmdao.org/>`_ components and groups.

WEIS works best by starting with pre-existing examples, which are described here: :ref:`section-weis_examples`.

WEIS can be adapted to solve a wide variety of problems.  If you have questions or would like to discuss WEIS's functionality further, please email dzalkind (at) nrel (dot) gov. 

.. image:: images/WEIS_Flow.png
  :width: 1000
  :alt: Stack of software tools contained in WEIS
