Optimization in WEIS
====================

WEIS is platform for multidisciplinary analysis and optimization (MDAO),
based on the OpenMDAO toolset. WEIS leverages the `optimization
capabilities of
OpenMDAO <https://openmdao.org/newdocs/versions/latest/features/building_blocks/drivers/index.html>`_
and `extensions in
WISDEM <https://wisdem.readthedocs.io/en/master/inputs/analysis_schema.html#driver>`_
to perform multidisciplinary and multifidelity wind turbine analyses
including controls.

Available Optimization Solvers
------------------------------

from ScipyOptimizeDriver via OpenMDAO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO includes access to a handful of methods from the python
scipy minimization toolbox:

-  “SLSQP”

   -  Sequential Least Squares Quadratic Programming
   -  sequential estimation of objective function by quadratic functions
   -  application of constraints by linear approximation
   -  ``scipy.optimize.minimize`` implementation

-  “Nelder-Mead”

   -  downhill simplex method for optimization
   -  ``scipy.optimize.minimize`` implementation

-  “COBYLA”

   -  Constrained Optimization BY Linear Approximation
   -  sequential estimation of objective function by linear functions
   -  application of constraints by linear approximation
   -  derivative free estimation
   -  ``scipy.optimize.minimize`` implementation

from PyOptSparseDriver via OpenMDAO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO includes access to a handful of methods from the python
scipy minimization toolbox:

-  “SNOPT”

   -  `Sparse Nonlinear OPTimizer <https://web.stanford.edu/group/SOL/software/snoptHelp/About_SNOPT.htm>`__
   -  local quadratic subproblems
   -  constraints by linear approximation
   -  implements system sparsity for efficiency on high-dimensional problems
   -  linesearch on subproblem for performance
   -  *(requires inclusion of proprietary SNOPT software in pyoptsparse install)*

-  “CONMIN”

   -  CONstrained function MINimization
   -  method of feasible directions: chooses direction & step to
      minimize function while satisfying constraints

-  “NSGA2”

   -  Non-dominating Sorting Genetic Algorithm 2
   -  genetic global optimization algorithm

from NLOpt via WISDEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WISDEM has wrapped the NLOpt package for use in the OpenMDAO
toolset. NLOpt prefixes all solvers with a two-letter code; the
first letter indicates either a "G"lobal or "L"ocal method, and the
second letter indicates “N"o-derivative or “D"erivative-based
optimizers.

-  “GN_DIRECT”

   -  `DIviding RECTangles algorithm for global
      optimization <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#direct-and-direct-l>`__
   -  also, variants “GN_DIRECT_L” and “GN_DIRECT_L_NOSCAL”

-  “GN_ORIG_DIRECT”

   -  legacy implementation of “GN_DIRECT” with hardcoded details
   -  also, variant “GN_ORIG_DIRECT_L”

-  “GN_AGS”

   -  `AGS algorithm: derivative-free nonlinear provably-convergant
      global solver with dimensionality
      reduction <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#ags>`__

-  “GN_ISRES”

   -  `Improved Stochastic Ranking Evolution
      Strategy <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy>`__
   -  evolutionary algorithm

-  “LN_COBYLA”

   -  `Constrained Optimization BY Linear
      Approximation <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations>`__
   -  sequential estimation of objective function by linear functions
   -  application of constraints by linear approximation
   -  derivative free estimation
   -  NLOpt implementation

-  “LD_MMA”

   -  `Method of Moving
      Asymptotes <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa>`__
   -  penalized local approximation method with fast appoximation solves
   -  conservative convex separable approximation (CCSA) method using
      MMA approximation as inner loop optimizer

-  “LD_CCSAQ”

   -  `Conservative Convex Separable Approximation with Quadratic
      approximations <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa>`__
   -  separable method with guaranteed convergence to local optimizer

-  “LD_SLSQP”

   -  `Sequential Least Squares Quadratic
      Programming <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp>`__
   -  sequential estimation of objective function by quadratic functions
   -  application of constraints by linear approximation
   -  NLOpt implementation

Comparison of Optimizers
------------------------------

===========   ===========  ======   ===========   ==========   ===========

Solver        Toolset      Scope    Derivatives   Convergent   Constraints

===========   ===========  ======   ===========   ==========   ===========
SLSQP         scipy        local    True          ???          =, < (NL)
Nelder-Mead   scipy        local    False         False        None
COBYLA        scipy        local    False         ???          =, < (NL)
SNOPT         pyoptsparse  local    True          ???          =, < (NL)
CONMIN        pyoptsparse  local    True          ???          =, < (NL)
NSGA2         pyoptsparse  global   ???           ???          =, < (NL)
===========   ===========  ======   ===========   ==========   ===========

Explanation of outputs
----------------------

*TO DO!!!*

Some tips and best practices
----------------------------

*TO DO!!!*

Optimization case study: IEA22
==============================

In ``WEIS/examples/17_IEA22_Optimization``, we have an optimization
example which can be used to design the semisubmersible platform for the
IEA 22 280m reference wind turbine. We will concentrate on the files
``analysis_options_raft_ptfm_opt.yaml`` and
``modeling_options_raft.yaml``, which specify the platform design study.

The study sets design variables: - ``floating.joints`` -
``z_coordinate[main_keel, col1_keel, col2_keel, col3_keel]`` -
``r_coordinate[main_keel, col1_keel, col2_keel, col3_keel]`` - not sure
exactly what these do, but presumably they set cylindrical coordinates
of the truss system members (less an angle?) - ``floating.members`` -
``groups["column1, column2, column3]:diameter`` - presumably this is
setting the diameters of the truss system members? and constraints: -
``floating.survival_heel``: upper bound - maximum pitching heel
allowable in parked conditions - ``floating.metacentric_height``: lower
bound - “Ensures hydrostatic stability with a positive metacentric
height” - distance between center of gravity of a marine vessel and its
metacenter (point between vessel-fixed vertical line through C.o.G. and
inertial-frame-fixed line through center of buoyancy) - dictates static
stability in the small-heel angle limit (i.e. characterizes stability) -
``floating.pitch_period``: upper & lower bound - period of the pitching
motion (bow (stern) up vs. down rotation about center of mass) -
``floating.heave_period``: upper & lower bound - period of the heave
(linear vertical motion of a marine vessel) -
``floating.fixed_ballast_capacity``: on - “Ensures that there is
sufficient volume to hold the specified fixed (permanent) ballast” -
``floating.variable_ballast_capacity``: on - “Ensures that there is
sufficient volume to hold the needed water (variable) ballast to achieve
neutral buoyancy” - ``floating.freeboard_margin``: on - “Ensures that
the freeboard (top points of structure) of floating platform stays above
the waterline at the survival heel offset” - the deck surface should not
be submerged in the worst-case conditions - ``floating.draft_margin``:
on - “keep draft from raising above water line during survival_heel,
largest wave” - the bottom of the hull should not rise above the water
surface in the worst-case conditions - ``floating.fairlead_depth``: on -
“keep the fairlead above bottom trough of largest wave” - don’t dunk the
fairlead in worst-case conditions - ``control.Max_PtfmPitch``: max -
“Maximum platform pitch displacement over all cases. Can be computed in
both RAFT and OpenFAST. The higher fidelity option will be used when
active.” - ``control.Std_PtfmPitch``: max - “Maximum platform pitch
standard deviation over all cases. Can be computed in both RAFT and
OpenFAST. The higher fidelity option will be used when active.” -
``control.nacelle_acceleration``: max - “Maximum Nacelle IMU
accelleration magnitude, i.e., sqrt(NcIMUTAxs^2 + NcIMUTAys^2 +
NcIMUTAzs^2). Can be computed in both RAFT and OpenFAST. The higher
fidelity option will be used when active.” with a merit figure of the
structural mass ``structural_mass``
(``floatingse.system_structural_mass``).

.. raw:: html

   <!-- -->
