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
LN_COBYLA     NLopt        local    False         ???          =, < (NL)
LD_SLSQP      NLopt        local    True          ???          =, < (NL)
SNOPT         pyoptsparse  local    True          ???          =, < (NL)
CONMIN        pyoptsparse  local    True          ???          =, < (NL)
NSGA2         pyoptsparse  global   ???           ???          =, < (NL)
===========   ===========  ======   ===========   ==========   ===========

Key
~~~

*Scope:* does the method solve a local or global problem?

*Derivatives:* does the method use derivatives for optimization

*Convergent:* does the method have a convergence proof for its scope

*Constraints:* constraints that can be used with the method

   - equality constraints: ``=``
   - inequality constraints: ``<``
   - compatibility with linear constraints only: ``(L)``
   - compatibility with nonlinear constraints: ``(NL)``

..
.. Explanation of outputs
.. ----------------------
..
.. *TO DO!!!*
..
.. Some tips and best practices
.. ----------------------------
..
.. *TO DO!!!*
..


Optimization and parallel performance
-------------------------------------

In general, industral use of optimization is a straightfoward two-step process:

1) take a certain amount of resources (time, labor hours, computational resources, etc.)
2) use them to arrive at the best possible design

A goal of the WEIS project is to enable wider use of system-level optimization
by industrial offshore wind practicioners.
Towards this end, we can quantify two metrics of cost that are of key interest
to practicioners, in order to better understand the tradeoffs implicit in
running optimizations:

1) the total cost of a simulation: quantifies amount of energy used or billable computer use-hours
2) the wall-clock time necessary to run a simulation: "get me an answer by Friday"

We start by assuming that the driving computational cost is a system simulation
that requires :math:`T_{\mathrm{case}}` of irreducable simulation time (i.e., it
can not be reduced by parallelization or saavy computational efforts),
representing one period of simulation time for one realization of metocean
conditions.
We also assume that a user is interested in :math:`M_{\mathrm{case}}` cases,
totaled across the specifications within any given DLC and across all DLCs;
these can be run multiple times for a statisically representative result, with
the :math:`m`-th case being run :math:`N_{\mathrm{seed}}^{(m)}` times.

The progression of any optimization method will require some algorithm-dependent
number :math:`P` of evaluations to sample the design space within a single iteration; this can also be
parallelized:

- :math:`P=1` for gradient-free methods
- :math:`P=2 N_{\mathrm{DV}}` for gradient-based methods with centered finite differences approximation
  - :math:`P \sim N_{\mathrm{DV}}` for gradient-based methods with generic gradient approximation
  - :math:`P \sim 1` for gradient-based methods with analytical or adjoint-based gradients
- :math:`P=p_{\mathrm{evo}} N_{\mathrm{DV}}` for evolutionary methods
  - in practice, :math:`P` can be varied arbitrarily, but :math:`P \sim N_{\mathrm{DV}}` gives more consistent performance across problem size
  - optimal choice of :math:`p_{\mathrm{evo}}` can vary based on problem and method
  - :math:`p_{\mathrm{evo}}` between 5-10 is a common rule of thumb

Thus, any given iteration will require

.. math::
    M_{\mathrm{iter}} = P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right)

parallelizable simulations, with a total cost given by

.. math::
   \begin{aligned}
      C_{\mathrm{iter}} &= M_{\mathrm{iter}} T_{\mathrm{case}} \\
      &= P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) T_{\mathrm{case}}
   \end{aligned}

for the iteration.
Over :math:`N_{\mathrm{iter}}` iterations of the optimization algorithm, we
arrive at a total cost:

.. math::
   \begin{aligned}
      C_{\mathrm{total}} &= N_{\mathrm{iter}} C_{\mathrm{iter}} \\
      &= N_{\mathrm{iter}} M_{\mathrm{iter}} T_{\mathrm{case}} \\
      &= N_{\mathrm{iter}} P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) T_{\mathrm{case}} = C_{\mathrm{total}}
   \end{aligned}

In practice, this total cost is not equivalent to the wall-clock time to a
solution because within an interation, :math:`M_{\mathrm{iter}}` can be divided
across the number of parallel computing cores available in a machine
:math:`N_{\mathrm{cores}}`:

.. math::
   \begin{aligned}
      T_{\mathrm{iter}} &= \left\lceil \frac{M_{\mathrm{iter}}}{\min(M_{\mathrm{iter}}, N_{\mathrm{cores}})} \right\rceil T_{\mathrm{case}} \\
      &= \left\lceil \frac{P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right)}{\min \left( P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right), N_{\mathrm{cores}} \right) } \right\rceil T_{\mathrm{case}} \\
      &\approx \frac{P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) T_{\mathrm{case}}}{\min \left( P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right), N_{\mathrm{cores}} \right)}
   \end{aligned}

This allows for the total wall-clock time:

.. math::
      \begin{aligned}
        T_{\mathrm{total}} &= N_{\mathrm{iter}} T_{\mathrm{iter}} \\
        &\approx \frac{N_{\mathrm{iter}} P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) T_{\mathrm{case}}}{\min \left( P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right), N_{\mathrm{cores}} \right)}
      \end{aligned}

which gives two limiting cases:

- many more cores than cases, :math:`P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) \ll N_{\mathrm{cores}}`

   .. math::
      T_{\mathrm{total}} \approx N_{\mathrm{iter}} T_{\mathrm{case}} \not\sim N_{\mathrm{cores}}

- many more cases than cores, :math:`P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) \gg N_{\mathrm{cores}}`

   .. math::
      T_{\mathrm{total}} \approx \frac{N_{\mathrm{iter}} P \left( \sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} \right) T_{\mathrm{case}}}{N_{\mathrm{cores}}} \sim N_{\mathrm{cores}}^{-1}

Thus, when there's work to spread out across a computer, we get strong scaling,
approaching a best-case performance where the cost of an optimization is
:math:`T_{\mathrm{case}}` times the number of iterations.

With this dual perspective, we can see the intereactions between the problem to
be solved, which impacts the parallelizability and both costs; the choice of
algorithm, which impacts parallelizability, total work, the amount of iterations
necessary to achieve a sufficiently optimal result, and both cost metrics;
and the choice of computer, which can decrease the wall-clock time necessary to
get an optimization done.
These all come together to impact the effectiveness of a given optimization
strategy.


Optimization case study: IEA22 Platform optimization
----------------------------------------------------


In ``WEIS/examples/17_IEA22_Optimization``, we have an optimization
example which can be used to design the semisubmersible platform for the
IEA 22 280m reference wind turbine. We will concentrate on the files
``analysis_options_raft_ptfm_opt.yaml`` and
``modeling_options_raft.yaml``, which specify the platform design study.

The study sets design variables:
   - ``floating.joints``
      - ``z_coordinate[main_keel, col1_keel, col2_keel, col3_keel]``  (Changes the z-location of all these joints together, i.e., the platform draft)
      - ``r_coordinate[col1_keel, col1_freeboard, col2_keel, col2_freeboard, col3_keel, col3_freeboard]``  (Changes the radial location of all these joints together, i.e., the column spacing)
   - ``floating.members``
      - ``groups[column1, column2, column3]:diameter`` (Changes the diameter of all these members, i.e., the outer column diameter)


and constraints:
   - ``floating.survival_heel``: upper bound
      - maximum pitching heel allowable in parked conditions, used to compute ``draft_`` and ``freeboard_margin``
   - ``floating.metacentric_height``: lower bound
      - “Ensures hydrostatic stability with a positive metacentric height”
      - distance between center of gravity of a marine vessel and its metacenter (point between vessel-fixed vertical line through C.o.G. and inertial-frame-fixed line through center of buoyancy)
      - dictates static stability in the small-heel angle limit (i.e. characterizes stability)
   - ``floating.pitch_period``: upper & lower bound
      - period of the pitching motion (fore-aft rotation about center of mass)
   - ``floating.heave_period``: upper & lower bound
      - period of the heave (linear vertical motion of a marine vessel)
   - ``floating.fixed_ballast_capacity``: true/false
      - “Ensures that there is sufficient volume to hold the specified fixed (permanent) ballast”
   - ``floating.variable_ballast_capacity``: on
      - “Ensures that there is sufficient volume to hold the needed water (variable) ballast to achieve neutral buoyancy”
   - ``floating.freeboard_margin``: on
      - “Ensures that the freeboard (top points of structure) of floating platform stays above the waterline at the survival heel offset”
      - the deck surface should not be submerged in the worst-case conditions
   - ``floating.draft_margin``: on
      - “keep draft from raising above water line during survival_heel, largest wave”
      - the bottom of the hull should not rise above the water surface in the worst-case conditions
   - ``control.Max_PtfmPitch``: max
      - “Maximum platform pitch displacement over all cases. Can be computed in both RAFT and OpenFAST. The higher fidelity option will be used when active.”
   - ``control.Std_PtfmPitch``: max
      - “Maximum platform pitch standard deviation over all cases. Can be computed in both RAFT and OpenFAST. The higher fidelity option will be used when active.”
   - ``control.nacelle_acceleration``: max
      - “Maximum Nacelle IMU accelleration magnitude, i.e., sqrt(NcIMUTAxs^2 + NcIMUTAys^2 + NcIMUTAzs^2). Can be computed in both RAFT and OpenFAST. The higher fidelity option will be used when active.”

with a merit figure of the structural mass
   - ``structural_mass`` (``floatingse.system_structural_mass``)


Optimization results with RAFT modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From our modeling and analysis options:
   - The time to run an OpenFAST simulation, :math:`T_{solve}`, is about 30 seconds.
   - The number of cases is :math:`\sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} = 1`.  RAFT is actually running 14 DLCs (12 DLC 1.6 and 2 DLC 6.1, seeds are not necessary for RAFT), but they are not parallelized, so for the purposes of our cost/time estimates, the number of cases is 1.
   - The number of cores is :math:`N_{\mathrm{cores}} = 100`, and 
   - The number of design variables is :math:`N_{\mathrm{DVs}} = 3`.  WEIS does paralleize the runs across DVs for the SLSQP and DE solvers.

Thus, the number of cores is much more than the cases per iteration, and the time to convergence is relative to the number of iterations.

.. .. image:: /images/opt/Ptfm_OpenFAST_Conv.png
..    :width: 55%

.. |cost_of| |time_of|

.. .. |cost_of| image:: /images/opt/Ptfm_OpenFAST_Cost.png
..    :width: 45%

.. .. |time_of| image:: /images/opt/Ptfm_OpenFAST_Time.png
..    :width: 45%

Optimization results with OpenFAST modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From our modeling and analysis options:
   - The time to run an OpenFAST simulation, :math:`T_{solve}`, is about 10 minutes.
   - The number of cases is :math:`\sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} = 3`.
   - The number of cores is :math:`N_{\mathrm{cores}} = 100`, and 
   - The number of design variables is :math:`N_{\mathrm{DVs}} = 3`.

Thus, the number of cores is much more than the cases per iteration, and the time to convergence is relative to the number of iterations.

.. image:: /images/opt/Ptfm_OpenFAST_Conv.png
   :width: 55%

|cost_of| |time_of|

.. |cost_of| image:: /images/opt/Ptfm_OpenFAST_Cost.png
   :width: 45%

.. |time_of| image:: /images/opt/Ptfm_OpenFAST_Time.png
   :width: 45%

.. .. image:: /images/opt/Ptfm_OpenFAST_DE.png
..    :width: 55%



Optimization case study: IEA22 Controller optimization
-------------------------------------------------------

Here, the goal is to optimize the ROSCO pitch controller of the IEA-22MW RWT.

We use the following design variables, constraints, and merit figure:

This optimization varies the design variables:
   - ``control.servo.pitch_control.omega``, which controls the bandwidth (speed) of the pitch response to generator speed transients.  This value can be an array.  For the IEA-22MW controller, it has a length of 3.
   - ``control.servo.pitch_control.zeta``, sets the desired damping of the pitch response.   This value can be an array.  For the IEA-22MW controller, it has a length of 3.
   - ``control.servo.pitch_control.Kp_float``, which determines the floating feedback gain for damping platform motion
   - ``control.servo.pitch_control.ptfm_freq``, sets the low pass filter on the floating feedback loop

The merit figure of this optimization to be minimized is ``DEL_TwrBsMyt``, or the tower base damage equivalent load.  

We have two constraints:
   -  ``control.rotor_overspeed``: (flag, min, max)
      -  Over all load cases, the (maximum generator speed - rated generator speed) / (rated generator speed)
      -  Sometimes, larger values are requiered for feasible floating controllers
   -  ``user.name.aeroelastic.max_pitch_rate_sim``: (upper_bound)
      - Over all load cases, the maximum pitch rate normalized by the maximum allowed pitch rate
      - Unstable controllers often result in pitch commands saturated by the rate limit.  This constraint ensures solutions are stable in nonlinear simulations.


From our modeling and analysis options:
   - The time to run an OpenFAST simulation, :math:`T_{solve}`, is about 10 minutes.
   - The number of cases is :math:`\sum_{m=1}^{M_{\mathrm{case}}} N_{\mathrm{seed}}^{(m)} = 3`.
   - The number of cores is :math:`N_{\mathrm{cores}} = 100` for COBYLA, and , :math:`N_{\mathrm{cores}} = 400` for SLSQP and DE.
   - The number of design variables is :math:`N_{\mathrm{DVs}} = 8`.  ``omega`` and ``zeta`` are 3 each.

In this case, for COBYLA, the number of cores is more than the cases per iteration, so the time to convergence is relative to the number of iterations.
For the other solvers, the number of cases per iteration is less than the number of cores, so the time to convergences is greater.

.. .. image:: /images/opt/Ptfm_OpenFAST_Conv.png
..    :width: 55%

.. .. |cost_of| |time_of|

.. .. |cost_of| image:: /images/opt/Ptfm_OpenFAST_Cost.png
..    :width: 45%

.. .. |time_of| image:: /images/opt/Ptfm_OpenFAST_Time.png
..    :width: 45%




