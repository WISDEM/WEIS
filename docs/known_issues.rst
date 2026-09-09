.. _known_issues:

Known issues within WEIS
========================

This doc page serves as a non-exhaustive record of any issues relevant to the usage of WISDEM.
Some of these items are features that would be nice to have, but are not necessarily in the pipeline of development.

Running on Eagle
----------------
Depending on the method that send batch scripts to Eagle, they may not run correctly in parallel.
Specifically, when calling ``sbatch submit_job.sh`` and the job involves MPI and WEIS, multiple users have reported issues.
These issues manifest as the script starting correctly, but then returning ``MPI_INIT`` errors.

One user could not successfully run jobs submitted via VS Code, but could via terminal.
For another user, the regular Windows command terminal worked, but not the Ubuntu subsystem.
**If you have any issues regarding running scripts on Eagle, first try a few different terminals to submit sbatch jobs**.

The reason for this error is not known.

Realized TI exceeds the requested TI for MHK (TurbSim ``TIDAL``) runs
---------------------------------------------------------------------

**Status: suspected TurbSim bug, not confirmed with upstream. Observed against
OpenFAST v4.2.0.**

For MHK turbines WEIS drives TurbSim with ``TurbModel = TIDAL``, where turbulence
is set by ``UStar`` rather than ``IECturbc`` (under ``TIDAL``, ``IECturbc`` and
``ScaleIEC`` are both ignored). ``Spec_TIDAL`` defines

.. code-block:: text

   Sigma_U2 = 4.5 * UStar^2 * exp(-2*Ht/RefHt)

which at hub height with ``HubHt == RefHt`` gives ``TI = 0.7804*UStar/URef``,
i.e. ``UStar = 1.2814*TI*URef``. That inversion is what
``weis/dlc_driver/dlc_generator_mhk.py`` uses to set ``UStar`` from
``current_TI_NTM``.

**Symptom.** The realized standard deviation in the generated ``.bts`` does not
match this target. It is *low* for short records and *high* for long ones, and
the discrepancy grows with height above the seabed. For an RM1 case requesting
9% TI at hub elevation:

=============  ==============  ====================
AnalysisTime   TI at z = 26 m  vs. requested (9.0%)
=============  ==============  ====================
120 s          11.4%           +27%
720 s          12.3%           +37%
1200 s         10.5%           +16%
=============  ==============  ====================

Two independent effects of opposite sign are in play, and they partially cancel
at short record lengths.

**1. Finite-record truncation (expected behavior, not a bug).**
``CalcTargetPSD`` discretizes the spectrum as ``S = SSVS*HalfDelF`` with no
renormalization, so realized variance is the spectrum summed over resolved
frequencies ``f_i = i/AnalysisTime`` only. The ``TIDAL`` spectrum is flat below
a knee at ``f_knee = (du/dz) * b1^(-3/5)``, which for typical tidal shear is
around 0.005-0.01 Hz, i.e. a period of 100-150 s. A 20 s record resolves nothing
below 0.05 Hz and therefore misses the entire energy-containing range:

============  ==============  ====================
AnalysisTime  var. resolved   sigma / sigma_target
============  ==============  ====================
20 s          32%             0.56
60 s          59%             0.77
300 s         90%             0.95
600 s         95%             0.97
1200 s        97%             0.99
============  ==============  ====================

Note that ``TimeSeriesScaling_IEC`` exists specifically to "account for
discretizing the spectra over a finite length of time", but ``TIDAL`` is routed
to ``TimeSeriesScaling_ReynoldsStress`` instead and so receives no such
correction. **Use** ``analysis_time`` **of at least 600 s for MHK DLCs.**

**2. Height-dependent variance leakage in the coherence factorization
(the suspected bug).**
In ``Coh2H`` (``modules/turbsim/src/TSsubs.f90``), the off-diagonal entries of
Veers' ``H`` matrix are scaled by the *geometric mean* of the two points'
spectra:

.. code-block:: fortran

   !         TRH(Indx) = TRH(Indx) * SQRT( ABS( S(IFreq,I,IVec) ) )
            TRH(Indx) = TRH(Indx) * SQRT( SQRT( ABS( S(IFreq,I,IVec) * S(IFreq,J,IVec) ) ) )

Recovering the target spectra from ``H = diag(sqrt(S)) * chol(Coh)`` requires
scaling by **row only**, ``sqrt(S_I)`` -- which is the commented-out line
directly above. With the geometric-mean form,
``E|V_I|^2 = S_I * sum_J L_IJ^2 * sqrt(S_J/S_I)``, so points whose neighbors
have larger variance inherit some of it.

This is invisible for the IEC models, where ``S`` does not vary with height and
the two expressions are algebraically identical. ``TIDAL`` is the model whose
``S`` varies strongly with height (``exp(-2*Ht/RefHt)``), so it is the one that
exposes the asymmetry. ``git log -L`` dates the change to a 2014 refactoring
commit whose message describes error handling and loop reordering, with no
mention of the spectral scaling change.

Measured against a 1200 s ``TIDAL`` field (truncation < 2%, so effect 1 is
negligible), the realized-to-target ratio climbs monotonically with height:

======  =========================
Height  sigma_real / sigma_target
======  =========================
9 m     1.01
18 m    1.11
27 m    1.16
36 m    1.26
63 m    1.70
======  =========================

while at a fixed height the realized PSD matches the analytic spectrum exactly
in the inertial subrange and is inflated only at low frequency, where coherence
is high:

=============  ==================
Band (Hz)      realized/analytic
=============  ==================
0.010 - 0.020  2.85
0.020 - 0.050  3.13
0.050 - 0.200  1.35
0.200 - 1.0    1.00
1.0 - 10.0     1.00
=============  ==================

This can be reproduced independently of TurbSim:

.. code-block:: python

   import numpy as np
   z = np.linspace(9., 63., 25)
   S = 4.5 * 0.173**2 * np.exp(-2 * z / 26.)     # TIDAL Sigma_U2 profile
   C = np.exp(-np.abs(z[:, None] - z[None, :]) / 20.)
   L = np.linalg.cholesky(C)

   correct = np.sqrt(S)[:, None] * L             # row scaling: exact
   turbsim = L * (S[:, None] * S[None, :])**0.25 # geometric mean
   turbsim[np.diag_indices_from(turbsim)] = np.diag(L) * np.sqrt(S)

   print(np.sqrt((correct**2).sum(1) / S))       # 1.00 at every height
   print(np.sqrt((turbsim**2).sum(1) / S))       # 1.00 at bottom -> 1.24 at top

**Consequences for MHK.** The magnitude depends on how much variance contrast
the grid spans, so it is coupled to grid sizing. WEIS sizes the MHK grid to
cover all AeroDyn tower/rotor, HydroDyn joint, and SeaState ``WaveKin`` nodes
(SeaState queries InflowWind at every node *before* the submergence mask is
applied, so out-of-water nodes must be inside the box). For RM1 this gives a
54 m box spanning roughly a 63x variance contrast, producing +11% to +26% over
the rotor span. Both the TI *level* and the turbulence *shear* across the rotor
disk are affected.

**Workarounds.**

* Use ``analysis_time >= 600.`` so the truncation deficit does not mask the
  excess, and use multiple seeds -- single-seed sigma scatter is around +/-14%
  at 600 s even with a perfect spectrum.
* Do not attempt to compensate by biasing ``UStar``. Under ``TIDAL``, ``UStar``
  also sets the H2L mean shear (``U(z) = ln(z/z_ref)*UStar/0.41 + U_ref``) and
  ``PC_UW``, so TI cannot be tuned without corrupting the mean profile and the
  Reynolds stress. The required correction is also grid- and height-dependent,
  so no single scalar is correct across the rotor.
* If exact TI is required, patch ``Coh2H`` locally to use the row-scaled form
  (restore line 811, remove line 812) and rebuild TurbSim.
