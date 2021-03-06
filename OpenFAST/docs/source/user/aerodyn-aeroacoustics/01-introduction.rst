.. _AA-introduction:

Introduction
------------

The increasing penetration of wind energy into the electricity mix has
been possible thanks to a constantly growing installed capacity, which
has so far been mostly located on land. Land-based installations are,
however, increasingly constrained by local ordinances and an
often-limiting factor that comprises maximum allowable levels of noise.
To further increase the number of land-based installations, it is
important to develop accurate modeling tools to estimate the noise
generated by wind turbines. This allows for a more accurate assessment
of the noise emissions and the possibility to design quieter wind
turbines.

Wind turbines emit two main sources of noise:

-  Aeroacoustics noise from the interaction between rotor blades and the
   turbulent atmospheric boundary layer

-  Mechanical noise from the nacelle component, mostly the gearbox,
   generator, and yaw mechanism.

This work targets the first class of noise generation and aims at
providing a set of open-source models to estimate the aeroacoustics
noise generated by an arbitrary wind turbine rotor. The models are
implemented in Fortran and are fully coupled to the aeroservoelastic
wind turbine simulator OpenFAST. The code is available in the GitHub
repository of OpenFAST. [1]_ The code builds on the implementation of
NAFNoise and the documentation presented in :cite:`aa-MoriartyMigliore:2003`
and :cite:`aa-Moriarty:2005`. OpenFAST is implemented as a modularization
framework and the aeroacoustics model is implemented as a submodule of
AeroDyn (:cite:`aa-MoriartyHansen:2005`).

The set of models is described in :numref:`AA-noise-models` and exercised on the
noise estimate of the International Energy Agency (IEA) land-based reference
wind turbine in :numref:`AA-model-verification`. In
:numref:`AA-model-verification`, we also show a comparison to results obtained
running the noise models implemented at the Technical University of Munich. This
documentation closes with conclusions, an outlook on future work, and
appendices, where the input files to OpenFAST are presented.


.. [1]
   https://github.com/OpenFAST/openfast

