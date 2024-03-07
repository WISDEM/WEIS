These examples run OpenFAST cases using the `aeroelasticse` drivers.

By setting up `case_input` dicts, `OpenFAST` parameters can be swept.

For example, [we sweep the wind speed here](https://github.com/WISDEM/WEIS/blob/bdf7cea127d99a707d565178ef1edbf65b7f635a/examples/01_aeroelasticse/run_general.py#L52) and can change the initial conditions, like [blade pitch](https://github.com/WISDEM/WEIS/blob/bdf7cea127d99a707d565178ef1edbf65b7f635a/examples/01_aeroelasticse/run_general.py#L55) to vary with the wind speed, by assigning them to the same group.
