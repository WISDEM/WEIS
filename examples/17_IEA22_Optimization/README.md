
# Example 17: IEA22 FOWT Optimization

This is an example of optimization and post-processing of an IEA 22MW RWT-based FOWT system.

## RAFT-based optimization

### Data generation

To run the cases, we use the standard WEIS setup, driven by `driver_weis_raft_opt.py`.
The driver leverages an analysis options file, `analysis_options_raft_ptfm_opt.yaml`, and the modeling options file `modeling_options_raft.yaml`.

In order to run different optimizers, we can edit `analysis_options_raft_ptfm_opt.yaml:127`, which reads
```
solver: LN_COBYLA
```
for the COBYLA optimizer.
We can switch this line to either of the following lines to run SLSQP or differential evolution (DE):
```
solver: LD_SLSQP
solver: DE
```

Once an optimization terminates, an output directory of the name `17_IEA22_Opt_Result` will be created and populated with data and metadata.
We recommend running terminal command such as:
```
mv 17_IEA22_Opt_Result 17_IEA22_Opt_Result_CASENAME
```
where `CASENAME` is replaced by `COBYLA`, `SLSQP`, and/or `DE` depending on the case you are running.
Alternately, standard move operations for a user's operating system of choice can be used.

## Analysis

Once `17_IEA22_Opt_Result_COBYLA`, `17_IEA22_Opt_Result_SLSQP`, and `17_IEA22_Opt_Result_DE` are populated, `analysis.ipynb` can be used to evaluate the results.
The notebook has detailed descriptions of its analysis.

## OpenFAST-based optimization

TO BE COMPLETED...
