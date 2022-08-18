Here are a set of jupyter notebooks for postprocessing WEIS results.  They run using a set of pre-defined output files that can be downloaded from here, or with the following set of commands

```
gdown --no-check-certificate 1_FJaN-W1DoPNmO6YLSjnftvq6-7bn4VI
unzip outputs
```

`plot_FAST.ipynb` is used to plot OpenFAST outputs.

`rev_DLCs_WEIS.ipynb` is used to review aggregate statistics generated using pCrunch in WEIS

`rev_Opt.ipynb` is used to review optimization iterations generated from openmdao logs

`rev_WEIS_CSV.ipynb` is use to review the csv output files of WEIS