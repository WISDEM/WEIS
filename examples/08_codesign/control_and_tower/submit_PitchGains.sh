#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=6:00:00
#SBATCH --job-name=control_and_tower
#SBATCH --nodes=1             # This should be nC/36 (36 cores on eagle)
#SBATCH --ntasks-per-node=36
#SBATCH --mail-user john.jasa@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=iea15.%j.out
#####SBATCH --partition=debug


nDV=5 # Number of design variables (x2 for central difference)
nOF=6  # Number of openfast runs per finite-difference evaluation
nC=$((nDV + nDV * nOF)) # Number of cores needed. Make sure to request an appropriate number of nodes = N / 36

source activate weis-env 

mpirun -v -np $nC python run_optimization.py
