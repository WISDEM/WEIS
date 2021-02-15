#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=4:00:00
#SBATCH --job-name=floating
#SBATCH --nodes=2             # This should be nC/36 (36 cores on eagle)
#SBATCH --ntasks-per-node=36
#SBATCH --mail-user dzalkind@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=output.%j.out
##SBATCH --partition=debug

nDV=2 # Number of design variables (x2 for central difference)
nOF=1  # Number of openfast runs per finite-difference evaluation
nC=$((nDV + nDV * nOF)) # Number of cores needed. Make sure to request an appropriate number of nodes = N / 36
## nC=72

source activate /home/dzalkind/.conda-envs/weis-env4
which python

# module purge
# module load conda
# module load comp-intel intel-mpi mkl


mpirun -v -np $nC python weis_driver.py
#  python weis_driver.py
