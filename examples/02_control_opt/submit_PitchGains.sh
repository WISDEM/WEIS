#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=1:00:00
#SBATCH --job-name=servoopt_iea15
#SBATCH --nodes=2             # This should be nC/36 (36 cores on eagle)
#SBATCH --ntasks-per-node=36
#SBATCH --mail-user dzalkind@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=job_servoopt_iea15.%j.out

#SBATCH --partition=short

nDV=4 # Number of design variables (x2 for central difference)
nOF=1  # Number of openfast runs per finite-difference evaluation
nC=$((nDV + nDV * nOF)) # Number of cores needed. Make sure to request an appropriate number of nodes = N / 36


module purge
module load comp-intel intel-mpi mkl
module unload gcc
module load conda   



source activate weis-env4

# conda list

# module list

# which python3.8

# echo $nC

# mpiexec -v -np $nC --mca opal_cuda_support 1 python /home/dzalkind/Tools/WEIS-4/examples/02_control_opt/iea15mw_servoopt/run_optimization.py
mpirun -v -np $nC python3.8 /home/dzalkind/Tools/WEIS-4/examples/02_control_opt/runOptimization.py
