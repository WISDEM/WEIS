#!/bin/bash
#SBATCH --account=allocation_name
#SBATCH --time=01:00:00
#SBATCH --job-name=Design1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --mail-user user@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=Case1.%j.out
####SBATCH --qos=high
####SBATCH --partition=debug

module purge
module load comp-intel intel-mpi mkl
module unload gcc

source activate weis-env

mpirun -np 12 python weis_driver.py
