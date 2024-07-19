#!/bin/bash
#SBATCH --ntasks        50
#SBATCH --cpus-per-task 1
#SBATCH --time          01:00:00
#SBATCH --mem-per-cpu   1G
#SBATCH --partition     acomputeq
#SBATCH --job-name      mpi
#SBATCH --output        mpi-%J.out
#SBATCH --error         mpi-%J.err

source ~/.bashrc
source activate weis-env
mpirun python weis_driver_level1_doe.py

