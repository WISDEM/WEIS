#!/bin/bash
#SBATCH --ntasks        384
#SBATCH --cpus-per-task 1
#SBATCH --time          02:00:00
#SBATCH --mem-per-cpu   1G
#SBATCH --partition     acomputeq
#SBATCH --job-name      mpi
#SBATCH --output        mpi-%J.out
#SBATCH --error         mpi-%J.err
#SBATCH --mail-user     ${USER}@memphis.edu
#SBATCH --mail-type     ALL

# Alternatives: openmpi/4.0.5/gcc.8.2.0, impi/2022.2
#module load openmpi/3.1.1/gcc.8.2.0
source ~/.bashrc
source activate weis-env

# You might want this line if it complains about temp files
#export TMPDIR=/tmp/

# Run the test program as seperate tasks
mpirun python weis_driver_level1_doe.py

# Exit
