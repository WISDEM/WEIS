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

output=$(python weis_driver.py --preMPIflag=True --maxCores=104)

# Extract the values from the output (adjust based on actual format)
nC=$(echo "$output" | grep 'nC=' | awk -F'=' '{print $2}')
n_FD=$(echo "$output" | grep 'n_FD=' | awk -F'=' '{print $2}')
n_OFp=$(echo "$output" | grep 'n_OFp=' | awk -F'=' '{print $2}')

mpirun -np $nC python runWEIS.py --n_FD=$n_FD --n_OF_parallel=$n_OFp
