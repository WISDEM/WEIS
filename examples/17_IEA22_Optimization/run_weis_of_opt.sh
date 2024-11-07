#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=iea22_ptfm
#SBATCH --mail-user dzalkind@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
##SBATCH --partition=debug
######SBATCH --qos=high
######SBATCH --mem=1000GB      # RAM in MB
#SBATCH --output=logs/job_log.%j.out  # %j will be replaced with the job ID

source /home/dzalkind/.bashrc
conda activate weis-new2


nC=85


# mpirun -np 73 python driver_weis.py
mpiexec -n $nC python driver_weis_openfast_opt.py

# python driver_weis.py

# python create_conv_plots.py

 
