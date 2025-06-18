#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-user athulsun@colostate.edu
#SBATCH --partition=sys_all
#SBATCH --nodes=1
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --ntasks=26
#SBATCH --error=job_log.%j.out

mpirun -np 26 python weis_driver_dfsm_fowt.py
                                                                                                                                                                                                                                         
                               