#!/bin/bash
## Modify walltime and account at minimum
#SBATCH --time=00:60:00
#SBATCH --account=weis

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=debug

module purge
module load conda


source activate /projects/weis/mchetan/weis-viz-test/env/weis-viz-demo
port=8050

echo "run the following command on your machine"
echo ""
echo "ssh -L $port:$HOSTNAME:$port $SLURM_SUBMIT_HOST.hpc.nrel.gov"

python ../app/mainApp.py --yaml /projects/weis/mchetan/weis-viz-test/example-opt/outputs/22-iea/vizInputFile.yaml --debug True
