# ------ Sync DLC Data from eagle runs locally -------

# --- 5MW LAND LEGACY ---


# SQL Files
# outdir='/scratch/dzalkind/WEIS-4/optimizations/outputs_opt_ball_step10/'
# indir='/Users/dzalkind/Tools/WEIS-4/optimizations/outputs_opt_ball_step10/'
# mkdir -p $indir;
# rsync -aP --no-g --include="*/" --include="log_opt.sql*" --exclude="*" dzalkind@eagle.hpc.nrel.gov:$outdir $indir


# FAST Outs
outdir='/scratch/dzalkind/WEIS-4/optimizations/outputs_opt_pitch_5en3/'
indir='/Users/dzalkind/Tools/WEIS-4/optimizations/outputs_opt_pitch_5en3/'
mkdir -p $indir;
rsync -aP --no-g --include="*/" --include="NREL5MW_OC3_spar_*_IEC_0.out" --exclude="*" dzalkind@eagle.hpc.nrel.gov:$outdir $indir

# rsync dzalkind@eagle.hpc.nrel.gov:$outdir/case_matrix.yaml $indir

# # --- 5MW LAND ROSCO ---
# outdir2='/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar'
# indir2='../BatchOutputs/5MW_Land/5MW_Land_ROSCO/'
# mkdir -p $indir2;
# rsync nabbas@eagle.hpc.nrel.gov:$outdir2*.outb $indir2
# rsync nabbas@eagle.hpc.nrel.gov:$outdir2/case_matrix.yaml $indir2
