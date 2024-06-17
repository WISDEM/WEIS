
CASE_NAME=SLSQP

timeout 90m python -u driver_weis_raft_opt.py > >(tee stdout.log) 2> >(tee stderr.log >&2)

mv stdout.log stdout_${CASE_NAME}.log
mv stderr.log stderr_${CASE_NAME}.log
rm -rf 17_IEA22_Opt_Result_${CASE_NAME}
mv 17_IEA22_Opt_Result 17_IEA22_Opt_Result_${CASE_NAME}
