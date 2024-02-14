
CASE_NAME=SLSQP

timeout 1h python -u driver_weis_raft_opt.py > >(tee stdout.log) 2> >(tee stderr.log >&2)

mv stdout.log stdout_${CASE_NAME}.log
mv stderr.log stderr_${CASE_NAME}.log
rm -rf 32_DesignRound1_${CASE_NAME}
mv 32_DesignRound1 32_DesignRound1_${CASE_NAME}
