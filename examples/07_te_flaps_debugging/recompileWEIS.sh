#!/bin/bash

# >>> conda initialize >>>
#conda init --all
# <<< conda initialize <<<

# get current location
currentLocation=$(pwd)
# get current environment
environmentReturn=$(conda env list)

# parse environmentReturn and return the current environment
condaEnv=$(python ~/Documents/Python/ScriptHelpers/condaEnvListParse.py "$environmentReturn")

# if the current conda environment is not weis-env, print error and exit
if [ "$condaEnv" != "weis-env" ] ; then
  echo "Error: conda environment is incorrect, please call 'conda activate weis-env' and retry"
  exit 1 ;
fi

# Otherwise, cd to the base repo and call the build script

# go to the weis repo
cd ~/gitRepos/WEIS

# run the WEIS build
python setup.py develop

# return to the original location when the bash script was called
cd $currentLocation
