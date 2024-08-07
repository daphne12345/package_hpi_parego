#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # put your install commands here (remove lines you don't need):
  	pip install -e .
  	pip install carps[smac,hpob,pymoo,yahpo]
  	pip install --upgrade numpy==1.26.4
  	pip install git+https://github.com/automl/HPOBench.git@fix/numpy_deprecation
  	pip install yahpo-gym
  	pip install configspace==0.7.1

  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi