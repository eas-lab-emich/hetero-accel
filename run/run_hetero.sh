#!/bin/bash

export HETERO_CACHE_DIR=/home/welby/workspace/hetero-accel
#export HETERO_MQ_HOST=18.117.167.181
#export HETERO_MQ_PASS=mypassword
#export HETERO_MQ_USER=worker
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

../.vence/bin/python ../main.py --yaml-cfg-file args_cfg.yaml --workload-cfg-file workloads.yaml
