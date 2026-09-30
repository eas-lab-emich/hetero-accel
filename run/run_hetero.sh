#!/bin/bash

export HETERO_CACHE_DIR=/home/welby/workspace/hetero-accel
export HETERO_MQ_HOST=18.117.167.181
export HETERO_MQ_PASS=mypassword
export HETERO_MQ_USER=worker
export TL_THREAD_COUNT=1
export PYTHONPATH=/home/welby/workspace/hetero-accel/intel_distiller
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/welby/workspace/hetero-accel/.vence/bin/python /home/welby/workspace/hetero-accel/main.py --yaml-cfg-file /home/welby/workspace/hetero-accel/run/args_cfg.yaml --workload-cfg-file /home/welby/workspace/hetero-accel/run/workloads.yaml 
