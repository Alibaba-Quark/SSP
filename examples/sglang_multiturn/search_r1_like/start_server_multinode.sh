#!/bin/bash
export NNODES=${NNODES}
export NODE_RANK=${RANK}
MODEL_PATH= #Your Judge Model Path
echo "Starting with NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, MODEL_PATH=${MODEL_PATH}"



if [ $NODE_RANK -eq 0 ]; then

    ray start --head --port=6379 --num-gpus=8 &
    sleep 30
    

    CURRENT_NODES=$(ray status | grep "node_" | wc -l)
    while [ "${CURRENT_NODES}" -lt "${NNODES}" ]; do
        echo "Waiting for nodes... Current: ${CURRENT_NODES}, Expected: ${NNODES}"
        sleep 10
        CURRENT_NODES=$(ray status | grep "node_" | wc -l)
    done
    
    nohup python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 5000 \
    --model $MODEL_PATH \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --gpu-memory-utilization 0.8 > vllm_server.log 2>&1 &
else

    ray start --address=${MASTER_ADDR}:6379 --num-gpus=8
    sleep 120
fi