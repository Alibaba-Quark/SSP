#!/bin/bash

set -x
ulimit -n 65535

# ============================================================================
# Configuration: Please modify these paths according to your setup
# ============================================================================

# Model checkpoint paths
export SOURCE_ACTOR_CHECKPOINT_DIR=/path/to/actor/checkpoint
export SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME=checkpoint_name
export SOURCE_REWARD_CHECKPOINT_DIR=/path/to/reward/checkpoint  # Optional if reward model disabled
export SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME=checkpoint_name
export SOURCE_CRITIC_CHECKPOINT_DIR=/path/to/critic/checkpoint
export SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME=checkpoint_name

# Data paths (space-separated for multiple files)
export DATA_PATH="/path/to/train_data1 /path/to/train_data2"
export TEST_DATA_PATH="/path/to/test_data1 /path/to/test_data2"

# Output paths
export OUTPUT_DIR=/path/to/output
export SAVE_CHECKPOINT_DIR=/path/to/save/checkpoints
export TENSORBOARD_LOG_DIR=/path/to/tensorboard/logs

# Distributed training settings
export NNODES=${NNODES:-1}  # Number of nodes
export RANK=${RANK:-0}      # Current node rank (0 for master)
export MASTER_ADDR=${MASTER_ADDR:-localhost}  # Master node address

# Judge and Search service settings
export QUARK_BASE_URL=http://judge_host:5000/v1  # LLM-as-a-Judge service URL
export QUARK_MODEL=judge_model_name              # Judge model name
export QUARK_SEARCH_CHAT_TEMPLATE=default        # Chat template: default, qwen2p5, R-Search, llama3p1
export SEARCH_IP=search_service_ip               # Retrieval service IP address

# ============================================================================
# Runtime Configuration
# ============================================================================

export LOG_LEVEL=DEBUG
export SELF_PLAY_DEBUG=True

echo "Log level set to: $LOG_LEVEL"

export NCCL_TIMEOUT=72000000  
export TORCH_DISTRIBUTED_TIMEOUT=72000

export NCCL_WORK_FIFO_DEPTH=4194304
export LANG=C.UTF-8
export LANGUAGE=C.UTF-8

pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip3 config set install.trusted-host mirrors.aliyun.com

pip3 install tensorboardX qwen_vl_utils
pip3 install transformers==4.55.0
pip3 install sglang==0.4.6.post5 sgl_kernel==0.1.5 cuda-python cuda-bindings torch_memory_saver torchao
pip3 install "pyarrow>=19.0.1"
pip3 install "optree>=0.13.0"
pip3 install torchdata math_verify==0.8.0

export WORKSPACE=/root/code/my_workspace

mkdir -p $WORKSPACE

if [ ! -d "${WORKSPACE}/verl" ]; then
    cp /root/code/verl.zip $WORKSPACE/verl.zip
    unzip -o $WORKSPACE/verl.zip -d $WORKSPACE
fi

if [ ! -d "${WORKSPACE}/quarl" ]; then
    cp /root/code/quarl.zip $WORKSPACE/quarl.zip
    unzip -o $WORKSPACE/quarl.zip -d $WORKSPACE
fi

export PYTHONPATH=$WORKSPACE/verl
export REPO_DIR=$WORKSPACE/quarl
export PYTHONPATH=$PYTHONPATH:$REPO_DIR

export TENSORBOARD_DIR=${TENSORBOARD_LOG_DIR}

ACTOR_PATH=${SOURCE_ACTOR_CHECKPOINT_DIR}/${SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME}
REWARD_PATH=${SOURCE_REWARD_CHECKPOINT_DIR}/${SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME}
CRITIC_PATH=${SOURCE_CRITIC_CHECKPOINT_DIR}/${SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME}

read -ra TRAIN_PATHS <<< "$DATA_PATH"
read -ra TEST_PATHS <<< "$TEST_DATA_PATH"

TRAIN_FILES=()
for path in "${TRAIN_PATHS[@]}"; do
    TRAIN_FILES+=("${path}.parquet")
done

TEST_FILES=()
for path in "${TEST_PATHS[@]}"; do
    TEST_FILES+=("${path}.parquet")
done

TRAIN_FILES_STR="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
TEST_FILES_STR="[$(IFS=,; echo "${TEST_FILES[*]}")]"

if [[ "${QUARK_SEARCH_CHAT_TEMPLATE}" == "R-Search" ]]; then
    echo "Detected QUARK_SEARCH_CHAT_TEMPLATE=${QUARK_SEARCH_CHAT_TEMPLATE}, processing test files..."
    
    PROCESSED_DIR="./processed_test_files"
    mkdir -p $PROCESSED_DIR
    
    PROCESSED_TEST_FILES=()
    for path in "${TEST_PATHS[@]}"; do
        input_file="${path}.parquet"
        filename=$(basename "$path")
        output_file="${PROCESSED_DIR}/${filename}_${QUARK_SEARCH_CHAT_TEMPLATE}.parquet"
        
        echo "Processing file: $input_file -> $output_file"
        python3 $REPO_DIR/examples/sglang_multiturn/search_r1_like/reprocess_data.py \
            -i "$input_file" \
            -o "$output_file" \
            -t "${QUARK_SEARCH_CHAT_TEMPLATE}"
        
        if [ $? -eq 0 ]; then
            PROCESSED_TEST_FILES+=("$output_file")
            echo "Successfully processed: $output_file"
        else
            echo "Processing failed: $input_file, using original file"
            PROCESSED_TEST_FILES+=("$input_file")
        fi
    done
    
    TEST_FILES_STR="[$(IFS=,; echo "${PROCESSED_TEST_FILES[*]}")]"
    echo "Updated TEST_FILES_STR to processed files: $TEST_FILES_STR"
    LANG=${QUARK_SEARCH_CHAT_TEMPLATE}
    echo "Set LANG=${LANG}"
elif [[ "${QUARK_SEARCH_CHAT_TEMPLATE}" == "qwen2p5" ]] || [[ "${QUARK_SEARCH_CHAT_TEMPLATE}" == "llama3p1" ]]; then
    echo "QUARK_SEARCH_CHAT_TEMPLATE=${QUARK_SEARCH_CHAT_TEMPLATE}, using original test files"
    LANG=en
    echo "Set LANG=${LANG}"
else
    echo "QUARK_SEARCH_CHAT_TEMPLATE=${QUARK_SEARCH_CHAT_TEMPLATE}, using original test files"
    LANG=zh
    echo "Set LANG=${LANG}"
fi

export NNODES=${NNODES}
export NODE_RANK=${RANK}

TOOL_CONFIG=${REPO_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml
MAX_TURNS=10

SEARCH_PORT="8000"  

sed -i "/wiki:/s|http://[0-9.]\+:${SEARCH_PORT}/retrieve|http://${SEARCH_IP}:${SEARCH_PORT}/retrieve|" "$TOOL_CONFIG"

echo "Updated wiki search service address to: http://${SEARCH_IP}:${SEARCH_PORT}/retrieve"

TRAIN_METHOD=grpo

BATCH_SIZE=256
MINI_BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
PPO_MAX_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
PPO_MICRO_BATCH_PER_GPU=1
LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU=1
REWARD_MIRCO_BATCH_SIZE_PER_GPU=1
CRITIC_MIRCO_BATCH_SIZE_PER_GPU=1
ROLLOUT_TP=4
ACTOR_SP=1
REWARD_SP=1
CRITIC_SP=1
ACTOR_FSDP_SIZE=-1
FSDP_TYPE=fsdp  # fsdp/fsdp2
REWARD_MODEL_ENABLE=False
CLIP_RATIO_HIGH=0.285

TOTAL_EPOCHS=10
VAL_N=1
VAL_TEMPERATURE=0.0
VAL_ONLY=False

if [[ "${TRAIN_METHOD}" == "grpo" ]]; then
    ROLLOUT_N=5
    ADV_ESTIMATOR=grpo
    ACTOR_USE_KL_LOSS=True
    USE_KL_IN_REWARD=False
else
    ROLLOUT_N=1
    ADV_ESTIMATOR=gae
    ACTOR_USE_KL_LOSS=False
    USE_KL_IN_REWARD=True
fi

TASK_TYPE=quark_deep_search

if [[ "${TASK_TYPE}" == "quark_deep_search" ]]; then
    RM_TYPE=default
    DATASET_CLASS=null
    RM_MANAGER=quark
else
    RM_TYPE=default
    DATASET_CLASS=null
    RM_MANAGER=naive
fi

if [[ "${RM_MANAGER}" == "quark" ]]; then
    USE_QUARK_SCORE=True
    CUSTOM_RM_ARGS="reward_model.reward_manager=${RM_MANAGER}"

    if [[ "${USE_QUARK_SCORE}" == "True" ]]; then
	CUSTOM_RM_ARGS="${CUSTOM_RM_ARGS} \
            +custom_reward_functions.quark_score.labels=['unknown'] \
            +custom_reward_functions.quark_score.integration=sum \
            quark.diff_val_reward_fn_config.reward_model.reward_manager=naive_with_prompt \
            quark.diff_val_reward_fn_config.custom_reward_function.path=$WORKSPACE/quarl/quarl/reward/score/search_eval_score.py \
            quark.diff_val_reward_fn_config.custom_reward_function.name=compute_score"
    fi
else
    CUSTOM_RM_ARGS="reward_model.reward_manager=${RM_MANAGER}"
fi

mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/val
mkdir -p $OUTPUT_DIR/rollout

if [ $NODE_RANK -eq 0 ]; then
    ray start --block --head --port=6379 &

    if [ ! $NNODES -eq 1 ]; then 
	sleep 120
    fi

    python3 -m quarl.main_rl \
	    quark.task_type=$TASK_TYPE \
            algorithm.adv_estimator=${ADV_ESTIMATOR} \
            data.train_files=${TRAIN_FILES_STR} \
            data.val_files=${TEST_FILES_STR} \
            data.train_batch_size=${BATCH_SIZE} \
            data.val_batch_size=${BATCH_SIZE} \
            data.max_prompt_length=${MAX_PROMPT_LENGTH} \
            data.max_response_length=${MAX_RESPONSE_LENGTH} \
            data.return_raw_chat=True \
            data.filter_overlong_prompts=True \
            data.truncation='left' \
            data.shuffle=True \
	    data.custom_cls.name=${DATASET_CLASS} \
            actor_rollout_ref.model.path=${ACTOR_PATH} \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
            actor_rollout_ref.model.use_remove_padding=True \
	    actor_rollout_ref.actor.strategy=${FSDP_TYPE} \
            actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
            actor_rollout_ref.actor.use_dynamic_bsz=False \
            actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SP} \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.actor.fsdp_config.offload_policy=True \
            actor_rollout_ref.actor.fsdp_config.fsdp_size=$ACTOR_FSDP_SIZE \
            actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
            actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
            actor_rollout_ref.rollout.name=sglang_async \
	    actor_rollout_ref.rollout.multi_turn.format=quark \
            actor_rollout_ref.rollout.multi_turn.enable=True \
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
	    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
	    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
            actor_rollout_ref.hybrid_engine=True \
	    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.n=${ROLLOUT_N} \
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
	    actor_rollout_ref.ref.strategy=${FSDP_TYPE} \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
            algorithm.kl_ctrl.kl_coef=0.001 \
            algorithm.kl_penalty='low_var_kl' \
            algorithm.gamma=1.0 \
            critic.optim.lr=1e-5 \
            critic.ppo_micro_batch_size_per_gpu=${CRITIC_MIRCO_BATCH_SIZE_PER_GPU} \
            critic.strategy=${FSDP_TYPE} \
            critic.model.use_remove_padding=True \
            critic.model.path=${CRITIC_PATH} \
            critic.model.enable_gradient_checkpointing=True \
            critic.model.fsdp_config.param_offload=True \
            critic.model.fsdp_config.optimizer_offload=True \
	    critic.ulysses_sequence_parallel_size=${CRITIC_SP} \
            reward_model.enable=${REWARD_MODEL_ENABLE} \
	    reward_model.strategy=${FSDP_TYPE} \
            reward_model.model.type=${RM_TYPE} \
            reward_model.model.path=${REWARD_PATH} \
            reward_model.model.use_remove_padding=True \
            reward_model.max_length=$PPO_MAX_TOKEN_LEN_PER_GPU \
            reward_model.micro_batch_size_per_gpu=${REWARD_MIRCO_BATCH_SIZE_PER_GPU} \
            reward_model.model.fsdp_config.param_offload=True \
            reward_model.ulysses_sequence_parallel_size=${REWARD_SP} \
	    ${CUSTOM_RM_ARGS} \
            trainer.critic_warmup=0 \
            trainer.logger=['console','tensorboard'] \
            trainer.project_name=quark_ssp_${TRAIN_METHOD} \
            trainer.experiment_name=ssp_${TRAIN_METHOD} \
            trainer.val_before_train=True \
            trainer.val_only=${VAL_ONLY} \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=${NNODES} \
            trainer.save_freq=5 \
            trainer.test_freq=5 \
            trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
            trainer.rollout_data_dir=$OUTPUT_DIR/rollout \
            trainer.validation_data_dir=$OUTPUT_DIR/val \
            trainer.total_epochs=${TOTAL_EPOCHS} \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
            actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            +actor_rollout_ref.rollout.val_kwargs.frequency_penalty=0 \
            +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1 \
            +actor_rollout_ref.rollout.repetition_penalty=1 \
            self_play.enable=True \
            self_play.lang=${LANG} \
            self_play.save_freq=5 \
            self_play.use_rag_filter=True \
            self_play.noisy_RAG_materials=4 \
            self_play.proposer.enable=True \
            self_play.proposer.warm_up_steps=-1 \
            self_play.proposer.format_penalty=0 \
            self_play.proposer.n=1 \
            self_play.proposer.reward_type=1-acc \
            self_play.proposer.adv_estimator=grpo \
            self_play.proposer.right=1.0 \
            self_play.proposer.left=0.0 \
            self_play.extraction_failure.strategy=reuse \
            self_play.solver.enable=True \
            self_play.dynamic_sampling.enable=False \
            self_play.reward_dynamic_sampling.enable=False \
            self_play.use_search_terms_filter=False \
            self_play.extraction_failure.reuse_success_rate_threshold=1.0 \
            self_play.combine_update=False \
            self_play.mini_epochs=1 \
            self_play.extraction_failure.pool_clear_interval=10 \
            self_play.answer_pattern=question \
            self_play.validate_config=True \
            self_play.extraction_failure.keep_ratio=0
else
    ray start --block --address=${MASTER_ADDR}:6379
    sleep 120
fi
