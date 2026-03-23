#
# Copyright 2026 Yuanfu Wang
# Modified by Yuanfu Wang (Shanghai Artificial Intelligence)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
export MODEL_PATH=/path/to/model
export MLP_WORKER_GPU=8
export MLP_WORKER_NUM=1

DATASET_NAME=tulu-3-sft-mixture-v1
REWARD_AGG_MODE=weighted_sum_log_prob
PROJECT_NAME='verl-nrt'
EXP_NAME=${DATASET_NAME}-${REWARD_AGG_MODE}-K1
export WANDB_DIR=outputs/wandb/${EXP_NAME}/
mkdir -p ${WANDB_DIR}

python3 -m recipe.nrt.main_nrt \
    algorithm.adv_estimator=grpo \
    data.train_files=recipe/nrt/data/processed/${DATASET_NAME}/train.parquet \
    data.val_files=recipe/nrt/data/processed/${DATASET_NAME}/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_think_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=512 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_nrt_loss=true \
    actor_rollout_ref.actor.trace_loss_coef=1.0 \
    actor_rollout_ref.actor.response_prefix_loss_coef=0.3 \
    actor_rollout_ref.actor.token_loss_coef=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    reward_model.reward_manager=naive \
    custom_reward_function.path=recipe/nrt/reward_score.py \
    custom_reward_function.name=compute_score \
    algorithm.response_format=default \
    algorithm.reward_agg_mode=${REWARD_AGG_MODE} \
    algorithm.center_reward_by=mean \
    algorithm.scale_reward_by=std \
    algorithm.use_fixed_token_reward=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=false \
    trainer.critic_warmup=0 \
    trainer.log_val_generations=30 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${MLP_WORKER_GPU} \
    trainer.nnodes=${MLP_WORKER_NUM} \
    trainer.rollout_data_dir=checkpoints/${PROJECT_NAME}/${EXP_NAME} \
    trainer.save_freq=100 \
    trainer.test_freq=400 \
    trainer.total_training_steps=810 \
    trainer.total_epochs=1 "$@"
    