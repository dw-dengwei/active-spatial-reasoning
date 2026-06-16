set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

num_cpus_per_env_worker=0.02 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=10000
train_batch_size=128
val_data_size=1050
val_batch_size=1050
group_size=8
model_path='/path/to/checkpoint/'

exp_name="exp_1"

rollout_data_dir="rollout_data/$exp_name"
validation_data_dir="validation_data/$exp_name"

project_name='active-spatial-reasoning'
export TENSORBOARD_DIR=tensorboard_log/$project_name/$exp_name

# # We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_jsonl_path=data/scaffold/all/MindCube_train.jsonl \
    data.val_jsonl_path=data/scaffold/all/MindCube_tinybench.jsonl \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.train_data_size=$train_data_size \
    data.val_data_size=$val_data_size \
    data.use_sac=True \
    data.use_cogmap_reward=True \
    data.use_retrieve_reward=True \
    data.max_retrieve=4 \
    data.max_prompt_length=4000 \
    data.max_response_length=700 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.dtype=bfloat16 \
    actor_rollout_ref.ref.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=1 \
    actor_rollout_ref.rollout.max_model_len=8192\
    actor_rollout_ref.rollout.max_num_seqs=1024\
    actor_rollout_ref.rollout.limit_images=1\
    algorithm.use_kl_in_reward=False \
    env.env_name=wondermind \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['swanlab'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.rollout_data_dir=$rollout_data_dir  \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.log_val_generations=0 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches=2 \
    trainer.val_before_train=False $@