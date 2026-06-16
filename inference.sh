set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1,2,3
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

CHECKPOINT_PATH='/path/to/checkpoint'
val_batch_size=1050
OUTPUT_DIR="results"

python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size 336 \
    --val_data_size 1050

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_jsonl_path=data/scaffold/all/MindCube_train.jsonl \
    data.val_jsonl_path=data/scaffold/all/MindCube_tinybench.jsonl \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4000 \
    data.max_response_length=2000 \
    data.max_retrieve=4 \
    data.filter_overlong_prompts=False \
    data.retrieve_strategy="adaptive" \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$CHECKPOINT_PATH \
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
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.max_model_len=8192\
    actor_rollout_ref.rollout.limit_images=1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=wondermind \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_wondermind' \
    trainer.experiment_name='grpo_qwen2.5_vl_3b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=500 \
    trainer.mode=inference \
    trainer.default_inference_output_dir=$OUTPUT_DIR \
    trainer.val_before_train=True $@

python3 evaluate.py "$OUTPUT_DIR"
