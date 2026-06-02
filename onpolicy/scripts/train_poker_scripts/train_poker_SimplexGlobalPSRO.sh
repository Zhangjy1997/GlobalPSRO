#!/bin/sh
# exp param
env="LEDUC_POKER"
scenario="POKER"
MSS_name="nash"  # valid MSS_name: nash, alpharank, uniform, PRD
exp="simplex_global_${MSS_name}"

# game param

# train param
num_env_steps=800000
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_SimplexGlobalPSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 51 --PE_interval 1 --max_workers 16 --upper_epsilon 0.00 \
--MSS_name ${MSS_name}
