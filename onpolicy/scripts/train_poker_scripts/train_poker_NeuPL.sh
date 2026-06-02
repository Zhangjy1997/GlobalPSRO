#!/bin/sh

env="LEDUC_POKER"
scenario="POKER"
MSS_name="nash"  # valid MSS_name: nash, alpharank, uniform, PRD
exp="neupl_${MSS_name}"

num_env_steps=800000
population_size=51

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_NeuPL.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size ${population_size} \
--MSS_name ${MSS_name} --PE_interval 2 \
--use_policy_freeze --use_best_model_history
