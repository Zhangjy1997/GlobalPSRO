#!/bin/sh
# exp param
env="LEDUC_POKER"
scenario="POKER"
exp="APSRO"

# game param

# train param
num_env_steps=800000

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_APSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 51 --calc_exp_interval 1 --PE_interval 1
