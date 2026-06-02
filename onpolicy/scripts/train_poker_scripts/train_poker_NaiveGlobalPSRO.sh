#!/bin/sh

env="LEDUC_POKER"
scenario="POKER"
exp="NaiveGlobal"
num_env_steps=800000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train/train_poker_NaiveGlobalPSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 51 --max_workers 16 --upper_epsilon 0.00
