#!/bin/sh
# exp param
env="GOOFSPIEL"
scenario="SPIEL"
exp="APSRO"

# game param

# train param
num_env_steps=2400000

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_goof_APSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 67 --calc_exp_interval 2
