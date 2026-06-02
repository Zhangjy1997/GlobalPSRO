#!/bin/sh

env="GOOFSPIEL"
scenario="SPIEL"
exp="NaiveGlobal"

num_env_steps=2400000
population_size=67
approx_pe_steps=2400000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train/train_goof_NaiveGlobalPSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size ${population_size} --calc_exp_interval 2 \
--approx_PE_steps ${approx_pe_steps} --max_workers 16 --upper_epsilon 0.00
