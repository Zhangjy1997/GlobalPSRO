#!/bin/sh

env="GOOFSPIEL"
scenario="SPIEL"
MSS_name="nash"  # valid MSS_name: nash, alpharank, uniform, PRD
exp="neupl_${MSS_name}"

num_env_steps=2400000
approx_pe_steps=2400000
population_size=67

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_goof_NeuPL.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size ${population_size} \
--MSS_name ${MSS_name} --calc_exp_interval 2 --approx_PE_steps ${approx_pe_steps} \
--use_policy_freeze --use_best_model_history
