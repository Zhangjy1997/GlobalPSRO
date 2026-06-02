#!/bin/sh
# exp param
env="GOOFSPIEL"
scenario="SPIEL"
MSS_name="nash"  # valid MSS_name: nash, alpharank, uniform, PRD
exp="psro_${MSS_name}"

# game param

# train param
num_env_steps=2400000
approx_pe_steps=2400000

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_goof_Standard_PSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 67 --MSS_name ${MSS_name} --calc_exp_interval 2 \
--approx_PE_steps ${approx_pe_steps}
