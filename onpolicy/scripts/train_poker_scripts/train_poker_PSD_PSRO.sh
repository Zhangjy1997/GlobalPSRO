#!/bin/sh
# exp param
env="LEDUC_POKER"
scenario="POKER"
MSS_name="nash"  # valid MSS_name: nash, alpharank, uniform, PRD
exp="psd_psro_${MSS_name}"

# game param

# train param
num_env_steps=800000

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_Standard_PSRO.py \
--env_name ${env} --scenario_name ${scenario} --experiment_name ${exp} --seed 1 \
--num_env_steps ${num_env_steps} --population_size 51 --MSS_name ${MSS_name} --use_psd_psro --kl_div_coef 0.1 \
--calc_exp_interval 1 --PE_interval 1
