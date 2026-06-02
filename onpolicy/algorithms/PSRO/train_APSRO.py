import copy
import os

import numpy as np
import torch

from onpolicy.algorithms.PSRO.utils.checkpoint import restore_eval_policy
from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy


class Anytime_PSRO_trainer:
    """Implementation of Anytime PSRO."""

    def __init__(
        self,
        args,
        anchor_policies,
        shared_policies,
        eval_policies,
        runner,
        role_names,
        save_dir,
        device=torch.device("cpu"),
    ):
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
        self.eval_policies = eval_policies
        self.runner = runner
        self.role_names = role_names
        self.save_dir = str(save_dir)
        self.device = device

        os.makedirs(self.save_dir, exist_ok=True)

        self.n_threads = self.args.n_rollout_threads
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.num_env_steps = self.args.num_env_steps
        self.policy_num = self.args.population_size
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]
        self.eval_policies[0] = self.policies_anchor[0]
        self.payoff_mat = np.zeros((1, 1))
        self.effect_population_size = 1
        self.probs_now = np.ones(1)
        self.terminal_state = False

    def step(self):
        if self.effect_population_size >= self.policy_num:
            self.terminal_state = True
            return copy.deepcopy(self.eval_policies[: self.effect_population_size]), self.probs_now.copy(), []

        probs_ini = np.ones(self.effect_population_size, dtype=float)
        probs_ini /= np.sum(probs_ini)
        mixed_policy = mixing_policy(
            self.n_threads,
            self.eval_policies[: self.effect_population_size],
            probs_ini.copy(),
            self.device,
        )
        self.runner.num_env_steps = self.num_env_steps
        self.runner.all_args.global_steps = self.g_step
        self.runner.envs.world.oppo_policy = mixed_policy
        self.runner.set_policy_n_prob(self.effect_population_size, probs_ini)

        train_logs = self.runner.run()
        if train_logs is None:
            train_logs = dict()

        policy_head = "policy_" + str(self.effect_population_size)
        self.runner.save_as_filename(policy_head)
        restore_eval_policy(
            self.eval_policies[self.effect_population_size],
            self.save_dir,
            head_str=policy_head,
        )

        self.g_step += self.num_env_steps
        self.probs_now = self.runner.probs.copy()
        self.effect_population_size += 1

        return copy.deepcopy(self.eval_policies[: self.effect_population_size - 1]), self.probs_now.copy(), [train_logs]
