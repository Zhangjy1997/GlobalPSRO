import copy
import os

import numpy as np
import torch

from onpolicy.algorithms.PSRO.utils.checkpoint import restore_eval_policy
from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy



class Standard_PSRO_trainer:
    """Implementation of standard PSRO, with support for diversity-driven PSD-PSRO."""

    def __init__(
        self,
        args,
        anchor_policies,
        shared_policies,
        eval_policies,
        runners,
        evaluator,
        meta_solver,
        role_names,
        save_dir,
        device=torch.device("cpu"),
    ):
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
        self.eval_policies = eval_policies
        self.runners = runners
        self.runner = runners[0] if isinstance(runners, (list, tuple)) else runners
        self.eval = evaluator
        self.meta_solver = meta_solver
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

        self.upper_std = getattr(args, "upper_std", 0.03)
        self.use_psd_psro = getattr(args, "use_psd_psro", False)

        self.payoff_mat = np.zeros((1, 1))
        self.effect_population_size = 1
        self.probs_now = np.ones(1)
        self.terminal_state = False


    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=float).copy()
        probs = np.clip(probs, 0.0, None)
        prob_sum = probs.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            return np.ones_like(probs) / len(probs)
        return probs / prob_sum

    def _current_meta_probs(self):
        meta_probs, _ = self.meta_solver(self.payoff_mat)
        return self._normalize_probs(np.asarray(meta_probs[0], dtype=float))


    def _set_runner_policy_context(self, runner, probs):
        if hasattr(runner, "trainer") and hasattr(runner.trainer, "policy"):
            runner.trainer.policy.reset_policy()
        if self.use_psd_psro and hasattr(runner, "set_oppo_policies"):
            runner.set_oppo_policies(self.eval_policies[:self.effect_population_size])

        mixed_policy = mixing_policy(
            self.n_threads,
            self.eval_policies[:self.effect_population_size],
            probs,
            self.device,
        )
        runner.num_env_steps = self.num_env_steps
        runner.all_args.global_steps = self.g_step
        runner.envs.world.oppo_policy = copy.deepcopy(mixed_policy)

        runner.set_policy_inx(0)

    def _update_payoff_matrix(self):
        self.eval.update_policy(self.eval_policies[:self.effect_population_size])
        mask_mat = np.zeros((self.effect_population_size, self.effect_population_size), dtype=bool)
        mask_mat[-1, :] = True
        mask_mat[-1, -1] = False
        delta_mat = self.upper_std * np.ones((self.effect_population_size, self.effect_population_size))

        print("Evaluator start!")
        try:
            payoff_mat_update = self.eval.get_win_prob_with_mask(
                self.n_threads,
                self.n_eval_eps,
                mask=mask_mat,
                delta_mat=delta_mat,
            )
        except TypeError:
            payoff_mat_update = self.eval.get_win_prob_with_mask(
                self.n_threads,
                self.n_eval_eps,
                mask=mask_mat,
            )

        payoff_mat_update = payoff_mat_update - payoff_mat_update.T
        expanded_payoff_mat = np.pad(self.payoff_mat, ((0, 1), (0, 1)), "constant", constant_values=0)
        self.payoff_mat = expanded_payoff_mat + payoff_mat_update
        print("payoff_mat = ", self.payoff_mat)

    def step(self):

        if self.effect_population_size >= self.policy_num:
            self.terminal_state = True
            return copy.deepcopy(self.eval_policies[:self.effect_population_size]), self.probs_now.copy(), []

        target_probs = self._current_meta_probs()
        print("standard PSRO meta probs = ", target_probs)

        self._set_runner_policy_context(self.runner, target_probs)

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

        self.effect_population_size += 1

        self._update_payoff_matrix()
        self.probs_now = self._current_meta_probs()
        print("standard PSRO updated meta probs = ", self.probs_now)

        return copy.deepcopy(self.eval_policies[:self.effect_population_size]), self.probs_now.copy(), [train_logs]
