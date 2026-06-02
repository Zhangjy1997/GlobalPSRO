import copy

import numpy as np
import torch
from open_spiel.python.algorithms import expected_game_score

from onpolicy.algorithms.PSRO.utils.checkpoint import restore_eval_policy
from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy


class NeuPL_Trainer:
    """NeuPL implementation.

    In the original NeuPL paper, policies are not frozen during population
    growth. In practice, keeping the whole population fully active made the
    population hard to stabilize and led to poor performance, so this trainer
    supports gradually freezing lower-level policies.
    """

    def __init__(
        self,
        args,
        anchor_policies,
        shared_policies,
        eval_policies,
        runner,
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
        self.runner = runner
        self.eval = evaluator
        self.eval_envs = copy.deepcopy(self.eval.envs)
        self.meta_solver = meta_solver
        self.role_names = role_names
        self.save_dir = save_dir
        self.device = device

        self.n_threads = self.args.n_rollout_threads
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.num_env_steps = self.args.num_env_steps
        self.policy_num = self.args.population_size
        self.condition_size = getattr(self.args, "latent_size", self.policy_num)
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]

        self.eval_policies[0] = self.policies_anchor[0]
        self.upper_std = args.upper_std
        self.payoff_mat = np.zeros((1, 1))
        self.probs_now = np.ones(1)
        self.active_number = 1
        self.lower_i = 0
        self.frozen_payoff_mat = self.payoff_mat.copy()
        self.last_reward = -np.inf
        self.max_last_reward = -np.inf
        self.best_model = None
        self.best_payoff_line = None
        self.use_policy_freeze = getattr(args, "use_policy_freeze", False)
        self.use_best_model_history = getattr(args, "use_best_model_history", False)
        self.upper_epsilon = getattr(args, "upper_epsilon", 0.0)
        self.freeze_no_improve_k = 2
        self.no_improve_count = 0
        if self.use_policy_freeze:
            print("Use reward-threshold policy freezing.")
        if self.use_best_model_history:
            print("Use best model history for frozen active sigma policies.")


        self.use_table_policy = getattr(args, "use_table_policy", False)
        if self.use_table_policy:
            print("Use table policy!")


    def _sigma_full(self, sigma):
        sigma = np.asarray(sigma)
        sigma_full = np.zeros(self.condition_size, dtype=sigma.dtype)
        sigma_full[: len(sigma)] = sigma
        return sigma_full

    def _build_sigma_matrix(self, active_number, policy_size):
        """Build one sigma row for each active policy from prefix meta-solves."""
        probs_mat = np.zeros((active_number, policy_size))
        for active_idx in range(active_number):
            prefix_size = self.lower_i + 1 + active_idx
            target_prob, _ = self.meta_solver(self.payoff_mat[:prefix_size, :prefix_size])
            probs_mat[active_idx, : len(target_prob[0])] = target_prob[0]
        return probs_mat

    def _build_payoff_eval_mask(self, matrix_size):
        mask_mat = np.zeros((matrix_size, matrix_size), dtype=bool)
        for row in range(1, matrix_size):
            mask_mat[row, :row] = True

        if self.use_policy_freeze and self.lower_i > 0:
            frozen_size = min(self.lower_i + 1, matrix_size)
            mask_mat[:frozen_size, :frozen_size] = False

        return mask_mat

    def _restore_frozen_payoffs(self, payoff_mat):
        if self.use_policy_freeze and self.lower_i > 0:
            frozen_size = min(
                self.lower_i + 1,
                payoff_mat.shape[0],
                self.frozen_payoff_mat.shape[0],
            )
            payoff_mat[:frozen_size, :frozen_size] = self.frozen_payoff_mat[:frozen_size, :frozen_size]
        return payoff_mat

    def _refresh_eval_policies(self, probs_mat):
        for active_idx in range(probs_mat.shape[0]):
            policy_idx = self.lower_i + 1 + active_idx
            sigma_full = self._sigma_full(probs_mat[active_idx])

            if self.use_table_policy:
                sample_network = copy.deepcopy(self.policies_shared[0])
                restore_eval_policy(sample_network, self.save_dir, head_str="policy_active")
                sample_network.set_sigma(np.tile(sigma_full, (1, 1)))
                sample_network.set_fusion_false()
                self.eval_policies[policy_idx] = self.trans2tablepolicy(
                    self.standard_game,
                    range(2),
                    [sample_network],
                    np.ones(1),
                    device=self.device,
                )
            else:
                restore_eval_policy(self.eval_policies[policy_idx], self.save_dir, head_str="policy_active")
                self.eval_policies[policy_idx].set_sigma(np.tile(sigma_full, (1, 1)))
                self.eval_policies[policy_idx].set_fusion_false()

    def _evaluate_active_payoffs(self, matrix_size):
        delta_mat = self.upper_std * np.ones((matrix_size, matrix_size))
        mask_mat = self._build_payoff_eval_mask(matrix_size)
        if self.use_table_policy:
            payoff_mat = np.zeros_like(delta_mat)
            for row in range(1, matrix_size):
                for col in range(row):
                    if not mask_mat[row][col]:
                        continue
                    payoff_0 = expected_game_score.policy_value(
                        self.standard_game.new_initial_state(),
                        [self.eval_policies[row], self.eval_policies[col]],
                    )
                    payoff_1 = expected_game_score.policy_value(
                        self.standard_game.new_initial_state(),
                        [self.eval_policies[col], self.eval_policies[row]],
                    )
                    payoff_mat[row][col] = (payoff_0[0] + payoff_1[1]) / 2
            return self._restore_frozen_payoffs(payoff_mat - payoff_mat.T)

        self.eval.update_policy(self.eval_policies[:matrix_size])
        print("NeuPL active evaluator start!")
        payoff_mat = self.eval.get_win_prob_with_mask(
            self.n_threads,
            self.n_eval_eps,
            mask=mask_mat,
            delta_mat=delta_mat,
        )
        return self._restore_frozen_payoffs(payoff_mat - payoff_mat.T)

    def _apply_policy_freeze(self, payoff_mat, probs_mat):
        if not self.use_policy_freeze or self.lower_i >= payoff_mat.shape[0] - 1:
            return payoff_mat

        next_frozen_idx = self.lower_i + 1
        prev_size = self.lower_i + 1
        res_reward = np.dot(
            payoff_mat[next_frozen_idx, :prev_size],
            probs_mat[0, :prev_size],
        )
        best_before = self.max_last_reward
        improved_best = res_reward > best_before + self.upper_epsilon
        if improved_best:
            self.max_last_reward = res_reward
            self.best_model = copy.deepcopy(self.eval_policies[next_frozen_idx])
            self.best_payoff_line = payoff_mat[next_frozen_idx, :prev_size].copy()
        else:
            self.no_improve_count += 1

        if self.use_best_model_history and self.best_model is not None:
            self.eval_policies[next_frozen_idx] = copy.deepcopy(self.best_model)
            payoff_mat[next_frozen_idx, :prev_size] = self.best_payoff_line
            payoff_mat[:prev_size, next_frozen_idx] = -self.best_payoff_line
            print(
                f"keep lower policy {next_frozen_idx} as best model: "
                f"best_reward = {self.max_last_reward}"
            )

        if self.no_improve_count >= self.freeze_no_improve_k:
            self.lower_i += 1
            self.active_number = max(1, self.active_number - 1)
            self.last_reward = -np.inf
            self.max_last_reward = -np.inf
            self.no_improve_count = 0

            ex_payoff_mat = np.pad(
                self.frozen_payoff_mat,
                ((0, 1), (0, 1)),
                "constant",
                constant_values=0,
            )
            if self.use_best_model_history and self.best_model is not None:
                self.eval_policies[self.lower_i] = copy.deepcopy(self.best_model)
                best_model_payoff_mat = np.zeros_like(ex_payoff_mat)
                best_model_payoff_mat[-1, : len(self.best_payoff_line)] = self.best_payoff_line
                best_model_payoff_mat = best_model_payoff_mat - best_model_payoff_mat.T
                ex_payoff_mat += best_model_payoff_mat
            else:
                ex_payoff_mat += payoff_mat[: self.lower_i + 1, : self.lower_i + 1]
            self.frozen_payoff_mat = ex_payoff_mat
            self.best_model = None
            self.best_payoff_line = None
            print(f"freeze lower policy: lower_i = {self.lower_i}, frozen_mat = {self.frozen_payoff_mat}")
        else:
            self.last_reward = res_reward

        payoff_mat[: self.lower_i + 1, : self.lower_i + 1] = self.frozen_payoff_mat.copy()
        return payoff_mat

    def step(self):

        active_number = max(1, min(self.active_number, self.policy_num - 1 - self.lower_i))
        policy_size = self.lower_i + active_number
        probs_mat = self._build_sigma_matrix(active_number, policy_size)
        print("NeuPL active probs_mat = ", probs_mat)

        mixed_policy = mixing_policy(
            self.n_threads,
            self.eval_policies[:policy_size],
            probs_mat[0],
            self.device,
        )
        self.runner.num_env_steps = self.num_env_steps
        self.runner.all_args.global_steps = self.g_step
        self.runner.envs.world.oppo_policy = mixed_policy
        self.runner.set_policy_size(policy_size, probs_mat, leader_probs=probs_mat[0])

        logs = self.runner.run()
        logs = [logs]

        self.runner.save_as_filename("policy_active")
        self._refresh_eval_policies(probs_mat)
        self.g_step += self.num_env_steps

        matrix_size = policy_size + 1
        self.payoff_mat = self._evaluate_active_payoffs(matrix_size)
        self.payoff_mat = self._apply_policy_freeze(self.payoff_mat, probs_mat)
        print("NeuPL active payoff_mat = ", self.payoff_mat)

        probs, _ = self.meta_solver(self.payoff_mat)
        self.probs_now = probs[0].copy()

        if self.active_number + self.lower_i < self.policy_num - 1:
            self.active_number += 1

        return copy.deepcopy(self.eval_policies[: len(self.probs_now)]), self.probs_now.copy(), logs

    def get_sub_meta_probs(self):
        meta_probs, _ = self.meta_solver(self.payoff_mat)
        return meta_probs[0].copy()


    def get_payoff_mat(self):
        return copy.deepcopy(self.payoff_mat)
