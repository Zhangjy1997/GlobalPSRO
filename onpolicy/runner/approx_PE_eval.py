import copy
import os
import time

import numpy as np
import torch

from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy


class ApproxPEEvaluator:
    """External neural RL approx-PE/probsE evaluator for experiment scripts."""

    def __init__(self, args, RM_estimator, role_name, save_dir, device=torch.device("cpu")):
        if RM_estimator is None:
            raise ValueError("ApproxPEEvaluator requires RM_estimator")

        self.args = args
        self.RM_estimator = RM_estimator
        self.role_name = role_name[0] if isinstance(role_name, (list, tuple)) else role_name
        self.save_dir = str(save_dir)
        self.device = device

        self.n_eval_eps = getattr(args, "eval_episode_num", 1)
        self.upper_std = getattr(args, "upper_std", 0.03)
        self.use_real_estimator = getattr(args, "use_real_estimator", True)
        self.approx_PE_steps = getattr(args, "approx_PE_steps", getattr(args, "RM_steps", getattr(args, "num_env_steps", 0)))
        self.approx_PE_eval_episodes = getattr(args, "approx_PE_eval_episodes", self.n_eval_eps)
        self.approx_PE_std = getattr(args, "approx_PE_std", self.upper_std)

        self.approx_PE_time = 0
        self.approx_PE_history = []
        self.approx_probsE_history = []
        self.approx_avg_PE_history = []
        self.approx_G_avg_history = []
        self.last_info = None
        self.last_policy_support = None

        if hasattr(self.RM_estimator, "all_args"):
            self.RM_estimator.all_args = copy.deepcopy(self.RM_estimator.all_args)

    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=float).copy()
        probs = np.clip(probs, 0.0, None)
        prob_sum = probs.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            return np.ones_like(probs) / len(probs)
        return probs / prob_sum

    def _get_payoff_sigma(self, estimator):
        try:
            return estimator.get_payoff_sigma(
                self.approx_PE_eval_episodes,
                self.approx_PE_std,
                self.use_real_estimator,
            )
        except TypeError:
            return estimator.get_payoff_sigma(self.approx_PE_eval_episodes, self.approx_PE_std)

    def _save_histories(self):
        np.save(
            os.path.join(self.save_dir, "approx_PE_" + str(self.role_name) + ".npy"),
            np.asarray(self.approx_PE_history, dtype=float),
        )
        np.save(
            os.path.join(self.save_dir, "probsE_" + str(self.role_name) + ".npy"),
            np.asarray(self.approx_probsE_history, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(self.save_dir, "approx_avg_PE_" + str(self.role_name) + ".npy"),
            np.asarray(self.approx_avg_PE_history, dtype=float),
        )
        np.save(
            os.path.join(self.save_dir, "approx_G_avg_" + str(self.role_name) + ".npy"),
            np.asarray(self.approx_G_avg_history, dtype=object),
            allow_pickle=True,
        )

    def _build_policy_support(self, eval_policies, effect_population_size, probs):
        probs = self._normalize_probs(probs)
        policy_count = min(len(probs), effect_population_size)
        support_probs = probs[:policy_count].copy()
        support_probs = self._normalize_probs(support_probs)
        support_policies = copy.deepcopy(eval_policies[:policy_count])
        return support_policies, support_probs

    def run(self, eval_policies, effect_population_size, g_step, n_threads):
        estimator = self.RM_estimator
        approx_start = time.time()

        if hasattr(estimator, "trainer") and hasattr(estimator.trainer, "policy"):
            estimator.trainer.policy.reset_policy()

        init_probs = np.ones(effect_population_size, dtype=float)
        init_probs = init_probs / init_probs.sum()
        mixed_policy = mixing_policy(
            n_threads,
            eval_policies[:effect_population_size],
            init_probs,
            self.device,
        )

        estimator.num_env_steps = self.approx_PE_steps
        estimator.all_args.global_steps = g_step
        estimator.envs.world.oppo_policy = copy.deepcopy(mixed_policy)
        if hasattr(estimator, "set_policy_n_prob"):
            estimator.set_policy_n_prob(0, init_probs)
        else:
            estimator.set_policy_inx(0)

        print("Approx PE estimator start!")
        approx_logs = estimator.run()
        approx_PE, approx_std = self._get_payoff_sigma(estimator)
        probsE = np.asarray(estimator.probs, dtype=float).copy()
        G_avg = np.asarray(getattr(estimator, "G_avg", np.zeros_like(probsE)), dtype=float).copy()
        approx_avg_PE = float(np.dot(G_avg, probsE)) if G_avg.shape == probsE.shape else np.nan

        self.approx_PE_time += time.time() - approx_start
        self.approx_PE_history.append(float(np.asarray(approx_PE).reshape(-1)[0]))
        self.approx_probsE_history.append(probsE.copy())
        self.approx_avg_PE_history.append(approx_avg_PE)
        self.approx_G_avg_history.append(G_avg.copy())
        self.last_info = {
            "approx_PE": copy.deepcopy(approx_PE),
            "approx_PE_std": copy.deepcopy(approx_std),
            "approx_probsE": probsE.copy(),
            "approx_avg_PE": approx_avg_PE,
            "approx_G_avg": G_avg.copy(),
            "round": effect_population_size - 1,
        }
        support_policies, support_probs = self._build_policy_support(eval_policies, effect_population_size, probsE)
        self.last_policy_support = {
            "policies": support_policies,
            "probs": support_probs,
            "round": effect_population_size - 1,
        }
        self._save_histories()

        return approx_logs
