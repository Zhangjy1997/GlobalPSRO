
import numpy as np
import torch
from onpolicy.algorithms.PSRO.utils.checkpoint import restore_eval_policy
from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy
import copy

def RM_runner_step(estimator, max_workers, control_dict):
    eff_size = control_dict["effect_size"]
    estimator.run()
    estimator.save_as_filename(f"frozen_estimator")
    target_probs_mat = estimator.probs_latent_mat.copy()
    G_avg = estimator.G_avg.copy()
    estimator.save_as_filename(f"frozen_{eff_size}_active")
    PE_dict = dict()
    for i in range(max_workers):
        PE_dict[f"probs_{eff_size}_{i}"] = target_probs_mat[i]
        avg_PE = np.dot(G_avg[i], target_probs_mat[i])
        PE_dict[f"avg_PE_{eff_size}_{i}"] = avg_PE
        PE_dict[f"G_avg_{eff_size}_{i}"] = G_avg[i]
    return PE_dict


class SimplexGlobal_PSRO_trainer:
    """Global PSRO reported in the ICML paper.

    Candidate policies share parameters through a conditional neural network.
    """

    def __init__(self, args, anchor_policies, shared_policies, eval_policies, runner, RM_estimator, evaluator, meta_solver, role_names, save_dir, trans2tablepolicy = None, device = torch.device("cpu")):
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
        self.eval_policies = eval_policies
        self.runner = runner
        self.RM_estimator = RM_estimator
        self.eval = evaluator
        self.meta_solver = meta_solver
        self.eval_envs = copy.deepcopy(self.eval.envs)
        self.role_names = role_names
        self.save_dir = save_dir
        self.device = device
        self.n_threads = self.args.n_rollout_threads
        self.gamma_alpha = self.args.gamma_alpha
        self.use_exploitation_only = getattr(args, "use_exploitation_only", False)
        self.target_index = 0 if self.use_exploitation_only else 1
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.num_env_steps = self.args.num_env_steps
        self.policy_num = self.args.population_size
        self.condition_size = getattr(self.args, "latent_size", self.policy_num)
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]
        self.eval_policies[0] = self.policies_anchor[0]
        self.upper_epsilon = args.upper_epsilon
        self.upper_std = args.upper_std
        self.payoff_mat = np.zeros((1,1))
        self.prev_payoff_mat = np.zeros((1,1))
        self.effect_population_size = 1
        self.lower_i = 0
        self.RM_round = True
        self.game = copy.deepcopy(self.eval.envs.world.standard_game)
        self.trans2tablepolicy = trans2tablepolicy

        self.use_random_select = args.use_random_select
        print("MC methods!")

        self.control_dict = dict()
        self.control_dict["last_PE"] = 0
        self.control_dict["last_probs"] = np.ones(1)
        self.control_dict["effect_size"] = 1
        self.prev_probs = np.ones(1)
        print("The metric is Upper value!")
        print("Use max(R_(t-1),avg_PE)!")

        self.max_workers = 1 if self.use_exploitation_only else args.max_workers
        self.terminal_state = False
        self.PE_state_dict = dict()
        self.PE_state_dict["last_PE"] = -1
        self.select_probs = np.ones(1)

    def pad_condition(self, condition):
        condition = np.asarray(condition)
        if len(condition) > self.condition_size:
            raise ValueError(
                "condition length {} exceeds latent/condition size {}".format(
                    len(condition), self.condition_size
                )
            )
        condition_full = np.zeros(self.condition_size, dtype=condition.dtype)
        condition_full[:len(condition)] = condition
        return condition_full


    def min_metric(self, PE_dicts, worker_n, initial_state=False):
        eff_size = self.control_dict["effect_size"]
        if self.use_exploitation_only:
            probs = np.asarray(PE_dicts[f"probs_{eff_size}_0"], dtype=float)
            probs = np.clip(probs, 0.0, None)
            probs_sum = probs.sum()
            self.select_probs = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
            return [0], np.ones(1, dtype=float)

        avg_PE_list = [PE_dicts[f"avg_PE_{eff_size}_{i}"] for i in range(worker_n)]
        R_upper_list = []
        max_arr = []
        for i in range(worker_n):
            probs = PE_dicts[f"probs_{eff_size}_{i}"]
            avg_G = PE_dicts[f"avg_PE_{eff_size}_{i}"]
            G_line = PE_dicts[f"G_avg_{eff_size}_{i}"]
            if initial_state:
                R_upper_value = avg_G
            else:
                regular_probs = probs[:-1] / (1 - probs[-1]) if (1 - probs[-1]) > 1e-5 else self.PE_state_dict["last_probs"].copy()
                R_upper_value = (1 - probs[-1]) * (
                    self.PE_state_dict["last_PE"]
                    + np.dot(regular_probs - self.PE_state_dict["last_probs"], G_line[:-1])
                ) + probs[-1] * G_line[-1]
            R_upper_list.append(R_upper_value)
            max_arr.append(max(R_upper_value, avg_G))

        print("R_upper = ", R_upper_list)
        print("avg_PE = ", avg_PE_list)
        print("Select candidate metrics!")

        if getattr(self, "use_random_select", False):
            chosen_idx = np.array([np.random.randint(worker_n)], dtype=int) if worker_n > 0 else np.array([0], dtype=int)
            weights = np.ones(1, dtype=float)
            probs = np.asarray(PE_dicts[f"probs_{eff_size}_{int(chosen_idx[0])}"], dtype=float)
            probs = np.clip(probs, 0.0, None)
            probs_sum = probs.sum()
            self.PE_state_dict["last_probs"] = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
            return chosen_idx.tolist(), weights

        arr = np.array(R_upper_list, dtype=float)
        max_arr = np.array(max_arr, dtype=float)
        probs_getter = lambda i: PE_dicts[f"probs_{eff_size}_{i}"]

        selection_arr = arr.copy()
        allowed_mask = np.ones_like(arr, dtype=bool)

        target_choice_idx = 0
        target_score = arr[target_choice_idx]
        threshold_mask = arr <= target_score
        allowed_mask &= threshold_mask
        allowed_mask[target_choice_idx] = True
        selection_arr[target_choice_idx] -= self.upper_epsilon

        feasible_idx = np.where(allowed_mask)[0]
        if feasible_idx.size == 0:
            feasible_idx = np.array([target_choice_idx], dtype=int)

        sorted_idx = feasible_idx[np.argsort(selection_arr[feasible_idx], kind="mergesort")]
        chosen_idx = int(sorted_idx[0])

        print("chosen_idx = {}".format(chosen_idx))

        probs = np.asarray(probs_getter(chosen_idx), dtype=float)
        probs = np.clip(probs, 0.0, None)
        probs_sum = probs.sum()
        self.PE_state_dict["last_probs"] = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
        self.PE_state_dict["last_PE"] = max_arr[chosen_idx]

        return [chosen_idx], np.ones(1, dtype=float)


    def step(self):
        if self.RM_round:
            print("Eval round!")
            if self.control_dict["effect_size"] <= 1:
                mixed_policy_ = mixing_policy(self.n_threads, self.eval_policies[:1], np.ones(1), self.device)
                self.RM_estimator.all_args.global_steps = self.g_step
                self.RM_estimator.envs.world.oppo_policy = mixed_policy_
                self.RM_estimator.set_policy_size(0, np.ones((1,1)), np.ones(1))
                PE_dict = RM_runner_step(self.RM_estimator, 1, self.control_dict)

                temp_policy_ = copy.deepcopy(self.eval_policies[-1])
                restore_eval_policy(temp_policy_, self.save_dir, head_str=f"frozen_estimator")
                p_sigma = np.ones(1)  # numpy 1D vector
                p_sigma_full = self.pad_condition(p_sigma)
                temp_policy_.set_sigma(np.tile(p_sigma_full,(1,1)))
                temp_policy_.set_fusion_false()
                policy_ = self.trans2tablepolicy(self.game, range(2), [temp_policy_], np.ones(1), device=self.device)

                self.eval_policies[1] = policy_

                self.effect_population_size += 1
                self.select_probs = PE_dict["probs_1_0"].copy()
                if not self.use_exploitation_only:
                    self.PE_state_dict["last_PE"] = PE_dict["avg_PE_1_0"]
                    self.PE_state_dict["last_probs"] = PE_dict["probs_1_0"]
                    print("last_PE = ", self.PE_state_dict["last_PE"])
            else:
                extended_policy_set = copy.deepcopy(self.eval_policies[:(self.control_dict["effect_size"]-1)] + self.can_policies)
                probs_ini = np.ones(len(extended_policy_set))
                probs_ini = probs_ini / np.sum(probs_ini)
                mixed_policy_ = mixing_policy(self.n_threads, extended_policy_set, probs_ini, self.device)
                probs_ini = np.ones(self.control_dict["effect_size"])
                probs_ini = probs_ini / np.sum(probs_ini)
                self.RM_estimator.all_args.global_steps = self.g_step
                self.RM_estimator.envs.world.oppo_policy = copy.deepcopy(mixed_policy_)
                self.RM_estimator.set_policy_size(self.control_dict["effect_size"]-1, self.samples, probs_ini)
                PE_dict = RM_runner_step(self.RM_estimator, len(self.can_policies), self.control_dict)

                min_indices, weights = self.min_metric(PE_dict, len(self.can_policies))

                real_num = len(min_indices)
                print("mixed number = ", real_num)
                print("weights = ", weights)

                estimated_policies = []

                for i in range(real_num):
                    estimated_policies.append(copy.deepcopy(self.eval_policies[-1]))
                    restore_eval_policy(estimated_policies[i], self.save_dir, head_str=f"frozen_estimator")
                    p_sigma = self.samples[min_indices[i]]  # numpy 1D vector
                    p_sigma_full = self.pad_condition(p_sigma)
                    estimated_policies[i].set_sigma(np.tile(p_sigma_full,(1,1)))
                    estimated_policies[i].set_fusion_false()

                policy_ = self.trans2tablepolicy(self.game, range(2), [self.can_policies[i] for i in min_indices], weights, device=self.device)
                rm_policy_ = self.trans2tablepolicy(self.game, range(2), estimated_policies, weights, device=self.device)
                self.eval_policies[self.control_dict["effect_size"]-1] = policy_
                self.eval_policies[self.control_dict["effect_size"]] = rm_policy_

                self.effect_population_size += 2

            self.eval.update_policy(self.eval_policies[:self.effect_population_size])
            mask_mat = np.zeros((self.effect_population_size, self.effect_population_size), dtype=bool)
            if self.control_dict["effect_size"] <= 1:
                mask_mat[-1, :] = True
                mask_mat[-1, -1] = False
                ex_payoff_mat = np.pad(self.payoff_mat, ((0, 1), (0, 1)), 'constant', constant_values=0)
            else:
                mask_mat[-2:, :] = True
                mask_mat[-2, -2] = False
                mask_mat[-2:, -1] = False
                ex_payoff_mat = np.pad(self.payoff_mat, ((0, 2), (0, 2)), 'constant', constant_values=0)
            delta_mat = self.upper_std * np.ones((self.effect_population_size, self.effect_population_size))
            print("Evaluator start!")
            payoff_mat_ = self.eval.get_win_prob_with_mask(self.n_threads, self.n_eval_eps, mask = mask_mat, delta_mat = delta_mat)
            payoff_mat_ = payoff_mat_ - payoff_mat_.T
            self.payoff_mat = ex_payoff_mat + payoff_mat_
            print("payoff_mat = ", self.payoff_mat)

        else:
            print("BR round!")
            alpha = np.ones(self.control_dict["effect_size"])
            print("alpha = ", alpha)
            probs, _ = self.meta_solver(self.payoff_mat)
            if self.use_exploitation_only:
                samples = probs[0][np.newaxis, :].copy()
            else:
                samples = np.random.dirichlet(alpha, size=self.max_workers)
                samples[self.target_index] = probs[0]

            probs_ini = np.ones(self.effect_population_size)
            probs_ini /= np.sum(probs_ini)
            if not self.use_exploitation_only:
                samples[0] = probs_ini
            print("samples = ", samples)
            mixed_policy_ = mixing_policy(self.n_threads, self.eval_policies[:self.effect_population_size], probs_ini, self.device)
            self.runner.all_args.global_steps = self.g_step
            self.runner.envs.world.oppo_policy = copy.deepcopy(mixed_policy_)
            runner_anytime_inx = None if self.use_exploitation_only else 0
            self.runner.set_policy_size(self.effect_population_size, samples, leader_probs=samples[self.target_index].copy(), anytime_inx=runner_anytime_inx)
            self.runner.run()
            self.runner.save_as_filename(f"frozen_runner")

            if not self.use_exploitation_only:
                target_probs = self.runner.probs.copy()
                samples[0] = target_probs

            temp_policy_list = []
            eff_size = self.control_dict["effect_size"]
            for i in range(self.max_workers):
                temp_policy_list.append(copy.deepcopy(self.eval_policies[-1]))
                restore_eval_policy(temp_policy_list[i], self.save_dir, head_str=f"frozen_runner")
                p_sigma = samples[i]  # numpy 1D vector
                p_sigma_full = self.pad_condition(p_sigma)
                temp_policy_list[i].set_sigma(np.tile(p_sigma_full,(1,1)))
                temp_policy_list[i].set_fusion_false()

            self.can_policies = temp_policy_list
            PE_dict = dict()
            if self.use_exploitation_only:
                target_probs = np.ones_like(samples[0], dtype=float)
                target_probs /= np.sum(target_probs)
                PE_dict[f"probs_{eff_size}_0"] = target_probs.copy()
            else:
                target_probs = self.runner.probs.copy()
                PE_dict[f"probs_{eff_size}_0"] = target_probs.copy()
                G_avg = self.runner.G_avg.copy()
                avg_PE = np.dot(G_avg, target_probs)
                PE_dict[f"avg_PE_{eff_size}_0"] = avg_PE
                PE_dict[f"G_avg_{eff_size}_0"] = G_avg

            probs = PE_dict[f"probs_{self.control_dict['effect_size']}_0"]
            self.select_probs = probs.copy()
            if not self.use_exploitation_only:
                avg_G = PE_dict[f"avg_PE_{self.control_dict['effect_size']}_0"]
                G_line = PE_dict[f"G_avg_{self.control_dict['effect_size']}_0"]
                regular_probs = probs[:-1] / (1 - probs[-1]) if (1 - probs[-1]) > 1e-5 else self.PE_state_dict["last_probs"].copy()
                R_upper_value = (1 - probs[-1]) * (
                    self.PE_state_dict["last_PE"]
                    + np.dot(regular_probs - self.PE_state_dict["last_probs"], G_line[:-1])
                ) + probs[-1] * G_line[-1]
                self.PE_state_dict["last_PE"] = max(R_upper_value, avg_G)
                self.PE_state_dict["last_probs"] = probs

                print("last_PE = ", self.PE_state_dict["last_PE"])

            if 0 <= self.target_index < len(self.can_policies):
                reduced_indices = [self.target_index] + [i for i in range(len(self.can_policies)) if i != self.target_index]
            else:
                reduced_indices = list(range(len(self.can_policies)))
            self.can_policies = [self.can_policies[i] for i in reduced_indices]
            self.samples = samples[reduced_indices].copy()

        self.g_step += (self.num_env_steps)
        self.RM_round = not self.RM_round
        self.control_dict["effect_size"] += 1
        select_probs = self.select_probs.copy() if self.use_exploitation_only else self.PE_state_dict["last_probs"].copy()

        return copy.deepcopy(self.eval_policies[:self.control_dict["effect_size"]-1]), select_probs


    def get_payoff_mat(self, scale_n = None):
        if scale_n is None:
            re_mat = copy.deepcopy(self.payoff_mat)
        else:
            re_mat = copy.deepcopy(self.payoff_mat[:scale_n, :scale_n])
        return re_mat
