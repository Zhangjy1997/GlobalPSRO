
import numpy as np
import torch
import multiprocessing
from onpolicy.algorithms.PSRO.utils.checkpoint import restore_eval_policy, transfer_policy_A2B_full
from onpolicy.algorithms.PSRO.utils.mixing_policy import Parallel_mixing_policy as mixing_policy
import copy
import pickle

def BR_runner_step_id(runner, policy_idx, control_dict, gpu_id):
    eff_size = control_dict["effect_size"]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    runner.transfer_model_to(device)
    runner.set_policy_inx(policy_idx)
    train_logs = runner.run()
    runner.save_as_filename(f"frozen_{eff_size}_{policy_idx}_active")
    PE_dict = dict()
    serialized_result = pickle.dumps((train_logs, PE_dict))
    return serialized_result

def RM_runner_step_id(estimator, policy_idx, control_dict, gpu_id):
    eff_size = control_dict["effect_size"]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    estimator.transfer_model_to(device)
    esm_logs = estimator.run()
    target_probs = estimator.probs.copy()
    estimator.save_as_filename(f"frozen_{eff_size}_{policy_idx}_active")
    PE_dict = dict()
    PE_dict[f"probs_{eff_size}_{policy_idx}"] = target_probs.copy()
    G_avg = estimator.G_avg.copy()
    PE_dict[f"avg_PE_{eff_size}_{policy_idx}"] = np.dot(G_avg, target_probs)
    PE_dict[f"G_avg_{eff_size}_{policy_idx}"] = G_avg
    serialized_result = pickle.dumps((esm_logs, PE_dict))
    return serialized_result

def parallel_BR_training_id(runners, policy_idx, control_dict):
    num_gpus = torch.cuda.device_count()
    with multiprocessing.Pool(processes=len(runners)) as pool:
        results = []
        for i in range(len(runners)):
            if i == 0:
                results.append(pool.apply_async(RM_runner_step_id, args=(copy.deepcopy(runners[i]), policy_idx[i], control_dict, i % num_gpus)))
            else:
                results.append(pool.apply_async(BR_runner_step_id, args=(copy.deepcopy(runners[i]), policy_idx[i], control_dict, i % num_gpus)))
        pool.close()
        pool.join()

        logs = []
        PE_dict = dict()
        for result in results:
            serialized_result = result.get()
            logs_, PE_dict_ = pickle.loads(serialized_result)
            logs.append(logs_)
            PE_dict.update(PE_dict_)

    return logs, PE_dict

def parallel_RM_training_id(runners, policy_idx, control_dict):
    num_gpus = torch.cuda.device_count()
    with multiprocessing.Pool(processes=len(runners)) as pool:
        results = [pool.apply_async(RM_runner_step_id, args=(copy.deepcopy(runners[i]), policy_idx[i], control_dict, i % num_gpus)) for i in range(len(runners))]
        pool.close()
        pool.join()

        logs = []
        PE_dict = dict()
        for result in results:
            serialized_result = result.get()
            logs_, PE_dict_ = pickle.loads(serialized_result)
            logs.append(logs_)
            PE_dict.update(PE_dict_)

    return logs, PE_dict


class NaiveGlobal_PSRO_trainer:
    """Naive Global PSRO implementation.

    This follows the Global PSRO principle, while candidate generation and PE evaluation are independent across candidates. It is recommended to run this algorithm on a multi-GPU device.
    """

    def __init__(self, args, anchor_policies, shared_policies, eval_policies, runners, RM_estimators, evaluator, meta_solver, role_names, save_dir, device = torch.device("cpu")):
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
        self.eval_policies = eval_policies
        self.runners = runners
        self.RM_estimators = RM_estimators
        self.eval = evaluator
        self.meta_solver = meta_solver
        self.save_dir = save_dir
        self.device = device
        self.n_threads = self.args.n_rollout_threads
        self.pr_meta = meta_solver is not None and self.n_threads > 1
        self.max_workers = args.max_workers
        self.target_index = 1 if self.pr_meta and self.max_workers > 1 else 0
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.num_env_steps = self.args.num_env_steps
        self.eval_policies[0] = self.policies_anchor[0]
        self.upper_epsilon = args.upper_epsilon
        self.upper_std = args.upper_std
        self.payoff_mat = np.zeros((1,1))
        self.effect_population_size = 1
        self.RM_round = True

        print("MC methods!")

        self.control_dict = dict()
        self.control_dict["effect_size"] = 1
        print("The metric is Upper value!")
        print("Use max(R_(t-1),avg_PE)!")

        self.PE_state_dict = dict()
        self.PE_state_dict["last_PE"] = -1

    def min_metric(self, PE_dicts, initial_state = False):
        eff_size = self.control_dict["effect_size"]
        avg_PE_list = [PE_dicts[f"avg_PE_{eff_size}_{i}"] for i in range(self.max_workers)]
        R_upper_list = []
        for i in range(self.max_workers):
            probs = PE_dicts[f"probs_{eff_size}_{i}"]
            avg_G = PE_dicts[f"avg_PE_{eff_size}_{i}"]
            G_line = PE_dicts[f"G_avg_{eff_size}_{i}"]
            if initial_state:
                R_upper_list.append(avg_G)
            else:
                regular_probs = probs[:-1] / (1 - probs[-1]) if (1 - probs[-1]) > 1e-5 else self.PE_state_dict["last_probs"].copy()
                R_upper_value = (1 - probs[-1]) * (
                    self.PE_state_dict["last_PE"]
                    + np.dot(regular_probs - self.PE_state_dict["last_probs"], G_line[:-1])
                ) + probs[-1] * G_line[-1]
                R_upper_list.append(R_upper_value)

        print("R_upper = ", R_upper_list)
        print("avg_PE = ", avg_PE_list)

        min_exp = np.min(R_upper_list)
        min_index = np.argmin(R_upper_list)
        if min_exp >= R_upper_list[self.target_index] - self.upper_epsilon:
            min_exp = R_upper_list[self.target_index]
            min_index = self.target_index

        self.PE_state_dict["last_PE"] = max(min_exp, avg_PE_list[min_index])
        self.PE_state_dict["last_probs"] = PE_dicts[f"probs_{eff_size}_{min_index}"].copy()
        return min_exp, min_index


    def step(self):
        if self.RM_round:
            print("Eval round!")
            if self.control_dict["effect_size"] <= 1:
                for i in range(self.max_workers):
                    probs_ini = np.ones(self.effect_population_size)
                    probs_ini /= np.sum(probs_ini)
                    mixed_policy_ = mixing_policy(self.n_threads, self.eval_policies[:self.effect_population_size], probs_ini, self.device)
                    self.RM_estimators[i].all_args.global_steps = self.g_step
                    self.RM_estimators[i].envs.world.oppo_policy = copy.deepcopy(mixed_policy_)
                    self.RM_estimators[i].set_policy_n_prob(i, probs_ini)
                logs, PE_dict = parallel_RM_training_id(self.RM_estimators, [i for i in range(self.max_workers)], self.control_dict)
                min_exp, min_index = self.min_metric(PE_dict, initial_state=True)

                transfer_policy_A2B_full(self.save_dir, f"frozen_{self.control_dict['effect_size']}_{min_index}_active", f"policy_{self.control_dict['effect_size']}")
                restore_eval_policy(self.eval_policies[self.control_dict['effect_size']], self.save_dir, head_str=f"policy_{self.control_dict['effect_size']}")
                self.effect_population_size += 1
            else:
                for i in range(self.max_workers):
                    self.RM_estimators[i].inherit_policy(self.save_dir, head_str = f"frozen_{self.control_dict['effect_size']-1}_0_active")
                    probs_ini = np.ones(self.control_dict["effect_size"])
                    probs_ini /= np.sum(probs_ini)
                    restore_eval_policy(self.eval_policies[self.control_dict["effect_size"]-1], self.save_dir, head_str=f"frozen_{self.control_dict['effect_size']-1}_{i}_active")
                    mixed_policy_ = mixing_policy(self.n_threads, self.eval_policies[:self.control_dict["effect_size"]], probs_ini, self.device)
                    self.RM_estimators[i].all_args.global_steps = self.g_step
                    self.RM_estimators[i].envs.world.oppo_policy = copy.deepcopy(mixed_policy_)
                    self.RM_estimators[i].set_policy_n_prob(i, probs_ini)
                logs, PE_dict = parallel_RM_training_id(self.RM_estimators, [i for i in range(self.max_workers)], self.control_dict)
                min_exp, min_index = self.min_metric(PE_dict, initial_state=False)

                transfer_policy_A2B_full(self.save_dir, f"frozen_{self.control_dict['effect_size']-1}_{min_index}_active", f"policy_{self.control_dict['effect_size']-1}")
                transfer_policy_A2B_full(self.save_dir, f"frozen_{self.control_dict['effect_size']}_{min_index}_active", f"policy_{self.control_dict['effect_size']}")
                restore_eval_policy(self.eval_policies[self.control_dict["effect_size"]-1], self.save_dir, head_str=f"policy_{self.control_dict['effect_size']-1}")
                restore_eval_policy(self.eval_policies[self.control_dict["effect_size"]], self.save_dir, head_str=f"policy_{self.control_dict['effect_size']}")
                self.effect_population_size += 2

            if self.pr_meta:
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
            samples = np.random.dirichlet(alpha, size=self.max_workers)
            if self.pr_meta:
                probs, _ = self.meta_solver(self.payoff_mat)
                samples[1] = probs[0]
            print("samples = ", samples)
            for i in range(self.max_workers):
                if i > 0:
                    self.runners[i].trainer.policy.reset_policy()

            if self.control_dict["effect_size"] > 1:
                self.runners[0].inherit_policy(self.save_dir, head_str = f"policy_{self.control_dict['effect_size'] - 1}")
            
            for i in range(self.max_workers):
                mixed_policy_ = mixing_policy(self.n_threads, self.eval_policies[:self.effect_population_size], samples[i], self.device)
                self.runners[i].all_args.global_steps = self.g_step
                self.runners[i].envs.world.oppo_policy = copy.deepcopy(mixed_policy_)
                if i==0:
                    probs_ini = np.ones(self.effect_population_size)
                    probs_ini /= np.sum(probs_ini)
                    self.runners[i].set_policy_n_prob(i, probs_ini)
                else:
                    self.runners[i].set_policy_inx(i)

            logs, PE_dict = parallel_BR_training_id(self.runners, [i for i in range(self.max_workers)], self.control_dict)
            probs = PE_dict[f"probs_{self.control_dict['effect_size']}_0"]
            avg_G = PE_dict[f"avg_PE_{self.control_dict['effect_size']}_0"]
            G_line = PE_dict[f"G_avg_{self.control_dict['effect_size']}_0"]
            regular_probs = probs[:-1] / (1 - probs[-1]) if (1 - probs[-1]) > 1e-5 else self.PE_state_dict["last_probs"].copy()
            R_upper_value = (1 - probs[-1]) * (
                self.PE_state_dict["last_PE"]
                + np.dot(regular_probs - self.PE_state_dict["last_probs"], G_line[:-1])
            ) + probs[-1] * G_line[-1]
            self.PE_state_dict["last_PE"] = max(R_upper_value, avg_G)
            self.PE_state_dict["last_probs"] = probs

        self.g_step += (self.num_env_steps)
        self.RM_round = not self.RM_round
        self.control_dict["effect_size"] += 1
        select_probs = self.PE_state_dict["last_probs"].copy()

        return copy.deepcopy(self.eval_policies[:self.control_dict["effect_size"]-1]), select_probs, logs
    

    def get_payoff_mat(self, scale_n = None):
        if scale_n is None:
            re_mat = copy.deepcopy(self.payoff_mat)
        else:
            re_mat = copy.deepcopy(self.payoff_mat[:scale_n, :scale_n])
        return re_mat
