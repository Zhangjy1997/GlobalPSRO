import numpy as np
import copy

def _t2n(x):
    return x.detach().cpu().numpy()

class eval_match:
    def __init__(self, policies, envs, policy_type = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        self.envs = envs
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.episode_length = self.envs.episode_length
        self.actor = self.policies[0].actor

    def update_policy(self, policies, policy_type = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.actor = self.policies[0].actor
    
    def calc_payoff(self, n_rollout_threads, total_episodes, inx_p1, inx_p2, delta = None):
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs, eval_a_acts = self.envs.reset()
        total_round = 0
        total_reward = 0
        eval_r_list = []
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        if delta is None:
            delta = np.inf

        while True:
            for episodes in range(total_episodes):
                for eval_step in range(self.episode_length):
                    selected_p1_policy.actor.eval()
                    eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks),
                                                        np.concatenate(eval_a_acts),
                                                        deterministic=True)
                    eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                    eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                    # Obser reward and next obs
                    # print("action network:", eval_actions_env)
                    eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)
                    # print(eval_infos)
                    for i in range(n_rollout_threads):
                        total_reward += eval_rewards[i][0][0]
                        if eval_dones[i].all():
                            eval_r_list.append(eval_rewards[i][0][0])
                            total_round += 1
                            if self.policy_type[inx_p1] == "mixed": 
                                selected_p1_policy.update_index_channel(i)
                            if self.policy_type[inx_p2] == "mixed": 
                                self.envs.world.oppo_policy.update_index_channel(i)

                    eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                    eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                    eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            std_ = np.std(np.array(eval_r_list))/np.sqrt(len(eval_r_list))
            payoff_ = total_reward/total_round
            print("policy {} vs policy {} : payoff {}, std {}, delta {}".format(inx_p1, inx_p2, payoff_, std_, delta))
            if std_ < delta:
                break
        
        return total_round, total_reward

    def simple_policy_eval(self, eval_eps, n_rollout_threads, policies, oppo_policies = None, policy_type = None, oppo_policy_type = None, delta = None, mask = None):
        n_p1 = len(policies)
        if policy_type is None:
            policy_type = dict()
            for i in range(n_p1):
                policy_type[i] = "pure"

        if oppo_policies is None:
            oppo_policies = copy.deepcopy(policies)
            oppo_policy_type = copy.deepcopy(policy_type)

        n_p2 = len(oppo_policies)

        if oppo_policy_type is None:
            oppo_policy_type = dict()
            for i in range(n_p2):
                oppo_policy_type[i] = "pure"

        rewards_mat = np.zeros((n_p1, n_p2))
        round_mat = np.zeros((n_p1, n_p2), dtype=int)
        std_mat = np.zeros((n_p1, n_p2))
        if delta is None:
            delta = np.inf

        if mask is None:
            mask = np.ones((n_p1, n_p2), dtype=bool)

        for p1_i in range(n_p1):
            for p2_i in range(n_p2):
                if mask[p1_i, p2_i] == False:
                    continue
                selected_p1_policy = copy.deepcopy(policies[p1_i])
                self.envs.world.oppo_policy = copy.deepcopy(oppo_policies[p2_i])
                eval_obs, eval_a_acts = self.envs.reset()
                total_round = 0
                total_reward = 0
                eva_r_list = []
                eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

                while True:
                    for episodes in range(eval_eps):
                        for eval_step in range(self.episode_length):
                            selected_p1_policy.actor.eval()
                            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                                np.concatenate(eval_rnn_states),
                                                                np.concatenate(eval_masks),
                                                                np.concatenate(eval_a_acts),
                                                                deterministic=True)
                            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                            # Obser reward and next obs
                            # print("action network:", eval_actions_env)
                            eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)
                            # print(eval_infos)
                            for i in range(n_rollout_threads):
                                total_reward += eval_rewards[i][0][0]
                                if eval_dones[i].all():
                                    total_round += 1
                                    eva_r_list.append(eval_rewards[i][0][0])
                                    if policy_type[p1_i] == "mixed": 
                                        selected_p1_policy.update_index_channel(i)
                                    if oppo_policy_type[p2_i] == "mixed": 
                                        self.envs.world.oppo_policy.update_index_channel(i)

                            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                        
                        # print("policy {} vs policy {} :episodes: {}/{}".format(p1_i, p2_i, episodes, eval_eps))
                    std_ = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))
                    print("standard value of match: {} vs {} = {}, delta = {}".format(p1_i, p2_i, std_, delta))
                    if std_ < delta:
                        break

                round_mat[p1_i, p2_i] = total_round
                rewards_mat[p1_i, p2_i] = total_reward
                std_mat[p1_i, p2_i] = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))

        mask_nonzero = round_mat > 0
        payoff_mat = np.zeros_like(rewards_mat)
        payoff_mat[mask_nonzero] = rewards_mat[mask_nonzero] / round_mat[mask_nonzero]
                
        return payoff_mat, std_mat


    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None, delta_mat = None):
        if mask is None:
            mask = np.ones((self.policy_num, self.policy_num), dtype=bool)
        for i in range(self.policy_num):
            for j in range(i):
                if mask[i][j]:
                    total_round_temp, total_reward_ = self.calc_payoff(n_rollout_threads, episode_num, i ,j, delta=None if delta_mat is None else delta_mat[i,j])
                    self.total_round_mat[i,j] = total_round_temp
                    self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat
