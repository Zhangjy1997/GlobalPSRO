import time
import numpy as np
import torch
import copy
from onpolicy.runner.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class EXP3_Runner(Runner):
    """Runner for EXP3-based response evaluation and policy training."""
    def __init__(self, config):
        super(EXP3_Runner, self).__init__(config)
        self.all_args = copy.deepcopy(self.all_args)
        self.exp3_interval = self.all_args.RM_interval
        self.pre_train_steps = self.all_args.RM_pre_train_steps
        self.post_train_steps = self.all_args.RM_post_train_steps
        self.avg_G_last_N = self.all_args.avg_G_last_N
        self.exp_yita_coef = self.all_args.RM_yita_coef

    def run(self):
        self.warmup()
        self.exp3_round = self.multi_round

        self.G_history_line = np.zeros(self.support_K)
        self.G_his_match = 0
        self.total_done = 0
        self.sub_ep = 0
        self.sub_G_line = np.zeros(self.support_K)
        exp3_time = 0
        PE_est_time = 0
        self.regret_value = 1e-4
        self.G_avg_array = [[] for _ in range(self.support_K)]
        self.G_avg = np.zeros(self.support_K)

        episodes = int(self.pre_train_steps + self.num_env_steps + self.post_train_steps) // self.episode_length // self.n_rollout_threads
        pre_episodes = int(self.pre_train_steps) // self.episode_length // self.n_rollout_threads
        post_episodes = int(self.post_train_steps) // self.episode_length // self.n_rollout_threads

        log_out_dict = dict()

        info_logs = dict()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic

                if self.all_args.use_mix_policy:
                    all_done = np.all(dones, axis=1)
                    done_indices = np.where(all_done)[0]
                    exp3_start = time.time()
                    if self.sub_ep >= self.exp3_interval/2:
                        sub_select = self.envs.world.oppo_policy.return_sub_select()
                        for i in done_indices:
                            self.total_done += 1
                            self.sub_G_line[sub_select[i]] += rewards[i][0][0]
                    exp3_end = time.time()
                    delta_exp3 = exp3_end - exp3_start
                    exp3_time += delta_exp3

                    self.envs.world.oppo_policy.update_index_multi_channels(done_indices)


                self.insert(data)
                for info in infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode >= pre_episodes:
                if (episode + 1) % self.exp3_interval == 0 and episode < episodes - post_episodes:
                    exp3_start = time.time()
                    self.warmup()
                    self.sub_G_line /= self.total_done
                    self.sub_G_line /= self.probs
                    self.sub_G_line *= self.multi_round
                    self.regret_value += np.max(self.sub_G_line) - np.dot(self.sub_G_line, self.probs)
                    self.G_history_line += self.sub_G_line

                    if len(self.G_avg_array[0]) >= self.avg_G_last_N:
                        for item_i in range(self.support_K):
                            self.G_avg_array[item_i].pop(0)

                    for item_i in range(self.support_K):
                        self.G_avg_array[item_i].append(self.sub_G_line[item_i])
                        self.G_avg[item_i] = np.mean(self.G_avg_array[item_i])

                    self.G_his_match += np.dot(self.sub_G_line, self.probs)
                    self.real_T = episode
                    self.exp3_update()

                    self.exp3_round += self.multi_round
                    self.sub_G_line = np.zeros(self.support_K)
                    self.total_done = 0
                    self.sub_ep = 0
                    exp3_end = time.time()
                    delta_exp3 = exp3_end - exp3_start
                    exp3_time += delta_exp3
                else:
                    self.sub_ep += 1

            if episode % self.log_interval == 0:


                policy_head = "Estimator_" + str(self.policy_inx) + "_"
                train_final_info = dict()
                for k in train_infos.keys():
                    train_final_info[policy_head + k] = _t2n(train_infos[k]) if (isinstance(train_infos[k], torch.Tensor) and train_infos[k].is_cuda) else train_infos[k]
                for k in info_logs.keys():
                    train_final_info[policy_head + k] = info_logs[k] / self.n_rollout_threads
                info_logs = dict()
                train_final_info[policy_head + "average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                log_out_dict[self.all_args.global_steps + total_num_steps] = train_final_info


            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        self.RM_time = exp3_time

        return log_out_dict

    def set_policy_n_prob(self, inx, probs):
        self.policy_inx = inx
        self.envs.world.oppo_policy.set_probs_all(probs)
        self.support_K = len(probs)
        self.probs = probs
        self.yita_ini = self.exp_yita_coef * 0.95*np.sqrt(np.log(self.support_K)/self.support_K)
        self.gamma_ini = 1.05*np.sqrt(self.support_K * np.log(self.support_K))
        self.beta_ini = np.sqrt(np.log(self.support_K)/self.support_K)
        self.exp3_round = 1
        self.multi_round = 1
        self.exp3_interval = int(self.multi_round * self.all_args.RM_interval)

    def warmup(self):
        obs, available_actions = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    def exp3_update(self):
        g_ = np.max(self.G_history_line) - self.G_history_line
        yita_ = self.yita_ini / np.sqrt(self.exp3_round)
        regret_ = self.regret_value
        gamma_ = min(1.0, np.sqrt(self.support_K * np.log(self.support_K))/((np.exp(1)-1)*self.exp3_round))
        probs_eq = np.ones(self.support_K) / np.sum(np.ones(self.support_K))
        self.probs = (1-gamma_) * np.exp(yita_ * g_) /np.sum(np.exp(yita_ * g_)) + gamma_ * probs_eq
        self.envs.world.oppo_policy.set_probs_all(self.probs)
        print("exp3_round = {}, total_regret = {}, probs = {}".format(self.exp3_round, regret_, self.probs))


    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = np.concatenate([actions[:, idx, :] for idx in range(self.num_agents)], axis=1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if "TimeLimit.truncated" in info else [1.0] for _ in range(self.num_agents)] for info in infos])

        if self.use_centralized_V:
            share_obs = obs
        else:
            share_obs = obs
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks=bad_masks, active_masks=active_masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        pass

    def transfer_model_to(self, device):
        self.device = device
        self.trainer.transfer_model_to(device)
        if self.envs.world.oppo_policy is not None:
            self.envs.world.oppo_policy.transfer_model_to(device)


    @torch.no_grad()
    def calc_win_prob(self, total_episodes, deterministic = True):
        eval_obs, eval_a_actions = self.envs.reset()
        self.total_round = 0
        self.total_N_array = np.zeros(total_episodes)
        self.total_reward = 0
        self.eva_r_list = []
        eval_rnn_states = np.zeros((self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    np.concatenate(eval_a_actions),
                                                    deterministic=deterministic)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.num_agents)], axis=1)

                eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_actions = self.envs.step(eval_actions_env)

                for i in range(self.n_rollout_threads):
                    self.total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        self.total_round += 1
                        self.total_N_array[episodes] += 1
                        self.eva_r_list.append(eval_rewards[i][0][0])
                        if self.all_args.use_mix_policy:
                            self.envs.world.oppo_policy.update_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)


    def get_payoff_sigma(self, total_episodes, delta = None, deterministic = True):
        eval_payoffs = 0

        print("eval_policy {}:".format(self.policy_inx))

        if delta is None:
            delta = np.inf

        total_reward_ = 0
        total_round_ = 0
        eval_r_list_ = []

        while True:
            self.calc_win_prob(total_episodes, deterministic)
            total_reward_ += self.total_reward
            total_round_ += self.total_round
            eval_r_list_ += copy.deepcopy(self.eva_r_list)
            payoff_p = (total_reward_)/(total_round_)
            std_ = np.std(np.array(eval_r_list_))/np.sqrt(len(eval_r_list_))
            print("standard value = {}, target = {}".format(std_, delta))
            if std_ < delta:
                break

        eval_payoffs = payoff_p
        return eval_payoffs, std_


    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor"  + ".pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic" + ".pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm" + ".pt")

    def save_as_filename(self, head_str):
        """Save policy's actor and critic networks."""
        label_str = head_str
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor_" +  label_str + ".pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_" + label_str + ".pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_" + label_str + ".pt")

    def inherit_policy(self, policy_str, head_str = None):
        if head_str is None:
            policy_actor_state_dict = torch.load(policy_str + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(policy_str + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
                if self.trainer._use_valuenorm:
                    policy_vnorm_state_dict = torch.load(policy_str + '/vnorm.pt')
                    self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

        else:
            policy_actor_state_dict = torch.load(policy_str  + '/actor_' + str(head_str) + '.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(policy_str + '/critic_' + str(head_str) + '.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
                if self.trainer._use_valuenorm:
                    policy_vnorm_state_dict = torch.load(policy_str + '/vnorm_'+ str(head_str) +'.pt')
                    self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_str = str(self.model_dir)
        self.inherit_policy(policy_str)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        raise NotImplementedError
