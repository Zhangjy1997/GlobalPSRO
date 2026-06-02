import time
import numpy as np
import torch
import copy
from onpolicy.runner.base_runner import Runner
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_obs_latent import R_MAPPOPolicy as Policy_sigma
from onpolicy.algorithms.r_mappo.r_mappo_sigma import R_MAPPO as TrainAlgo
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from gym import spaces

def _t2n(x):
    return x.detach().cpu().numpy()

class BR_Simplex_Runner(Runner):
    """Runner for simplex-conditioned best-response policy training."""
    def __init__(self, config):
        super(BR_Simplex_Runner, self).__init__(config)
        self.simplex_eps = self.all_args.simplex_eps
        self.full_policy_size = self.all_args.latent_size
        self.use_uniform_simplex = self.all_args.use_uniform_simplex
        self.exp3_interval = getattr(self.all_args, "RM_interval", 10)
        self.pre_train_steps = getattr(self.all_args, "RM_pre_train_steps", 0)
        self.post_train_steps = getattr(self.all_args, "RM_post_train_steps", 0)
        self.avg_G_last_N = getattr(self.all_args, "avg_G_last_N", 10)
        self.exp_yita_coef = getattr(self.all_args, "RM_yita_coef", 1.0)
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        shape_obs = self.envs.observation_space[0].shape
        shape_obs = shape_obs[-1]
        obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=((shape_obs + self.all_args.latent_size),), dtype=np.float32)
        shape_cent_obs = share_observation_space.shape
        shape_cent_obs = shape_cent_obs[-1]
        cent_obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=(shape_cent_obs + self.all_args.latent_size,), dtype=np.float32)

        self.policy = Policy_sigma(self.all_args,
                            obs_fusion,
                            cent_obs_fusion,
                            self.envs.action_space[0],
                            device = self.device)

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        obs_fusion,
                                        cent_obs_fusion,
                                        self.envs.action_space[0])


    def run(self):
        self.warmup()
        self.trainer.policy.set_fusion_true()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print_interval = max(1, episodes // 10)

        info_logs = dict()

        if self.use_anytime:
            self.G_history_line = np.zeros(self.support_K)
            self.G_his_match = 0
            self.total_done = 0
            self.sub_ep = 0
            self.sub_G_line = np.zeros(self.support_K)
            self.regret_value = 1e-4
            self.G_avg_array = [[] for _ in range(self.support_K)]
            self.G_avg = np.zeros(self.support_K)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                if self.all_args.use_mix_policy:
                    all_done = np.all(dones, axis=1)
                    done_indices = np.where(all_done)[0]
                    done_size = len(done_indices)
                    sub_select = self.envs.world.oppo_policy.return_sub_select()

                    if done_size > 0:
                        leader_probs = np.array(self.leader_probs, dtype=np.float32)
                        leader_probs /= leader_probs.sum()

                        for idx in done_indices:
                            if self.use_anytime and self.selected_inx[idx] == self.anytime_inx:
                                self.total_done += 1
                                self.sub_G_line[sub_select[idx]] += rewards[idx][0][0]
                            if np.random.rand() < self.simplex_eps:
                                if self.use_uniform_simplex:
                                    self.probs_mat[idx] = np.random.dirichlet(self.D_alpha)
                                    self.selected_inx[idx] = 0
                                else:
                                    rand_idx = np.random.randint(self.random_mat.shape[0])
                                    self.probs_mat[idx] = self.random_mat[rand_idx]
                                    self.selected_inx[idx] = rand_idx
                            else:
                                self.probs_mat[idx] = leader_probs
                                self.selected_inx[idx] = 1

                    self.envs.world.oppo_policy.set_probs_multi_channel(self.probs_mat, done_indices)


                n_envs, old_size = self.probs_mat.shape
                _, n_agents, obs_dim = obs.shape
                full_size = self.full_policy_size

                new_obs = np.empty((n_envs, n_agents, obs_dim + full_size), dtype=obs.dtype)

                new_obs[..., :obs_dim] = obs

                new_obs[..., obs_dim:] = 0

                new_obs[..., obs_dim:obs_dim + old_size] = self.probs_mat[:, None, :]

                obs = new_obs

                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic

                self.insert(data)
                for info in infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]


            self.compute()
            train_infos = self.train()


            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if self.use_anytime:
                if (episode + 1) % self.exp3_interval == 0:
                    if self.total_done > 0:
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
                    self.warmup()

                    self.exp3_round += self.multi_round
                    self.sub_G_line = np.zeros(self.support_K)
                    self.total_done = 0
                    self.sub_ep = 0
                else:
                    self.sub_ep += 1

            if episode % self.log_interval == 0:
                end = time.time()
                if episode % print_interval == 0:
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, num timesteps {}/{}, FPS {}.\n"
                            .format(self.all_args.scenario_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start))))

                env_infos = {}

                for k in info_logs.keys():
                    train_infos[k] = info_logs[k] / self.n_rollout_threads
                info_logs = dict()
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                if episode % print_interval == 0:
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))


            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def set_policy_sigma(self, sigma):
        self.trainer.set_policy_sigma(sigma)


    def transfer_model_to(self, device):
        self.device = device
        self.trainer.transfer_model_to(device)
        self.envs.world.oppo_policy.transfer_model_to(device)

    def exp3_update(self):
        g_ = np.max(self.G_history_line) - self.G_history_line
        yita_ = self.yita_ini / np.sqrt(self.exp3_round)
        regret_ = self.regret_value
        gamma_ = min(1.0, np.sqrt(self.support_K * np.log(self.support_K))/((np.exp(1)-1)*self.exp3_round))
        probs_eq = np.ones(self.support_K) / np.sum(np.ones(self.support_K))
        self.probs = (1-gamma_) * np.exp(yita_ * g_) /np.sum(np.exp(yita_ * g_)) + gamma_ * probs_eq
        self.random_mat[self.anytime_inx] = self.probs.copy()
        print("exp3_round = {}, total_regret = {}, probs = {}".format(self.exp3_round, regret_, self.probs))

    def set_policy_size(self, size_N, random_mat, leader_probs=None, anytime_inx = None):
        self.oppo_policy_size = size_N
        self.random_mat = random_mat
        self.D_alpha = np.ones(size_N)
        self.selected_inx = np.zeros(self.n_rollout_threads, dtype=int)
        if anytime_inx is None:
            self.use_anytime = False
        else:
            self.use_anytime = True
            self.anytime_inx = anytime_inx

        if self.use_anytime:
            self.support_K = len(random_mat[0])
            self.probs = self.random_mat[anytime_inx].copy()
            self.yita_ini = self.exp_yita_coef * 0.95*np.sqrt(np.log(self.support_K)/self.support_K)
            self.gamma_ini = 1.05*np.sqrt(self.support_K * np.log(self.support_K))
            self.beta_ini = np.sqrt(np.log(self.support_K)/self.support_K)
            self.exp3_round = 1
            self.multi_round = 1
            self.exp3_interval = int(self.multi_round * getattr(self.all_args, "RM_interval", 10))

        if leader_probs is None:
            raise ValueError("leader_probs must be provided.")
        leader_probs = np.array(leader_probs, dtype=np.float32)

        leader_probs /= leader_probs.sum()

        probs_mat = np.zeros((self.n_rollout_threads, size_N), dtype=np.float32)

        for i in range(self.n_rollout_threads):
            if np.random.rand() < self.simplex_eps:
                if self.use_uniform_simplex:
                    probs_mat[i] = np.random.dirichlet(self.D_alpha)
                    self.selected_inx[i] = 0
                else:
                    rand_idx = np.random.randint(self.random_mat.shape[0])
                    probs_mat[i] = self.random_mat[rand_idx]
                    self.selected_inx[i] = rand_idx
            else:
                probs_mat[i] = leader_probs
                self.selected_inx[i] = 1

        self.leader_probs = leader_probs
        self.probs_mat = probs_mat
        self.envs.world.oppo_policy.set_probs_mat(self.probs_mat)


    def warmup(self):
        obs, available_actions = self.envs.reset()
        n_envs, old_size = self.probs_mat.shape
        full_size = self.full_policy_size

        padded_probs = np.zeros((n_envs, full_size), dtype=self.probs_mat.dtype)
        padded_probs[:, :old_size] = self.probs_mat

        expanded_probs = np.repeat(padded_probs[:, np.newaxis, :], obs.shape[1], axis=1)
        obs = np.concatenate((obs, expanded_probs), axis=-1)
        if self.use_centralized_V:
            share_obs = obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

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

    @torch.no_grad()
    def calc_win_prob(self, total_episodes):
        eval_obs, eval_a_acts = self.envs.reset()
        self.total_round = 0
        self.total_N_array = np.zeros(total_episodes)
        self.total_reward = 0
        self.eva_r_list = []
        self.trainer.policy.set_fusion_false()
        eval_rnn_states = np.zeros((self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    np.concatenate(eval_a_acts),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.num_agents)], axis=1)

                eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)

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


    def get_payoff_sigma(self, total_episodes, delta = None):
        eval_payoffs = dict()
        standard_vaules = dict()
        for idx, row in zip(range(len(self.random_mat)), self.random_mat):
            padded_probs = np.zeros(self.full_policy_size, dtype=self.probs_mat.dtype)
            padded_probs[:len(row)] = row
            self.set_policy_sigma(np.tile(padded_probs,(1,1)))
            self.envs.world.oppo_policy.set_probs_all(row)
            print("eval_policy {}:".format(idx))
            if delta is None:
                delta = np.inf

            total_reward_ = 0
            total_round_ = 0
            eval_r_list_ = []

            while True:
                self.calc_win_prob(total_episodes)
                total_reward_ += self.total_reward
                total_round_ += self.total_round
                eval_r_list_ += copy.deepcopy(self.eva_r_list)
                payoff_p = (total_reward_)/(total_round_)
                std_ = np.std(np.array(eval_r_list_))/np.sqrt(len(eval_r_list_))
                print("standard value = {}, target = {}".format(std_, delta))
                if std_ < delta:
                    break

            eval_payoffs[idx] = payoff_p
            standard_vaules[idx] = std_
        return eval_payoffs, standard_vaules

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
