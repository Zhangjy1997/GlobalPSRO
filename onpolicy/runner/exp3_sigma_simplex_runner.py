import time
import numpy as np
import torch
import copy
from gym import spaces

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as SigmaPolicy
from onpolicy.algorithms.r_mappo.r_mappo_sigma import R_MAPPO as TrainAlgo
from onpolicy.runner.base_runner import Runner
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


def _ensure_sigma_policy_defaults(args):
    defaults = {
        "sigma_layer_N": 1,
        "sigma_encoder_layer_N": 1,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


class EXP3_Sigma_Simplex_Runner(Runner):
    """EXP3 simplex runner that initializes the sigma-conditioned policy path."""

    def __init__(self, config):
        super(EXP3_Sigma_Simplex_Runner, self).__init__(config)
        _ensure_sigma_policy_defaults(self.all_args)
        self.simplex_eps = self.all_args.simplex_eps
        self.max_latent_size = self.all_args.population_size
        self.use_uniform_simplex = self.all_args.use_uniform_simplex
        self.exp3_interval = self.all_args.RM_interval
        self.pre_train_steps = self.all_args.RM_pre_train_steps
        self.post_train_steps = self.all_args.RM_post_train_steps
        self.avg_G_last_N = self.all_args.avg_G_last_N
        self.exp_yita_coef = self.all_args.RM_yita_coef
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        self.policy = SigmaPolicy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
        )
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        shape_obs = self.envs.observation_space[0].shape[-1]
        obs_fusion = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(shape_obs + self.all_args.population_size,),
            dtype=np.float32,
        )
        shape_cent_obs = share_observation_space.shape[-1]
        cent_obs_fusion = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(shape_cent_obs + self.all_args.population_size,),
            dtype=np.float32,
        )
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            obs_fusion,
            cent_obs_fusion,
            self.envs.action_space[0],
        )


    def run(self):
        self.warmup()
        self.trainer.policy.set_fusion_true()

        self.G_history_line = np.zeros((self.can_policy_size, self.support_K))
        self.total_done = np.zeros(self.can_policy_size)
        self.sub_ep = 0
        self.sub_G_line = np.zeros((self.can_policy_size, self.support_K))
        self.regret_value = 1e-4 * np.ones(self.can_policy_size)
        self.G_avg_array = [ [[] for _ in range(self.support_K)] for _ in range(self.can_policy_size) ]
        self.G_avg = np.zeros((self.can_policy_size, self.support_K))
        self.G_his_match = np.zeros(self.can_policy_size)

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_logs = dict()

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

                    if done_size > 0:

                        if self.sub_ep >= self.exp3_interval/2:
                            sub_select = self.envs.world.oppo_policy.return_sub_select()
                            for i in done_indices:
                                self.total_done[self.select_inx[i]] += 1
                                self.sub_G_line[self.select_inx[i]][sub_select[i] if sub_select[i] < self.support_K else self.support_K -1 ] += rewards[i][0][0]

                        for idx in done_indices:
                            if np.random.rand() < self.simplex_eps:
                                rand_idx = np.random.randint(self.random_mat.shape[0])
                                self.latent_mat[idx] = self.random_mat[rand_idx]
                                self.probs_mat[idx][:self.fixed_policy_size] = self.probs_latent_mat[rand_idx][:-1]
                                self.probs_mat[idx][self.fixed_policy_size + rand_idx] = self.probs_latent_mat[rand_idx][-1]
                                self.select_inx[idx] = rand_idx
                            else:
                                self.latent_mat[idx] = self.random_mat[self.leader_inx]
                                self.probs_mat[idx][:self.fixed_policy_size] = self.probs_latent_mat[self.leader_inx][:-1]
                                self.probs_mat[idx][self.fixed_policy_size + self.leader_inx] = self.probs_latent_mat[self.leader_inx][-1]
                                self.select_inx[idx] = self.leader_inx

                    self.envs.world.oppo_policy.set_probs_multi_channel(self.probs_mat, done_indices)

                n_envs, old_size = self.latent_mat.shape
                full_size = self.max_latent_size

                padded_probs = np.zeros((n_envs, full_size), dtype=self.probs_mat.dtype)
                padded_probs[:, :old_size] = self.latent_mat

                expanded_probs = np.repeat(padded_probs[:, np.newaxis, :], obs.shape[1], axis=1)

                obs = np.concatenate((obs, expanded_probs), axis=-1)
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

            if (episode + 1) % self.exp3_interval == 0:
                for k in range(self.can_policy_size):
                    if self.total_done[k] != 0:
                        self.sub_G_line[k] /= self.total_done[k]
                    self.sub_G_line[k] /= self.probs_latent_mat[k]
                    self.sub_G_line[k] *= self.multi_round
                    self.regret_value[k] += np.max(self.sub_G_line[k]) - np.dot(self.sub_G_line[k], self.probs_latent_mat[k])
                    self.G_history_line[k] += self.sub_G_line[k]

                    if len(self.G_avg_array[0][0]) >= self.avg_G_last_N:
                        for item_i in range(self.support_K):
                            self.G_avg_array[k][item_i].pop(0)

                    for item_i in range(self.support_K):
                        self.G_avg_array[k][item_i].append(self.sub_G_line[k][item_i])
                        self.G_avg[k][item_i] = np.mean(self.G_avg_array[k][item_i])

                    self.G_his_match[k] += np.dot(self.sub_G_line[k], self.probs_latent_mat[k])
                    self.exp3_update(self.G_history_line[k], k)

                self.warmup()
                self.exp3_round += self.multi_round
                self.sub_G_line = np.zeros((self.can_policy_size, self.support_K))
                self.total_done = np.zeros(self.can_policy_size)
                self.sub_ep = 0
            else:
                self.sub_ep += 1

            if episode % self.log_interval == 0:
                end = time.time()

                env_infos = {}

                for k in info_logs.keys():
                    train_infos[k] = info_logs[k] / self.n_rollout_threads
                info_logs = dict()
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                self.log_train(train_infos, self.all_args.global_steps + total_num_steps)
                self.log_env(env_infos, self.all_args.global_steps + total_num_steps)


            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def set_policy_sigma(self, sigma):
        self.trainer.set_policy_sigma(sigma)


    def set_policy_size(self, fixed_size_N, random_latent_mat, probs_ini, leader_inx = None):
        self.fixed_policy_size = fixed_size_N
        self.can_policy_size = random_latent_mat.shape[0]
        self.full_policy_size = fixed_size_N + self.can_policy_size
        self.random_mat = random_latent_mat
        self.select_inx = np.ones(self.n_rollout_threads, dtype=int)

        if leader_inx is None:
            self.simplex_eps = 1.0
            self.leader_inx = 0
        else:
            self.leader_inx = leader_inx

        latent_mat = np.zeros((self.n_rollout_threads, random_latent_mat.shape[1]), dtype=np.float32)
        probs_mat = np.zeros((self.n_rollout_threads, self.full_policy_size), dtype=np.float32)
        probs_latent_mat = np.zeros((len(self.random_mat), self.fixed_policy_size + 1))
        for i in range(len(probs_latent_mat)):
            probs_latent_mat[i] = probs_ini

        for i in range(self.n_rollout_threads):
            if np.random.rand() < self.simplex_eps:
                rand_idx = np.random.randint(self.random_mat.shape[0])
                latent_mat[i] = self.random_mat[rand_idx]
                probs_mat[i][:self.fixed_policy_size] = probs_latent_mat[rand_idx][:-1]
                probs_mat[i][self.fixed_policy_size + rand_idx] = probs_latent_mat[rand_idx][-1]
                self.select_inx[i] = rand_idx
            else:
                latent_mat[i] = self.random_mat[leader_inx]
                probs_mat[i][:self.fixed_policy_size] = probs_latent_mat[leader_inx][:-1]
                probs_mat[i][self.fixed_policy_size + leader_inx] = probs_latent_mat[leader_inx][-1]
                self.select_inx[i] = leader_inx


        self.latent_mat = latent_mat
        self.probs_mat = probs_mat
        self.probs_latent_mat = probs_latent_mat
        self.envs.world.oppo_policy.set_probs_mat(self.probs_mat)

        self.support_K = self.fixed_policy_size + 1
        self.yita_ini = self.exp_yita_coef * 0.95*np.sqrt(np.log(self.support_K)/self.support_K)
        self.gamma_ini = 1.05*np.sqrt(self.support_K * np.log(self.support_K))
        self.beta_ini = np.sqrt(np.log(self.support_K)/self.support_K)
        self.exp3_round = 1
        self.multi_round = 1
        self.exp3_interval = int(self.multi_round * self.all_args.RM_interval)

    def exp3_update(self, G_line, inx):
        g_ = np.max(G_line) - G_line
        yita_ = self.yita_ini / np.sqrt(self.exp3_round)
        regret_ = self.regret_value[inx]
        gamma_ = min(1.0, np.sqrt(self.support_K * np.log(self.support_K))/((np.exp(1)-1)*self.exp3_round))
        probs_eq = np.ones(self.support_K) / np.sum(np.ones(self.support_K))
        self.probs_latent_mat[inx] = (1-gamma_) * np.exp(yita_ * g_) /np.sum(np.exp(yita_ * g_)) + gamma_ * probs_eq
        print("exp3_round = {}, inx = {}, total_regret = {}, probs = {}".format(self.exp3_round, inx, regret_, self.probs_latent_mat[inx]))


    def set_id_sigma(self, id_sigma, id_inx = None, flatten_id_sigma = None):
        self.id_sigma = id_sigma
        if id_inx is None:
            self.id_inx = np.arange(1,len(self.id_sigma)+1,1)
        else:
            self.id_inx = id_inx
        if flatten_id_sigma is None:
            self.flatten_id = copy.deepcopy(id_sigma)
        else:
            self.flatten_id = flatten_id_sigma

        self.probs_mat = np.zeros((self.n_rollout_threads,id_sigma.shape[-1]))
        self.flatten_probs_mat = np.zeros((self.n_rollout_threads, self.flatten_id.shape[-1]))
        self.porbs_inx = np.zeros(self.n_rollout_threads, dtype=int)
        for i in range(self.n_rollout_threads):
            random_inx = np.random.choice(id_sigma.shape[0])
            self.probs_mat[i] = id_sigma[random_inx]
            self.flatten_probs_mat[i] = self.flatten_id[random_inx]
            self.porbs_inx[i] = self.id_inx[random_inx]
        self.envs.world.oppo_policy.set_probs_mat(self.flatten_probs_mat)
        self.set_policy_sigma(self.probs_mat)

    def warmup(self):
        obs, available_actions = self.envs.reset()
        n_envs, old_size = self.latent_mat.shape
        full_size = self.max_latent_size

        padded_probs = np.zeros((n_envs, full_size), dtype=self.probs_mat.dtype)
        padded_probs[:, :old_size] = self.latent_mat

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

            print("episodes: {}/{}".format(episodes, total_episodes))


    def get_payoff_sigma(self, total_episodes, delta = None, low_i = -1):
        eval_payoffs = dict()
        standard_vaules = dict()
        for idx, row, probs_row in zip(self.id_inx, self.id_sigma, self.flatten_id):
            if idx < low_i:
                continue
            self.set_policy_sigma(np.tile(row,(1,1)))
            self.envs.world.oppo_policy.set_probs_all(probs_row)
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
