import time
import numpy as np
import torch
import copy
from onpolicy.runner.base_runner import Runner
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_diversity import R_MAPPOPolicy as Policy
from onpolicy.algorithms.r_mappo.r_mappo_diversity import R_MAPPO as TrainAlgo

def _t2n(x):
    return x.detach().cpu().numpy()

class DIV_Runner(Runner):
    """Runner for diversity-regularized best-response policy training."""
    def __init__(self, config):
        super(DIV_Runner, self).__init__(config)
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        self.kl_div_coef = self.all_args.kl_div_coef


    def run(self):
        self.warmup()

        log_out_dict = dict()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_logs = dict()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, kl_divs = self.collect(step)
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, kl_divs

                if self.all_args.use_mix_policy:
                    all_done = np.all(dones, axis=1)
                    done_indices = np.where(all_done)[0]
                    self.envs.world.oppo_policy.update_index_multi_channels(done_indices)


                self.insert(data)
                for info in infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]

            kl_fl = self.buffer.kl_divs.reshape(*self.buffer.kl_divs.shape[:2], -1)
            kl_sum = np.sum(np.sum(kl_fl, axis = -1), axis = 0)
            min_policy_inx = np.argmin(kl_sum)
            self.buffer.rewards += self.kl_div_coef * self.buffer.kl_divs[:, min_policy_inx]
            self.trainer.update_anchor_policy([self.oppo_policies[min_policy_inx]])

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, policy {}, min_kl_index = {}, timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                self.policy_inx,
                                min_policy_inx,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                env_infos = {}

                policy_head = "policy_" + str(self.policy_inx) + "_"
                for k in info_logs.keys():
                    train_infos[policy_head + k] = info_logs[k] / self.n_rollout_threads
                info_logs = dict()
                train_infos[policy_head + "average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos[policy_head + "average_episode_rewards"]))
                log_out_dict[self.all_args.global_steps + total_num_steps] = train_infos


            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        return log_out_dict

    def set_policy_inx(self, inx):
        self.policy_inx = inx

    def transfer_model_to(self, device):
        self.device = device
        self.trainer.transfer_model_to(device)
        self.envs.world.oppo_policy.transfer_model_to(device)
        if hasattr(self, "oppo_policies"):
            for i in range(len(self.oppo_policies)):
                self.oppo_policies[i].transfer_model_to(device)

    def set_oppo_policies(self, oppo_policies):
        self.oppo_policies = oppo_policies
        self.buffer.kl_divs = np.zeros(
            (self.episode_length, len(oppo_policies), self.n_rollout_threads, 1, 1), dtype=np.float32)

    def warmup(self):
        obs, available_actions = self.envs.reset()
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
        kl_div = self.calc_kl_div(step, actions)
        kl_divs = np.array(_t2n(kl_div.unsqueeze(-1)))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = np.concatenate([actions[:, idx, :] for idx in range(self.num_agents)], axis=1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, kl_divs

    def calc_kl_div(self, step, actions):

        kl_divs = self.trainer.policy.get_kl_divergence(np.concatenate(self.buffer.obs[step]),
                                                        np.concatenate(self.buffer.rnn_states[step]),
                                                        np.concatenate(actions),
                                                        np.concatenate(self.buffer.masks[step]),
                                                        self.oppo_policies,
                                                        np.concatenate(self.buffer.available_actions[step]),
                                                        np.concatenate(self.buffer.active_masks[step]))

        return kl_divs

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, kl_divs = data

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
        self.buffer.kl_divs[self.buffer.step] = kl_divs.copy()
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks=bad_masks, active_masks=active_masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        pass

    @torch.no_grad()
    def calc_win_prob(self, total_episodes):
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
                                                    deterministic=True)
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

            print("episodes: {}/{}".format(episodes, total_episodes))

    def get_payoff_sigma(self, total_episodes):
        eval_payoffs = 0
        standard_vaules = 0

        print("eval_policy {}:".format(self.policy_inx))
        self.calc_win_prob(total_episodes)
        payoff_p = (self.total_reward)/(self.total_round)
        eval_payoffs = payoff_p
        standard_vaules = np.std(np.array(self.eva_r_list))/np.sqrt(len(self.eva_r_list))
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

    def save_as_filename(self, head_str, only_eval = False):
        """Save policy's actor and critic networks."""
        label_str = head_str
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor_" +  label_str + ".pt")
        if only_eval == False:
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

    def restore(self, head_str = None):
        """Restore policy's networks from a saved model."""
        policy_str = str(self.model_dir)
        self.inherit_policy(policy_str, head_str)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        raise NotImplementedError
