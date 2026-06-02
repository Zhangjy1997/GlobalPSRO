import time
import numpy as np
import torch
import copy
from onpolicy.runner.base_runner import Runner
from pathlib import Path
import os
import tempfile

def _t2n(x):
    return x.detach().cpu().numpy()

def atomic_torch_save(obj, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(final_path.parent),
        prefix=final_path.name + ".tmp.",
        suffix=".pt",
    )
    os.close(fd)

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

class BR_Runner(Runner):
    """Runner class for training BR policies against a mixed opponent policy."""
    def __init__(self, config):
        super(BR_Runner, self).__init__(config)
        self.all_args = copy.deepcopy(self.all_args)


    def run(self):
        self.warmup()
        self.save_as_filename_atom("active_policy_" + str(self.policy_inx))

        log_out_dict = dict()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        sub_episodes = episodes // 10

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
                    self.envs.world.oppo_policy.update_index_multi_channels(done_indices)


                self.insert(data)
                for info in infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save_as_filename_atom("active_policy_" + str(self.policy_inx))

            if episode % self.log_interval == 0:
                end = time.time()
                if episode % sub_episodes == 0:
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, policy {}, num timesteps {}/{}, FPS {}.\n"
                            .format(self.all_args.scenario_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    self.policy_inx,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start))))

                policy_head = "policy_" + str(self.policy_inx) + "_"
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

        return log_out_dict

    def set_policy_inx(self, inx):
        self.policy_inx = inx
        self.save_as_filename_atom("active_policy_" + str(self.policy_inx))

    def transfer_model_to(self, device):
        self.device = device
        self.trainer.transfer_model_to(device)
        self.envs.world.oppo_policy.transfer_model_to(device)

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


    def get_payoff_sigma(self, total_episodes, delta = None):
        eval_payoffs = 0

        print("eval_policy {}:".format(self.policy_inx))

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

        eval_payoffs = payoff_p
        return eval_payoffs, std_


    def save(self, only_eval = False):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor"  + ".pt")
        if only_eval == False:
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

    def save_as_filename_atom(self, head_str, only_eval=False):
        """Save policy's actor and critic networks safely."""
        label_str = str(head_str)
        save_dir = Path(self.save_dir)

        policy_actor = self.trainer.policy.actor
        atomic_torch_save(policy_actor.state_dict(), save_dir / f"actor_{label_str}.pt")

        if only_eval is False:
            policy_critic = self.trainer.policy.critic
            atomic_torch_save(policy_critic.state_dict(), save_dir / f"critic_{label_str}.pt")

            if self.trainer._use_valuenorm:
                policy_vnorm = self.trainer.value_normalizer
                atomic_torch_save(policy_vnorm.state_dict(), save_dir / f"vnorm_{label_str}.pt")


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
