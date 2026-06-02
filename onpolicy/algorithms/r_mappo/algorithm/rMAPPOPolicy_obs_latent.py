import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_latent import R_Actor_sigma, R_Critic_sigma
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import check
import numpy as np
import copy

def kl_divergence(prob_a, prob_b, soft_mask = True):

    ori_shape = prob_b.shape
    prob_a = prob_a.reshape(-1)
    prob_b = prob_b.reshape(-1)
    if soft_mask:
        prob_b[prob_a<1e-5] = -1e10
    else:
        prob_b[prob_a<1e-5] = 0
    prob_b = prob_b.detach()
    prob_a = prob_a.reshape(ori_shape)
    prob_b = prob_b.reshape(ori_shape)

    if soft_mask:
        prob_b = torch.softmax(prob_b, -2)
    res = torch.zeros_like(prob_a)
    mask_a = prob_a>=1e-5
    res[mask_a] = (prob_a[mask_a] * (torch.log(prob_a[mask_a]/(prob_b[mask_a]+1e-5))))
    res = res.reshape(-1)
    prob_a = prob_a.reshape(-1)
    res = res.reshape(ori_shape)

    return res.sum(1)

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.sigma_tensor = None
        self.latent_size = args.latent_size
        self.sigma_fusion = True
        self.args = copy.deepcopy(args)
        self.use_soft_kl = args.use_soft_kl

        self.actor = R_Actor_sigma(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_sigma(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.lr, eps=self.opti_eps,
                                            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.critic_lr,
                                             eps=self.opti_eps,
                                             weight_decay=self.weight_decay)

    def set_sigma(self, sigma_tensor):
        sigma_tensor = check(sigma_tensor).to(**self.tpdv)
        self.sigma_tensor = sigma_tensor

    def set_fusion_true(self):
        self.sigma_fusion = True

    def set_fusion_false(self):
        self.sigma_fusion = False


    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.latent_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                rnn_states_actor,
                                                                masks,
                                                                sigma,
                                                                available_actions,
                                                                deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.sigma_fusion:
            cent_obs, sigma = np.split(cent_obs, [cent_obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.latent_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     sigma,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values, action_log_probs, dist_entropy

    def get_kl_divergence(self, obs, rnn_states_actor, action, masks,
                            anchor_policies = None, available_actions=None, active_masks=None):
        """
        This function is designed for mappo and rmappo is not supported!
        """
        assert self.act_space.__class__.__name__ == "Discrete", "The action space is not supported."
        max_n = self.act_space.n
        if anchor_policies is None:
            return

        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        probs_self = self.get_probs(obs, rnn_states_actor, sigma, max_n, masks, available_actions, active_masks)


        kl_divs = []

        for i in range(len(anchor_policies)):
            probs_anchor = anchor_policies[i].get_probs(obs, rnn_states_actor, sigma, max_n, masks, available_actions, active_masks)
            kl_div = kl_divergence(probs_self, probs_anchor, self.use_soft_kl)
            kl_divs.append(kl_div)

        kl_divs = torch.stack(kl_divs, dim=0)


        return kl_divs

    def get_kl_divergence_with_idx(self, obs, rnn_states_actor, action, masks,
                            anchor_policies = None, select_inx = None, available_actions=None, active_masks=None):
        """
        This function is designed for mappo and rmappo is not supported!
        """
        assert self.act_space.__class__.__name__ == "Discrete", "The action space is not supported."
        max_n = self.act_space.n
        if anchor_policies is None:
            return

        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        probs_self = self.get_probs(obs, rnn_states_actor, sigma, max_n, masks, available_actions, active_masks)


        kl_divs = []

        for i in range(len(anchor_policies)):
            probs_anchor = anchor_policies[i].get_probs(obs, rnn_states_actor, sigma, max_n, masks, available_actions, active_masks)
            kl_div = kl_divergence(probs_self, probs_anchor, self.use_soft_kl)
            kl_divs.append(kl_div)

        kl_divs = torch.stack(kl_divs, dim=0)

        indices_t = torch.from_numpy(select_inx).long().to(kl_divs.device)

        out = kl_divs[indices_t.detach(), torch.arange(kl_divs.shape[1], device=kl_divs.device)]

        out = out.unsqueeze(0)


        return out

    def get_probs(self, obs, rnn_states, sigma, max_n, masks, available_actions=None, active_masks=None):

        probs_all = self.actor.get_probs(obs, rnn_states, sigma, max_n, masks, available_actions, active_masks)

        return probs_all

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.latent_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, sigma, available_actions, deterministic)
        return actions, rnn_states_actor

    def reset_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.lr, eps=self.opti_eps,
                                            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.critic_lr,
                                             eps=self.opti_eps,
                                             weight_decay=self.weight_decay)

    def reset_act_layer(self):
        self.actor.reset_act_layer()
        self.reset_optimizer()

    def reset_policy(self):
        self.actor = R_Actor_sigma(self.args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_sigma(self.args, self.share_obs_space, self.device)

        self.reset_optimizer()

    def transfer_model_to(self, device):
        self.device = device
        self.actor.transfer_to(device)
        self.critic.transfer_to(device)
        self.reset_optimizer()
