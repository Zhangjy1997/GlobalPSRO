import copy

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_sigma import R_Actor_sigma, R_Critic_sigma
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """Sigma-conditioned policy used by the simplex estimator bootstrap path."""

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
        self.population_size = args.population_size
        self.sigma_fusion = True
        self.args = copy.deepcopy(args)

        self.actor = R_Actor_sigma(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_sigma(args, self.share_obs_space, self.device)
        self.reset_optimizer()

    def set_sigma(self, sigma_tensor):
        sigma_tensor = check(sigma_tensor).to(**self.tpdv)
        self.sigma_tensor = sigma_tensor

    def set_fusion_true(self):
        self.sigma_fusion = True

    def set_fusion_false(self):
        self.sigma_fusion = False

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            sigma,
            available_actions,
            deterministic,
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        if self.sigma_fusion:
            cent_obs, sigma = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            sigma,
            available_actions,
            active_masks,
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, sigma, available_actions, deterministic)
        return actions, rnn_states_actor

    def reset_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def reset_act_layer(self):
        self.actor.reset_act_layer()
        self.reset_optimizer()

    def reset_policy(self):
        self.actor = R_Actor_sigma(self.args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_sigma(self.args, self.share_obs_space, self.device)
        self.reset_optimizer()

    def transfer_model_to(self, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.actor.transfer_to(device)
        self.critic.transfer_to(device)
        self.reset_optimizer()
