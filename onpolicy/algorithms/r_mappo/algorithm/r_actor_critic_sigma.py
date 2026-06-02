import torch
import torch.nn as nn

from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.encoder import Encoder
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import init, check
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor_sigma(nn.Module):
    """Sigma-conditioned actor used for simplex runner initialization parity."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_sigma, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        self.act_space = action_space
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.population_size = args.population_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        sigma_encoder_layer_N = getattr(args, "sigma_encoder_layer_N", 1)
        sigma_layer_N = getattr(args, "sigma_layer_N", 1)
        self.sigma_encoder = Encoder(
            self.population_size,
            self.hidden_size,
            sigma_encoder_layer_N,
            args.use_orthogonal,
            args.use_ReLU,
            args.use_feature_normalization,
        )
        self.add_sigma = Encoder(
            2 * self.hidden_size,
            self.hidden_size,
            sigma_layer_N,
            args.use_orthogonal,
            args.use_ReLU,
            args.use_feature_normalization,
        )
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def reset_act_layer(self):
        self.act = ACTLayer(self.act_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(self.device)

    def forward(self, obs, rnn_states, masks, sigma, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        sigma = check(sigma).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        sigma_full = torch.repeat_interleave(sigma, actor_features.size(0) // sigma.size(0), dim=0)
        sigma_features = self.sigma_encoder(sigma_full)
        soft_features = self.add_sigma(torch.cat([actor_features, sigma_features], dim=-1))
        actions, action_log_probs = self.act(soft_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, sigma, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        sigma = check(sigma).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        sigma_full = torch.repeat_interleave(sigma, actor_features.size(0) // sigma.size(0), dim=0)
        sigma_features = self.sigma_encoder(sigma_full)
        soft_features = self.add_sigma(torch.cat([actor_features, sigma_features], dim=-1))

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            soft_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy

    def transfer_to(self, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(self.device)


class R_Critic_sigma(nn.Module):
    """Sigma-conditioned critic used for simplex runner initialization parity."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_sigma, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.population_size = args.population_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        sigma_encoder_layer_N = getattr(args, "sigma_encoder_layer_N", 1)
        sigma_layer_N = getattr(args, "sigma_layer_N", 1)
        self.sigma_encoder = Encoder(
            self.population_size,
            self.hidden_size,
            sigma_encoder_layer_N,
            args.use_orthogonal,
            args.use_ReLU,
            args.use_feature_normalization,
        )
        self.add_sigma = Encoder(
            2 * self.hidden_size,
            self.hidden_size,
            sigma_layer_N,
            args.use_orthogonal,
            args.use_ReLU,
            args.use_feature_normalization,
        )

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, sigma):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        sigma = check(sigma).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        sigma_full = torch.repeat_interleave(sigma, critic_features.size(0) // sigma.size(0), dim=0)
        sigma_features = self.sigma_encoder(sigma_full)
        soft_features = self.add_sigma(torch.cat([critic_features, sigma_features], dim=-1))
        values = self.v_out(soft_features)

        return values, rnn_states

    def transfer_to(self, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)
