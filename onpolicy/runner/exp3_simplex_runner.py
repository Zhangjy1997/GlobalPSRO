import numpy as np
from gym import spaces

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_obs_latent import R_MAPPOPolicy as ObsLatentPolicy
from onpolicy.algorithms.r_mappo.r_mappo_sigma import R_MAPPO as TrainAlgo
from onpolicy.runner.exp3_sigma_simplex_runner import EXP3_Sigma_Simplex_Runner
from onpolicy.utils.shared_buffer import SharedReplayBuffer


class EXP3_Simplex_Runner(EXP3_Sigma_Simplex_Runner):
    """EXP3 simplex runner backed by the obs-latent conditional policy."""

    def __init__(self, config):
        super(EXP3_Simplex_Runner, self).__init__(config)
        self.max_latent_size = self.all_args.latent_size

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )
        shape_obs = self.envs.observation_space[0].shape[-1]
        obs_fusion = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(shape_obs + self.all_args.latent_size,),
            dtype=np.float32,
        )
        shape_cent_obs = share_observation_space.shape[-1]
        cent_obs_fusion = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(shape_cent_obs + self.all_args.latent_size,),
            dtype=np.float32,
        )

        self.policy = ObsLatentPolicy(
            self.all_args,
            obs_fusion,
            cent_obs_fusion,
            self.envs.action_space[0],
            device=self.device,
        )
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            obs_fusion,
            cent_obs_fusion,
            self.envs.action_space[0],
        )
