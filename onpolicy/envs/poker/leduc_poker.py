import gym
import torch
import numpy as np
import random

class Poker(gym.Env):
    def __init__(self, world):

        self.world = world
        self.action_space = self.world.action_space
        self.observation_space = self.world.observation_space
        self.share_observation_space = self.world.observation_space
        self.episode_length = self.world.episode_length

    def step(self, action_n):
        obs_n, reward_n, done_n, info_n, a_ac_n = self.world.step(action_n)
        return obs_n, reward_n, done_n, info_n, a_ac_n

    def reset(self):
        obs_n, a_ac_n = self.world.reset()
        return obs_n, a_ac_n
    
    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)