import numpy as np
import torch
import pyspiel
from gym import spaces

import copy

def _t2n(x):
    return x.detach().cpu().numpy()

class leduc_poker_symmetry:
    def __init__(self, num_threads, oppo_policy = None):
        self.standard_game = pyspiel.load_game("leduc_poker(players=2)")

        observation_length = self.standard_game.information_state_tensor_size()
        act_dim = self.standard_game.num_distinct_actions()

        self.num_threads = num_threads

        self.game_list = []
        for i in range(self.num_threads):
            self.game_list.append(pyspiel.load_game("leduc_poker(players=2)"))

        self.states = [None for _ in range(self.num_threads)]

        self.episode_length = (self.standard_game.max_game_length() // 2)
        print("env name = leduc poker!")
        print("episode_length = ", self.episode_length)

        self.role_flags = np.zeros(self.num_threads)

        self.dones = np.zeros((self.num_threads, 1), dtype=bool)
        self.rewards = np.zeros((self.num_threads, 1))
        self.obs = np.zeros((self.num_threads, observation_length))

        self.infos = []
        for i in range(self.num_threads):
            info = dict()
            self.infos.append(info)

        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.share_observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.action_space = [spaces.Discrete(act_dim) for _ in range(1)]

        self.obs_dim = observation_length
        self.act_dim = act_dim

        self.oppo_obs_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.oppo_act_space = [spaces.Discrete(act_dim) for _ in range(1)]

        if oppo_policy is None:
            self.oppo_policy = None
        else:
            self.oppo_policy = copy.deepcopy(oppo_policy)

    def reset(self):
        self.role_flags = np.random.randint(2, size=self.num_threads)
        self.states = []
        for i in range(self.num_threads):
            state = self.game_list[i].new_initial_state()
            self.states.append(state)

        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, 1, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)

        self.across_chance_node()
        cur_players = self.collect_current_player()
        mask_oppo_terminal =  np.equal(cur_players, 1 - self.role_flags)
        s_all, a_acts = self.collect_state()
        oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_terminal)
        self.apply_action_multi(oppo_action, mask_oppo_terminal)
        s_all, a_acts = self.collect_state()
        self.obs = s_all
        self.legal_act = a_acts
        
        self.obs = np.array(self.obs)
        self.legal_act = np.array(self.legal_act)

        # print("reset player = ", self.collect_current_player() - self.role_flags)

        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.legal_act[:, np.newaxis, :])
    
    def collect_state(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        s_all = np.zeros((self.num_threads, self.obs_dim))
        a_acts_all = np.zeros((self.num_threads, self.act_dim), dtype=int)
        for i in range(self.num_threads):
            if mask[i]:
                cur_player = self.states[i].current_player()
                s_all[i] = self.states[i].information_state_tensor(cur_player)
                legal_actions = self.states[i].legal_actions()
                a_acts_all[i] = self.legal2available(legal_actions)[0]
        return s_all, a_acts_all
    
    def across_chance_node(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        for i in range(self.num_threads):
            if mask[i]:
                while self.states[i].is_chance_node():
                    outcomes_with_probs = self.states[i].chance_outcomes()
                    action_list, prob_list = zip(*outcomes_with_probs)
                    action = np.random.choice(action_list, p=prob_list)
                    self.states[i].apply_action(action)

    def apply_action_multi(self, action_p, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        
        for i in range(self.num_threads):
            if mask[i]:
                self.states[i].apply_action(action_p[i])

    def collect_current_player(self):
        cur_players = np.zeros((self.num_threads), dtype=int)
        for i in range(self.num_threads):
            cur_players[i] = self.states[i].current_player()

        return cur_players
    
    def check_terminal_state(self):
        mask = np.zeros((self.num_threads), dtype=bool)
        for i in range(self.num_threads):
            mask[i] = self.states[i].is_terminal()

        return mask
    
    def get_rewards(self, mask):
        rewards = np.zeros((self.num_threads, 1))
        for i in range(self.num_threads):
            if mask[i]:
                reward_ = self.states[i].returns()
                rewards[i] = reward_[self.role_flags[i]]

        return rewards

    def update_obs_all(self, action_p):
        # print("step player = ", self.collect_current_player(), self.role_flags)
        # print("dones = ", self.dones)
        s_all, a_acts = self.collect_state()
        self.apply_action_multi(np.concatenate(action_p + 0.1).astype(int))
        
        cur_players = self.collect_current_player()
        # print("nxt player = ", cur_players , self.role_flags)
        mask_master_player = np.equal(cur_players, self.role_flags)
        mask_terminal = self.check_terminal_state()

        mask_ending = np.logical_or(mask_master_player, mask_terminal)

        while np.all(mask_ending) == False:
            self.across_chance_node()
            cur_players = self.collect_current_player()
            mask_player = cur_players >= 0
            mask_oppo_player = np.equal(cur_players, 1 - self.role_flags)
            s_all, a_acts = self.collect_state(mask_player)
            oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_player)
            self.apply_action_multi(oppo_action, mask_oppo_player)
            
            cur_players = self.collect_current_player()
            mask_master_player = np.equal(cur_players, self.role_flags)
            mask_terminal = self.check_terminal_state()
            mask_ending = np.logical_or(mask_master_player, mask_terminal)

        cur_players = self.collect_current_player()
        mask_player = cur_players >= 0
        self.dones = self.check_terminal_state()[:, np.newaxis]
        self.rewards = self.get_rewards(self.dones)
        s_all, a_acts = self.collect_state(mask_player)
        self.obs = s_all
        self.legal_act = a_acts

    def is_terminal_all(self):
        mask_terminal = copy.deepcopy(self.dones[:, 0])
        for i in range(self.num_threads):
            if mask_terminal[i]:
                self.role_flags[i] = np.random.randint(2)
                self.states[i] = self.game_list[i].new_initial_state()
        self.across_chance_node(mask_terminal)
        cur_players = self.collect_current_player()
        mask_oppo_player = np.equal(cur_players, 1 - self.role_flags)
        mask_oppo_terminal =  np.logical_and(mask_oppo_player, mask_terminal)
        s_all, a_acts = self.collect_state(mask_terminal)
        oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_terminal)
        self.apply_action_multi(oppo_action, mask_oppo_terminal)
        s_all, a_acts = self.collect_state(mask_terminal)
        self.obs[mask_terminal] = s_all[mask_terminal]
        self.legal_act[mask_terminal] = a_acts[mask_terminal]
            

    def step(self, actions):
        self.update_obs_all(actions)
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.dones == True] = np.zeros(((self.dones == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
            self.oppo_masks[self.dones == True] = np.zeros(((self.dones == True).sum(), 1), dtype=np.float32)
        self.is_terminal_all()
        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.rewards[:,:,np.newaxis]), copy.deepcopy(self.dones), copy.deepcopy(self.infos), copy.deepcopy(self.legal_act[:,np.newaxis,:])

    def legal2available(self, legal_action):
        available_actions = np.zeros(self.action_space[0].n, dtype=int)
        available_actions[legal_action] = 1
        available_actions = available_actions[np.newaxis, :]

        return available_actions
    
    @torch.no_grad()
    def get_oppo_action_multi(self, oppo_obs, available_actions, mask):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                np.concatenate(available_actions),
                                                deterministic=True)
        oppo_rnn_states_ = np.array(np.split(_t2n(oppo_rnn_states), self.num_threads))
        self.oppo_rnn_states[mask] = oppo_rnn_states_[mask]
        oppo_action = np.concatenate(_t2n(oppo_action) + 0.1)
        oppo_action = oppo_action.astype(int)

        return oppo_action
