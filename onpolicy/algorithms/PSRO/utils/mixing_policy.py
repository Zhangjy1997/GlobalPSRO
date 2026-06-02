import torch
import numpy as np
import copy
class Mixing_policy:
    def __init__(self, policies):
        self.policy_num = len(policies)
        self.policy_list  = policies
        self.probs = np.random.rand(self.policy_num)
        self.probs = self.probs / self.probs.sum()
        self.selected_index = np.random.choice(self.policy_num, p=self.probs)
        self.history_index = np.zeros(self.policy_num)

    def set_probs(self, probs):
        probs[probs<0] = 0
        self.probs = probs
        self.probs /= sum(self.probs)
        self.history_index = np.zeros(self.policy_num)
        self.update_selected_index()

    def update_selected_index(self):
        self.selected_index = np.random.choice(self.policy_num, p=self.probs)
        self.history_index[self.selected_index] += 1

    def get_selected_index(self):
        return self.selected_index
    
    def act(self, obs, rnn_state, rnn_mask, deterministic = True):
        nxt_action, nxt_rnn_states = self.policy_list[self.selected_index].act(obs, rnn_state, rnn_mask, deterministic = deterministic)
        return nxt_action, nxt_rnn_states

# bridge class
class Parallel_mixing_policy:
    def __init__(self, num_threads, policies, probs = None, device = torch.device("cpu")):
        self.num_threads = num_threads
        self.policies = policies
        self.actor = policies[0].actor
        self.sub_select = np.empty(self.num_threads, dtype=int)
        self.sort_line = np.argsort(self.sub_select)
        self.anti_sort_line = np.empty_like(self.sort_line)
        self.num_policy = len(policies)
        self.tpdv = dict(dtype=torch.float32, device=device)
        for i in range(len(policies)):
            self.policies[i].actor.eval()
        
        self.mp_list = []
        for i in range(self.num_threads):
            self.mp_list.append(Mixing_policy(self.policies))

        if probs is not None:
            self.probs = probs
            self.set_probs_all(probs)
        else:
            self.get_sub_select()

    def set_sort_line(self):
        self.sort_line = np.argsort(self.sub_select)
        self.anti_sort_line[self.sort_line] = np.arange(len(self.sort_line))

    def transfer_model_to(self, device):
        self.tpdv = dict(dtype=torch.float32, device=device)
        for policy in self.policies:
            policy.transfer_model_to(device)

    def get_sub_select(self):
        for i in range(self.num_threads):
            self.sub_select[i] = self.mp_list[i].get_selected_index()

        self.num_count = np.bincount(self.sub_select)
        count_len = len(self.num_count)
        if count_len < self.num_policy:
            self.num_count = np.pad(self.num_count ,(0,self.num_policy - count_len),'constant')
        self.set_sort_line()

    def return_sub_select(self):
        return copy.deepcopy(self.sub_select)

    def expand_sort_line(self, group_size):
        offsets = np.arange(group_size)
        all_indices = self.sort_line[:, None] * group_size + offsets
        expand_indices = all_indices.flatten()

        inver_all_inx = np.empty_like(expand_indices)
        inver_all_inx[expand_indices] = np.arange(len(expand_indices))

        return expand_indices, inver_all_inx
        
    def set_probs_all(self, probs):
        self.probs = probs.copy()
        for i in range(self.num_threads):
            self.mp_list[i].set_probs(probs)
        self.get_sub_select()

    def set_probs_channel(self, probs, i):
        self.mp_list[i].set_probs(probs)
        self.get_sub_select()

    def set_probs_mat(self, probs_mat):
        for i in range(self.num_threads):
            self.mp_list[i].set_probs(probs_mat[i])
        self.get_sub_select()

    def set_probs_multi_channel(self, probs_mat, inx):
        for i in inx:
            self.mp_list[i].set_probs(probs_mat[i])
        self.get_sub_select()

    def update_index_all(self):
        for i in range(self.num_threads):
            self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def update_index_channel(self, i):
        self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def update_index_multi_channels(self, inx):
        for i in inx:
            self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def act(self, obs, rnn_state, rnn_mask, available_actions=None, deterministic = True):
        group_size = len(obs) // self.num_threads
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        obs_channels = obs[ex_sort]
        rnn_state_channels = rnn_state[ex_sort]
        rnn_mask_channels = rnn_mask[ex_sort]
        # obs_channels = check(obs_channels).to(**self.tpdv)
        # rnn_state_channels = check(rnn_state_channels).to(**self.tpdv)
        # rnn_mask_channels = check(rnn_mask_channels).to(**self.tpdv)
        if available_actions is not None:
            available_actions_channels = available_actions[ex_sort]
            # available_actions_channels = check(available_actions_channels).to(**self.tpdv)
            use_a_acts = True
        else:
            use_a_acts = False
        actions_ch = []
        rnn_state_ch = []
        inx_data = 0
        #print(self.num_count)
        for i in range(len(self.policies)):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                action_temp, rnn_state_temp = self.policies[i].act(obs_channels[indices_data], 
                                                                   rnn_state_channels[indices_data], 
                                                                   rnn_mask_channels[indices_data], 
                                                                   available_actions=available_actions_channels[indices_data] if use_a_acts else None, 
                                                                   deterministic = deterministic)
                actions_ch.append(action_temp)
                rnn_state_ch.append(rnn_state_temp)
                inx_data +=group_size * self.num_count[i]
        action_out = torch.cat(actions_ch, dim=0)
        rnn_state_out = torch.cat(rnn_state_ch, dim=0)
        action_out = action_out[inver_sort]
        rnn_state_out = rnn_state_out[inver_sort]
        return action_out, rnn_state_out
