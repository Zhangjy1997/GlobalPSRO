import numpy as np
import torch
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python import policy as openspiel_policy


def _t2n(x):
    return x.detach().cpu().numpy()

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def check_np(x):
    if isinstance(x, torch.Tensor):
        return _t2n(x)
    else:
        return x

def legal2available(n, legal_action):
    available_actions = np.zeros(n, dtype=int)
    available_actions[legal_action] = 1
    available_actions = available_actions[np.newaxis, :]

    return available_actions

class policy_like_spiel(openspiel_policy.Policy):
    def __init__(self, game, player_ids, policy_network, random_policy = False, batch_num = 100):
        super().__init__(game, player_ids)
        self.policy_network = policy_network
        self.act_dim = self.game.num_distinct_actions()
        self.obs_dim = self.game.information_state_tensor_size()
        self.random_policy = random_policy
        self.batch_num = batch_num
        self.tree_dict()

    def tree_dict(self):
        action_tree_dict = dict()
        root_state = self.game.new_initial_state()
        self.act_state_collection(root_state, action_tree_dict)
        if self.random_policy:
            self.action_tree_dict = action_tree_dict
        else:
            self.action_tree_dict = self.batch_dict(action_tree_dict)

    def batch_dict(self, state_tree_dict):
        action_tree_dict = dict()
        obs = np.zeros((self.batch_num, self.obs_dim))
        rnn_states = np.zeros((self.batch_num, self.policy_network.actor._recurrent_N, self.policy_network.actor.hidden_size), dtype=np.float32)
        masks = np.ones((self.batch_num, 1), dtype=np.float32)
        a_acts = np.zeros((self.batch_num, self.act_dim), dtype=int)
        sub_i = 0
        for s, state_ in state_tree_dict.items():
            focus_player = state_.current_player()
            s_t = state_.information_state_tensor(focus_player)
            obs[sub_i, :] = s_t
            legal_actions = state_.legal_actions()
            a_act_ = legal2available(self.act_dim, legal_actions)
            a_acts[sub_i, :] = a_act_[0]
            sub_i += 1

            if sub_i >= self.batch_num:
                act_states, _ = self.policy_network.act(obs, rnn_states, masks, available_actions=a_acts, deterministic=True)
                for k in range(self.batch_num):
                    s_ = tuple(obs[k])
                    act_max = act_states[k][0]
                    action_tree_dict[s_] = dict()
                    for i in range(self.act_dim):
                        if i == act_max:
                            action_tree_dict[s_][i] = 1.0
                        elif a_acts[k][i] > 0.5:
                            action_tree_dict[s_][i] = 0.0
                sub_i = 0

        if sub_i > 0:
            act_states, _ = self.policy_network.act(obs[:sub_i], rnn_states[:sub_i], masks[:sub_i], available_actions=a_acts[:sub_i], deterministic=True)
            for k in range(sub_i):
                s_ = tuple(obs[k])
                act_max = act_states[k][0]
                action_tree_dict[s_] = dict()
                for i in range(self.act_dim):
                    if i == act_max:
                        action_tree_dict[s_][i] = 1.0
                    elif a_acts[k][i] > 0.5:
                        action_tree_dict[s_][i] = 0.0

        return action_tree_dict


    def act_state_collection(self, state, action_dict):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                if prob > 0 :
                    new_state = state.child(action)
                    self.act_state_collection(new_state, action_dict)
            return
        
        focus_player = state.current_player()
        s = state.information_state_tensor(focus_player)
        legal_actions = state.legal_actions()
        s_t = tuple(s)
        if self.random_policy:
            action_dict[s_t] = dict()
            for i in range(self.act_dim):
                if i in legal_actions:
                    action_dict[s_t][i] = 1.0 / len(legal_actions)
        else:
            action_dict[s_t] = state

        for action in state.legal_actions():
            new_state = state.child(action)
            self.act_state_collection(new_state, action_dict)

        return

        



    def action_probabilities(self, state, player_id = None):
        if player_id is None:
            focus_player = state.current_player()
        else:
            focus_player = player_id
        s = state.information_state_tensor(focus_player)
        s_t = tuple(s)
        act_prob_dict = self.action_tree_dict[s_t]

        return act_prob_dict

class fake_actor:
    def __init__(self, actor):
        self._recurrent_N = actor._recurrent_N
        self.hidden_size = actor.hidden_size

    def eval(self):
        pass
    
class exec_mixed_policy(openspiel_policy.Policy):
    def __init__(self, game, player_ids, policy_networks, probs, deterministic_policy = None, random_policy = False, batch_num = 100, device = torch.device("cpu")):
        super().__init__(game, player_ids)
        self.policy_networks = policy_networks
        self.probs = probs
        self.actor = fake_actor(self.policy_networks[0].actor)
        assert len(self.policy_networks) == len(self.probs), "wrong policy numbers"
        if deterministic_policy is None:
            self.deter_mask = np.ones(len(self.probs), dtype=bool)
        else:
            self.deter_mask = deterministic_policy
        self.act_dim = self.game.num_distinct_actions()
        self.obs_dim = self.game.information_state_tensor_size()
        self.transfer_model_to(device)
        self.random_policy = random_policy
        self.batch_num = batch_num
        self.tree_dict()
        if hasattr(self, "policy_networks"):
            del self.policy_networks

    def tree_dict(self):
        action_tree_dict = dict()
        root_state = self.game.new_initial_state()
        self.act_state_collection(root_state, action_tree_dict)
        if self.random_policy:
            self.action_tree_dict = action_tree_dict
        else:
            self.action_tree_dict = self.batch_dict(action_tree_dict)
        
        s_t = np.zeros(self.obs_dim)
        self.action_tree_dict[tuple(s_t)] = {0:1.0}

    def batch_dict(self, state_tree_dict):
        action_tree_dict = dict()
        obs = np.zeros((self.batch_num, self.obs_dim))
        rnn_states = np.zeros((self.batch_num, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        masks = np.ones((self.batch_num, 1), dtype=np.float32)
        a_acts = np.zeros((self.batch_num, self.act_dim), dtype=int)
        sub_i = 0
        for s, state_ in state_tree_dict.items():
            focus_player = state_.current_player()
            s_t = state_.information_state_tensor(focus_player)
            obs[sub_i, :] = np.array(s_t)
            legal_actions = state_.legal_actions()
            a_act_ = legal2available(self.act_dim, legal_actions)
            a_acts[sub_i, :] = a_act_[0]
            sub_i += 1

            if sub_i >= self.batch_num:
                probs_lists = []
                for policy_i in range(len(self.probs)):
                    if self.deter_mask[policy_i]:
                        act_states, _ = self.policy_networks[policy_i].act(obs, rnn_states, masks, available_actions=a_acts, deterministic=True)
                        act_states = check_np(act_states)
                        probs_all = np.zeros((self.batch_num, self.act_dim))
                        probs_all[np.arange(self.batch_num), act_states[:, 0]] = 1.0
                    else:
                        probs_all = self.policy_networks[policy_i].get_probs_np(obs[:sub_i], available_actions=a_acts[:sub_i])
                    probs_lists.append(probs_all)
                for k in range(self.batch_num):
                    s_ = tuple(obs[k])
                    action_tree_dict[s_] = dict()
                    act_probs = np.zeros(self.act_dim)
                    for policy_i in range(len(self.probs)):
                        act_probs += self.probs[policy_i] * probs_lists[policy_i][k]
                    for i in range(self.act_dim):
                        if a_acts[k][i] > 0.5:
                            action_tree_dict[s_][i] = act_probs[i]
                sub_i = 0

        if sub_i > 0:
            probs_lists = []
            for policy_i in range(len(self.probs)):
                if self.deter_mask[policy_i]:
                    act_states, _ = self.policy_networks[policy_i].act(obs[:sub_i], rnn_states[:sub_i], masks[:sub_i], available_actions=a_acts[:sub_i], deterministic=True)
                    act_states = check_np(act_states)
                    probs_all = np.zeros((sub_i, self.act_dim))
                    probs_all[np.arange(sub_i), act_states[:, 0]] = 1.0
                else:
                    probs_all = self.policy_networks[policy_i].get_probs_np(obs[:sub_i], available_actions=a_acts[:sub_i])
                probs_lists.append(probs_all)
            for k in range(sub_i):
                s_ = tuple(obs[k])
                action_tree_dict[s_] = dict()
                act_probs = np.zeros(self.act_dim)
                for policy_i in range(len(self.probs)):
                    act_probs += self.probs[policy_i] * probs_lists[policy_i][k]
                for i in range(self.act_dim):
                    if a_acts[k][i] > 0.5:
                        action_tree_dict[s_][i] = act_probs[i]

        return action_tree_dict


    def act_state_collection(self, state, action_dict):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                if prob > 0 :
                    new_state = state.child(action)
                    self.act_state_collection(new_state, action_dict)
            return
        
        focus_player = state.current_player()
        s = state.information_state_tensor(focus_player)
        s_f = state.information_state_tensor(1-focus_player)
        legal_actions = state.legal_actions()
        s_t = tuple(np.array(s))
        if self.random_policy:
            action_dict[s_t] = dict()
            for i in range(self.act_dim):
                if i in legal_actions:
                    action_dict[s_t][i] = 1.0 / len(legal_actions)
        else:
            action_dict[s_t] = state

        for action in state.legal_actions():
            new_state = state.child(action)
            self.act_state_collection(new_state, action_dict)

        return

    def action_probabilities(self, state, player_id = None):
        if player_id is None:
            focus_player = state.current_player()
        else:
            focus_player = player_id
        s = state.information_state_tensor(focus_player)
        s_t = tuple(np.array(s))
        act_prob_dict = self.action_tree_dict[s_t]

        return act_prob_dict
    
    def transfer_model_to(self, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        if hasattr(self, "policy_networks"):
            for i in range(len(self.probs)):
                self.policy_networks[i].transfer_model_to(device)
    
    def get_probs(self, obs, rnn_states, actions_example, max_n, masks, available_actions=None, active_masks=None):
        a_acts = check_np(available_actions)
        obs_np = check_np(obs)
        probs_all = np.zeros((a_acts.shape[0], a_acts.shape[1]))
        for i in range(a_acts.shape[0]):
            s_t = tuple(obs_np[i])
            dict_probs = self.action_tree_dict[s_t]
            for idx, prob_ in dict_probs.items():
                probs_all[i][idx] = prob_
        probs_all = check(probs_all).to(**self.tpdv)
        
        return probs_all.unsqueeze(-1)
    
    def get_probs_np(self, obs, rnn_states = None, actions_example = None, max_n = None, masks = None, available_actions=None, active_masks=None):
        a_acts = check_np(available_actions)
        obs_np = check_np(obs)
        probs_all = np.zeros((a_acts.shape[0], a_acts.shape[1]))
        for i in range(a_acts.shape[0]):
            s_t = tuple(obs_np[i])
            dict_probs = self.action_tree_dict[s_t]
            for idx, prob_ in dict_probs.items():
                probs_all[i][idx] = prob_
        
        return probs_all
    
    def act(self, obs, rnn_state, rnn_mask, available_actions, deterministic = True):
        a_acts = check_np(available_actions)
        obs_np = check_np(obs)
        actions = np.zeros(obs_np.shape[0], dtype=int)
        for i in range(obs.shape[0]):
            probs_ = np.zeros(a_acts.shape[1])
            s_t = tuple(obs_np[i])
            dict_probs = self.action_tree_dict[s_t]
            for idx, prob_ in dict_probs.items():
                probs_[idx] = prob_

            action_ = np.random.choice(self.act_dim, p=probs_)
            actions[i] = action_

        return check(actions[:, np.newaxis]).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)


def calc_exp(game, policy):
    exp, expl_per_player = exploitability.nash_conv(
        game, policy, return_only_nash_conv=False)
    return np.array(exp / 2), expl_per_player

def gen_mix_spiel_policy(game, player_id, policies, probs):
    aggregator = policy_aggregator.PolicyAggregator(game)
    aggr_policies = aggregator.aggregate(player_id, policies, probs)
    return aggr_policies
