import numpy as np
import torch

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

def norm_batch(raw_mat):
    norms = np.linalg.norm(raw_mat, axis=-1)
    mask = norms > 1
    raw_mat[mask, :] = raw_mat[mask, :] / norms[mask, np.newaxis]
    return raw_mat

class Goofspiel_random:
    def __init__(self):
        self.policy_name = "random_policy"

    def act(self, obs, a_acts):
        actions = np.zeros((a_acts.shape[0], 1), dtype=int)

        for i, row in enumerate(a_acts):
            ones_indices = np.where(row > 0.5)[0]
            if ones_indices.size > 0:
                chosen_index = np.random.choice(ones_indices, size=1)
                actions[i] = chosen_index
            else:
                actions[i] = -1

        return actions
    
class Policy_goofspiel_random:
    def __init__(self, policy_network, device = torch.device("cpu")):
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = Goofspiel_random()

    def transfer_model_to(self, device):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy_network.transfer_model_to(device)
        self.actor = self.policy_network.actor

    def act(self, obs, rnn_state, rnn_mask, available_actions, deterministic = True):
        actions = self.action_generator.act(obs, check_np(available_actions))
        return check(actions).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
    def get_probs(self, obs, rnn_states, actions_example, max_n, masks, available_actions=None, active_masks=None):
        a_acts = check_np(available_actions)
        probs_all = torch.zeros(a_acts.shape[0], a_acts.shape[1]).to(**self.tpdv)
        probs_all[a_acts == 0] = -1e10
        probs_all=torch.softmax(probs_all, -1)
        
        return probs_all.unsqueeze(-1)