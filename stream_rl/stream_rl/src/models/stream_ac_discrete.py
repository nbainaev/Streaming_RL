import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from stream_rl.src.utils.sparse_init import sparse_init
from stream_rl.src.utils.optim import ObGD as Optimizer


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, memory_in=0):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        # Optional memory branch (HELM/SHELM-style): see StreamQ for details.
        self.mem_adapter = None
        head_in = hidden_size
        if memory_in and memory_in > 0:
            self.mem_adapter = nn.Sequential(
                nn.Linear(memory_in, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
            )
            head_in = 2 * hidden_size
        self.hidden_layer = nn.Linear(head_in, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x, mem=None):
        h = self.fc_layer(x)
        # uncomment if you don't use one=hot observations
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        if self.mem_adapter is not None and mem is not None:
            if isinstance(mem, np.ndarray):
                mem = torch.tensor(np.array(mem), dtype=torch.float)
            h = torch.cat([h, self.mem_adapter(mem)], dim=-1)
        h = self.hidden_layer(h)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        pref = self.fc_pi(h)
        return pref

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128, memory_in=0):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        # Optional memory branch (HELM/SHELM-style): see StreamQ for details.
        self.mem_adapter = None
        head_in = hidden_size
        if memory_in and memory_in > 0:
            self.mem_adapter = nn.Sequential(
                nn.Linear(memory_in, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
            )
            head_in = 2 * hidden_size
        self.hidden_layer  = nn.Linear(head_in, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x, mem=None):
        h = self.fc_layer(x)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        if self.mem_adapter is not None and mem is not None:
            if isinstance(mem, np.ndarray):
                mem = torch.tensor(np.array(mem), dtype=torch.float)
            h = torch.cat([h, self.mem_adapter(mem)], dim=-1)
        h = self.hidden_layer(h)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        return self.linear_layer(h)

class StreamAC(nn.Module):
    def __init__(self,
                 n_obs=11,
                 n_actions=3,
                 hidden_size=32,
                 lr=1.0,
                 gamma=0.99,
                 lamda=0.8,
                 kappa_policy=3.0,
                 kappa_value=2.0,
                 entropy_coeff=False,
                 device='cpu',
                 memory_in=0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = device
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, memory_in=memory_in).to(self.device)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size, memory_in=memory_in).to(self.device)

        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)


    def pi(self, x, mem=None):
        preferences = self.policy_net(x, mem)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x, mem=None):
        return self.value_net(x, mem)

    def sample_action(self, s, mem=None):
        x = torch.from_numpy(s).float()
        probs = self.pi(x.to(self.device), mem).to('cpu')
        dist = Categorical(probs)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, overshooting_info=False, mem=None, mem_prime=None):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float).to(self.device), torch.tensor(np.array(a)).to(self.device), \
                                         torch.tensor(np.array(r)).to(self.device), torch.tensor(np.array(s_prime), dtype=torch.float).to(self.device), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float).to(self.device)

        v_s, v_prime = self.v(s, mem), self.v(s_prime, mem_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s
        
        probs = self.pi(s, mem)
        dist = Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -self.entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s, mem), self.v(s_prime, mem_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")