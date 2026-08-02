import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from stream_rl.src.utils.sparse_init import sparse_init
from stream_rl.src.utils.optim import ObGD as Optimizer


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class ActorContinuous(nn.Module):
    def __init__(self, n_obs=3, n_actions=1, hidden_size=128):
        super(ActorContinuous, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions) - 0.5)
        self.apply(initialize_weights)

    def forward(self, x):
        h = self.fc_layer(x)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        h = self.hidden_layer(h)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        mu = self.mu_head(h)
        # clamp prevents std from collapsing to 0 or exploding to inf
        log_std = torch.clamp(self.log_std, min=-2.0, max=1.0)
        std = log_std.exp().expand_as(mu)
        return mu, std


class Critic(nn.Module):
    def __init__(self, n_obs=3, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        h = self.fc_layer(x)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        h = self.hidden_layer(h)
        h = F.layer_norm(h, h.size())
        h = F.leaky_relu(h)
        return self.linear_layer(h)


class StreamACContinuous(nn.Module):
    def __init__(self,
                 n_obs=3,
                 n_actions=1,
                 hidden_size=32,
                 action_low=-2.0,
                 action_high=2.0,
                 lr=1.0,
                 gamma=0.99,
                 lamda=0.8,
                 kappa_policy=3.0,
                 kappa_value=2.0,
                 entropy_coeff=0.0):
        super(StreamACContinuous, self).__init__()
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.action_low = action_low
        self.action_high = action_high

        self.policy_net = ActorContinuous(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)

        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        mu, std = self.policy_net(x)
        return Normal(mu, std)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        dist = self.pi(x)
        raw_action = dist.sample()
        # clip into the valid action range (Pendulum torque bounds)
        action = torch.clamp(raw_action, self.action_low, self.action_high)
        return action.numpy(), raw_action.detach()

    def update_params(self, s, raw_action, r, s_prime, done, overshooting_info=False):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_mask), dtype=torch.float)
        a = raw_action if torch.is_tensor(raw_action) else torch.tensor(np.array(raw_action), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        dist = self.pi(s)
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
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")