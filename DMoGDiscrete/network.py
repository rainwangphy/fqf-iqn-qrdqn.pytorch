import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fqf_iqn_qrdqn.network import NoisyLinear


class MoGNet(nn.Module):
    def __init__(self, num_actions, num_gaussians=5,
                 log_var_min=-20, log_var_max=2,
                 embedding_dim=7 * 7 * 64, dueling_net=False,
                 noisy_net=False):
        super(MoGNet, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        self.num_actions = num_actions
        self.num_gaussians = num_gaussians
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.out_dim = num_actions * num_gaussians * 3

        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.net = nn.Sequential(
            linear(embedding_dim, 512),
            nn.ReLU(),
            linear(512, self.out_dim)
        )

    def forward(self, state_embeddings):
        batch_size = state_embeddings.shape[0]

        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)
        dmog_values = self.net(state_embeddings)
        [out_pi, out_mu, out_sigma] = torch.chunk(dmog_values, chunks=3, dim=2)

        out_pi = out_pi.view(batch_size, self.num_gaussians, self.num_actions)
        out_mu = out_mu.view(batch_size, self.num_gaussians, self.num_actions)
        out_sigma = out_sigma.view(batch_size, self.num_gaussians, self.num_actions)

        out_pi = torch.softmax(out_pi, dim=1)
        out_sigma = torch.clamp(out_sigma, self.log_var_min, self.log_var_max)
        out_sigma = torch.exp(out_sigma)
        return out_pi, out_mu, out_sigma
