# from fqf_iqn_qrdqn.utils import cdf_gauss
# import torch
# print(cdf_gauss(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)))

import os
import yaml
import argparse
from datetime import datetime

import torch

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.network import DQNBase
from DMoGDiscrete.network import MoGNet

env = make_pytorch_env('PongNoFrameskip-v4')
print(env.action_space.n)
dqn_net = DQNBase(num_channels=env.observation_space.shape[0])
dmog_net = MoGNet(env.action_space.n)

state = env.reset()
state = torch.ByteTensor(
    state).unsqueeze(0).to('cpu').float() / 255.
dqn_out = dqn_net(state)
# print(dqn_out.shape)
out_pi, out_mu, out_sigma = dmog_net(dqn_out)
print(out_pi)
print(out_mu)
print(out_sigma)
print(out_pi[0, :, 3])
print(torch.sum(out_pi, dim=2, keepdim=True))
print(torch.sum(out_pi, dim=2, keepdim=True).mean())
