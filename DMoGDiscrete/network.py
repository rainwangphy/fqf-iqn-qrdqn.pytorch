import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fqf_iqn_qrdqn.network import NoisyLinear


class MoGNet(nn.Module):
    def __init__(self, num_actions, embedding_dim=7 * 7 * 64, dueling_net=False,
                 noisy_net=False):
        super(MoGNet, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
