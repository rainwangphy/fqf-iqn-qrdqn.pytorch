from fqf_iqn_qrdqn.model.base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase
from DMoGDiscrete.network import MoGNet


class DMoGQ(BaseModel):

    def __init__(self, num_channels, num_actions,
                 num_gaussians=5,
                 embedding_dim=7 * 7 * 64,
                 dueling_net=False, noisy_net=False):
        super(DMoGQ, self).__init__()
        # Feature extractor of DQN: Mapping the state, i.e., image, to the embedding
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Then, mapping the embedding to the Q value distribution
        self.mog_net = MoGNet(num_actions=num_actions, num_gaussians=num_gaussians)

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        out_pi, out_mu, out_sigma = self.mog_net(state_embeddings)
        return out_pi, out_mu, out_sigma
