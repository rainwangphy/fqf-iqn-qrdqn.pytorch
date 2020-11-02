from fqf_iqn_qrdqn.model.base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase


class DMoGQ(BaseModel):

    def __init__(self, num_channels, num_actions, N=200, embedding_dim=7 * 7 * 64,
                 dueling_net=False, noisy_net=False):
        super(DMoGQ, self).__init__()

        # Feature extractor of DQN: Mapping the state, i.e., image, to the embedding
        self.dqn_net = DQNBase(num_channels=num_channels)

        # Then, mapping the embedding to the Q value distribution

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
