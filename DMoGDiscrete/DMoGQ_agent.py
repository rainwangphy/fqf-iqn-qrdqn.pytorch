
from fqf_iqn_qrdqn.agent.base_agent import BaseAgent
from DMoGDiscrete.DMoGQ import DMoGQ
from fqf_iqn_qrdqn.utils import disable_gradients, update_params


class DMoGQAgent(BaseAgent):
    def __init__(self, env, test_env, log_dir, num_steps=5 * (10 ** 7),
                 batch_size=32, N=200, kappa=1.0, lr=5e-5, memory_size=10 ** 6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(DMoGQAgent, self).__init__(env, test_env, log_dir, num_steps, batch_size, memory_size,
                                         gamma, multi_step, update_interval, target_update_interval,
                                         start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
                                         double_q_learning, dueling_net, noisy_net, use_per, log_interval,
                                         eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
                                         cuda, seed)

        # Online network.
        self.online_net = DMoGQ(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)

        # Target network.
        self.target_net = DMoGQ(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)
