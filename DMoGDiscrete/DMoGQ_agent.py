from fqf_iqn_qrdqn.agent.base_agent import BaseAgent
from DMoGDiscrete.DMoGQ import DMoGQ
from fqf_iqn_qrdqn.utils import disable_gradients, update_params
from torch.optim import Adam
import torch
from DMoGDiscrete.utils import calculate_dmog_loss, evaluate_mog_at_action


class DMoGQAgent(BaseAgent):
    def __init__(self, env, test_env, log_dir, num_steps=5 * (10 ** 7),
                 batch_size=32,
                 num_gaussians=5, eta=0.5, beta=3, delta=10,
                 lr=5e-5, memory_size=10 ** 6,
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
        self.num_gaussians = num_gaussians
        self.eta = eta
        self.beta = beta
        self.delta = delta
        # Online network.
        self.online_net = DMoGQ(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions,
            num_gaussians=num_gaussians,
            dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)

        # Target network.
        self.target_net = DMoGQ(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions,
            num_gaussians=num_gaussians,
            dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2 / batch_size)

    def learn(self):
        self.learning_steps += 1
        self.online_net.sample_noise()
        self.target_net.sample_noise()

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights = \
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size)
            weights = None

        dmog_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones, weights)

        update_params(
            self.optim, dmog_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

    def calculate_loss(self, states, actions, rewards, next_states, dones,
                       weights):
        mog_pi, mog_mu, mog_sigma = self.online_net(states=states)
        mog_pi_sa, mog_mu_sa, mog_sigma_sa = evaluate_mog_at_action(mog_pi=mog_pi, mog_mu=mog_mu, mog_sigma=mog_sigma,
                                                                    actions=actions)
        assert mog_pi_sa.shape == (self.batch_size, self.num_gaussians, 1)

        with torch.no_grad():
            next_mog_pi, next_mog_mu, next_mog_sigma = self.target_net(states=next_states)
            mog_q_value = torch.sum(next_mog_pi * next_mog_mu, dim=1)
            next_actions = torch.argmax(mog_q_value, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            next_mog_pi_sa, next_mog_mu_sa, next_mog_sigma_sa = \
                evaluate_mog_at_action(mog_pi=next_mog_pi, mog_mu=next_mog_mu, mog_sigma=next_mog_sigma,
                                       actions=next_actions)
            assert next_mog_pi_sa.shape == (self.batch_size, 1, self.num_gaussians)

            # Calculate target mog values.
            target_mog_mu_sa = rewards[..., None] + (1.0 - dones[..., None]) * self.gamma_n * next_mog_mu_sa
            target_mog_pi_sa = torch.tensor(1.0 / self.num_gaussians) * dones[..., None] + (
                    1.0 - dones[..., None]) * next_mog_pi_sa
            target_mog_sigma_sa = torch.tensor(1.0) * dones[..., None] + (
                    1.0 - dones[..., None]) * self.gamma_n * next_mog_sigma_sa
            assert target_mog_mu_sa.shape == (self.batch_size, self.num_gaussians, 1)

        dmog_loss = calculate_dmog_loss(mog_pi_sa, mog_mu_sa, mog_sigma_sa,
                                        target_mog_mu_sa, target_mog_pi_sa, target_mog_sigma_sa,
                                        eta=self.eta, beta=self.beta, delta=self.delta,
                                        weight=weights)
        return dmog_loss
