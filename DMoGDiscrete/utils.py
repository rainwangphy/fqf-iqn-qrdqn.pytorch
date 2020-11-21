import torch


###########-----------------DMOG------------------############################

def evaluate_mog_at_action(mog_pi, mog_mu, mog_sigma, actions):
    assert mog_pi.shape[0] == actions.shape[0]

    batch_size = mog_pi.shape[0]
    num_gaussian = mog_pi.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, num_gaussian, 1)

    # Calculate quantile values at specified actions.
    mog_pi_sa = mog_pi.gather(dim=2, index=action_index)
    mog_mu_sa = mog_mu.gather(dim=2, index=action_index)
    mog_sigma_sa = mog_sigma.gather(dim=2, index=action_index)

    return mog_pi_sa, mog_mu_sa, mog_sigma_sa


def cdf_gauss(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x - mu) / (torch.sqrt(torch.tensor(2.0)) * sigma)))


def calculate_dmog_loss(pi, mu, sigma,
                        tar_pi, tar_mu, tar_sigma,
                        eta, beta, delta,
                        weight):
    dmog_loss = 0

    # cramer distance

    return dmog_loss
