import torch
import math


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


# ======================================
# From ICLR paper: A Distributional Perspective on Actor-Critic Framework
# ======================================

def _sqdiff(mu1, sig1, w1, mu2, sig2, w2):
    _mu = torch.abs(mu1.unsqueeze(1) - mu2.unsqueeze(2))
    _var = sig1.unsqueeze(1).pow(2) + sig2.unsqueeze(2).pow(2)
    mu = (torch.sqrt(_var * 2 / math.pi)
          * torch.exp(-_mu.pow(2) / (2 * _var + 1e-8))
          + _mu * torch.erf(_mu / (torch.sqrt(2 * _var) + 1e-8)))
    d = mu
    summ = w1.unsqueeze(1) * w2.unsqueeze(2) * d
    return summ.sum(-1).sum(-1, keepdim=True)


def Cramer(mu1, sig1, w1, mu2, sig2, w2):
    mu2 = mu2.detach()
    sig2 = sig2.detach()
    w2 = w2.detach()
    loss = (2 * _sqdiff(mu1, sig1, w1, mu2, sig2, w2)
            - _sqdiff(mu1, sig1, w1, mu1, sig1, w1)
            - _sqdiff(mu2, sig2, w2, mu2, sig2, w2))
    return loss


def huber_quantile_loss(input, target, quantiles):
    n_quantiles = quantiles.size(-1)
    diff = target.unsqueeze(2) - input.unsqueeze(1)
    taus = quantiles.unsqueeze(-1).unsqueeze(1)
    taus = taus.expand(-1, n_quantiles, -1, -1)
    loss = diff.pow(2) * (taus - (diff < 0).float()).abs()
    return loss.squeeze(3).sum(-1).mean(-1)


def sample_cramer(samples1, samples2):
    d = (2 * _sqdiff_sample(samples1, samples2)
         - _sqdiff_sample(samples1, samples1)
         - _sqdiff_sample(samples2, samples2))
    return d


def _sqdiff_sample(samples1, samples2):
    assert samples1.size() == samples2.size()
    assert len(samples1.size()) == 2
    diff = samples1.unsqueeze(1) - samples2.unsqueeze(2)
    return diff.abs().mean(-1, keepdim=True)
