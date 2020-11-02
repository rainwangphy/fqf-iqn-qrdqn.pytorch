from fqf_iqn_qrdqn.utils import cdf_gauss
import torch
print(cdf_gauss(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)))
