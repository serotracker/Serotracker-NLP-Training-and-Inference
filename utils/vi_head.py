import torch
import math

class VIHead(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mean = torch.nn.Parameter(torch.Tensor(1, 768))
    self.bias = torch.nn.Parameter(torch.Tensor(1))
    self.L = torch.nn.Parameter(torch.Tensor(768, 768))

    self.reset_parameters()

  def reset_parameters(self) -> None:
    torch.nn.init.kaiming_uniform_(self.mean, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mean)
      bound = 1 / math.sqrt(fan_in)
      torch.nn.init.uniform_(self.bias, -bound, bound)

    torch.nn.init.normal_(self.L, std = 0.00001)

  def forward(self, x, n_samples = 40):
    output_mean = x @ self.mean.T + self.bias #b 1
    noise_std = x @ torch.tril(self.L) #b d
    noise_std = torch.sum(noise_std**2, 1) #b
    noise_samples = noise_std[:, None] * torch.empty([x.shape[0], n_samples], dtype = x.dtype, device = x.device).normal_() #b n
    output_samples = output_mean + noise_samples #+ torch.zeros_like(noise_samples)

    return torch.stack([output_samples, torch.zeros_like(output_samples)], -1)
  
  def get_kl(self):
    pv = .003
    det_term = -torch.sum(torch.log(torch.diag(self.L)**2))
    d_term = -768
    tr_term = torch.sum(torch.tril(self.L)**2)/pv
    mean_term = torch.sum(self.mean**2/ pv)
    
    return 0.5 * (det_term + d_term + tr_term + mean_term)



# class VIHead(torch.nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.mean = torch.nn.Parameter(torch.Tensor(1, 769))
#     self.L = torch.nn.Parameter(torch.Tensor(769, 769))

#     self.reset_parameters()

#   def reset_parameters(self) -> None:
#     torch.nn.init.kaiming_uniform_(self.mean, a=math.sqrt(5))

#     torch.nn.init.normal_(self.L, std = 0.00001)

#   def forward(self, x, n_samples = 40):
#     x_tilde = torch.cat([x, torch.ones([x.shape[0], 1], device = x.device, dtype = x.dtype)], 1)
#     output_mean = x_tilde @ self.mean.T
#     noise_std = x_tilde @ torch.tril(self.L) #b d
#     noise_std = torch.sum(noise_std**2, 1) #b
#     noise_samples = noise_std[:, None] * torch.empty([x.shape[0], n_samples], dtype = x.dtype, device = x.device).normal_() #b n
#     output_samples = output_mean + noise_samples #+ torch.zeros_like(noise_samples)

#     return torch.stack([output_samples, torch.zeros_like(output_samples)], -1)
  
#   def get_kl(self):
#     pv = .003
#     det_term = -torch.sum(torch.log(torch.diag(self.L)**2))
#     d_term = -769
#     tr_term = torch.sum(torch.tril(self.L)**2)/pv
#     mean_term = torch.sum(self.mean**2/ pv)
    
#     return 0.5 * (det_term + d_term + tr_term + mean_term)