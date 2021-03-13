import torch
import math

class VIHead(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mean = torch.nn.Parameter(torch.Tensor(1, 768))
    self.bias = torch.nn.Parameter(torch.Tensor(1))
    self.L = torch.nn.Parameter(torch.Tensor(768, 768))
    
    one_inverse = torch.eye(768) - torch.diag(torch.ones(767), -1)

    g = torch.tril(torch.ones_like(self.L))
    self.precond = torch.tril((1/torch.sqrt(torch.diag(g @ g.T))).view(-1, 1) * torch.ones([1, 768]))
    self.precond = one_inverse @ self.precond
    

    self.reset_parameters()

    print((torch.tril(self.L) @ self.precond) @ (torch.tril(self.L) @ self.precond).T)
    # print((torch.tril(self.L)) @ (torch.tril(self.L)).T)

  def reset_parameters(self) -> None:
    torch.nn.init.kaiming_uniform_(self.mean, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mean)
      bound = 1 / math.sqrt(fan_in)
      torch.nn.init.uniform_(self.bias, -bound, bound)

    torch.nn.init.normal_(self.L, std = 0.01)

  def forward(self, x, n_samples = 40):
    output_mean = x @ self.mean.T + self.bias #b 1
    noise_std = x @ (torch.tril(self.L) @ self.precond.to(x.device).type(x.dtype)) #b d
    noise_std = torch.sum(noise_std**2, 1) #b
    noise_samples = noise_std[:, None] * torch.empty([x.shape[0], n_samples], dtype = x.dtype, device = x.device).normal_() #b n
    output_samples = output_mean + noise_samples #+ torch.zeros_like(noise_samples)

    return torch.stack([output_samples, torch.zeros_like(output_samples)], -1)
  
  def get_kl(self):
    precond = self.precond.to(self.L.device).type(self.L.dtype)
    pv = .01
    det_term = -torch.sum(torch.log(torch.diag(self.L)**2)) - torch.sum(torch.log(torch.diag(precond**2)))
    # print(torch.diag(precond).shape)
    # print(det_term)
    d_term = -768
    tr_term = torch.sum((torch.tril(self.L) @ precond)**2)/pv
    mean_term = torch.sum(self.mean**2/ pv)
    # print(0.5 * (det_term + d_term + tr_term + mean_term))
    
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