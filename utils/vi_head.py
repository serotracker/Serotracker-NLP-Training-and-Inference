import torch

class VIHead(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mean = torch.nn.Parameter(torch.Tensor(1, 768))
    self.bias = torch.nn.Parameter(torch.Tensor(1))
    self.L = torch.nn.Parameter(torch.Tensor(768, 768))

  def reset_parameters(self) -> None:
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mean)
      bound = 1 / math.sqrt(fan_in)
      torch.nn.init.uniform_(self.bias, -bound, bound)

    torch.nn.init.normal_(self.L, std = 0.03)

  def forward(self, x, n_samples = 20):
    output_mean = x @ self.mean.T + self.bias #b 1
    noise_std = x @ self.L #b d
    noise_std = torch.sum(noise_std**2, 1) #b
    noise_samples = noise_std[:, None] * torch.empty([x.shape[0], n_samples], dtype = x.dtype, device = x.device) #b n
    output_samples = output_mean + noise_samples

    return torch.stack([output_samples, torch.zeros_like(output_samples)], -1)
  
  def get_kl(self):
    det_term = -torch.sum(torch.log(torch.diag(self.L)**2))
    d_term = -768
    tr_term = torch.sum(torch.tril(self.L)**2)
    mean_term = torch.sum(self.mean**2)
    
    return 0.5 * (det_term + d_term + tr_term + mean_term)