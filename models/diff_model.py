import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import per_player_frechet_loss, per_player_fde_loss

class DiffusionTrajectoryModel(nn.Module):
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        
        ts = torch.linspace(0, 1, num_steps)
        betas = beta_start + (beta_end - beta_start) * (ts ** 2)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_hat', alpha_hat)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(a_hat) * x_0 + torch.sqrt(1 - a_hat) * noise
        return x_t, noise

    def forward(self, x_0, cond_info=None, self_cond=None):
        B = x_0.size(0)
        device = x_0.device
        
        
        
        t = torch.randint(0, self.num_steps, (B,), device=device)
        x_t, noise = self.q_sample(x_0, t)
        
        noise_pred = self.model(x_t, t, cond_info, self_cond)
        noise_loss = F.mse_loss(noise_pred, noise)
        
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - torch.sqrt(1 - a_hat) * noise_pred) / torch.sqrt(a_hat)
        
        player_frechet = per_player_frechet_loss(x_0_pred, x_0)
        player_fde = per_player_fde_loss(x_0_pred, x_0)
        return noise_loss, player_frechet, player_fde

    @torch.no_grad()
    def generate(self, shape, cond_info=None, num_samples=10):
        B, T, N, D = shape
        device = next(self.parameters()).device
        
        if cond_info is not None:
            cond_info = cond_info.to(device).repeat(num_samples, 1, 1, 1)
            
        x_t = torch.randn(num_samples * B, T, N, D, device=device)
        self_cond = None
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x_t, t_batch, cond_info, self_cond)
            
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            
            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]
            
            x_prev = (1.0 / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred) + torch.sqrt(beta) * noise
            
            self_cond = x_prev
            x_t = x_prev
        return x_t.view(num_samples, B, T, N, D)
