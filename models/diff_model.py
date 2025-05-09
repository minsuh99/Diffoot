import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import per_player_frechet_loss, per_player_fde_loss

class DiffusionTrajectoryModel(nn.Module):
    # def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
    def __init__(self, model, num_steps=1000, cosine_s=0.008):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        
        # ts = torch.linspace(0, 1, num_steps)
        # betas = beta_start + (beta_end - beta_start) * (ts ** 2)
        # alphas = 1.0 - betas
        # alpha_hat = torch.cumprod(alphas, dim=0)
        
        t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
        alphas_cumprod = torch.cos((t + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, min=0.0001, max=0.9999)
        
        alphas    = 1.0 - betas
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
        x_t_in = x_t.permute(0, 3, 2, 1)

        
        noise_pred = self.model(x_t_in, t, cond_info, self_cond)
        noise_true = noise.permute(0, 3, 2, 1)
        
        noise_loss = F.mse_loss(noise_pred, noise_true)
        noise_pred = noise_pred.permute(0, 3, 2, 1)
        
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - torch.sqrt(1 - a_hat) * noise_pred) / torch.sqrt(a_hat)

        player_frechet = per_player_frechet_loss(x_0_pred, x_0)
        player_fde = per_player_fde_loss(x_0_pred, x_0)
        
        return noise_loss, player_frechet, player_fde

    # DDIM Sampling
    @torch.no_grad()
    def generate(self, shape, cond_info=None, ddim_steps=50, eta=0.0, num_samples=1):
        B, T, N, D = shape
        device = next(self.parameters()).device

        timesteps = torch.linspace(0, self.num_steps - 1, ddim_steps, device=device).long()
        alpha_hat = self.alpha_hat

        if cond_info is not None:
            cond_info = cond_info.to(device).unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
            cond_info = cond_info.view(num_samples * B, *cond_info.shape[2:])

        x = torch.randn(num_samples * B, T, N, D, device=device)

        for i, t in enumerate(reversed(timesteps)):
            t_prev = 0 if i == ddim_steps - 1 else timesteps[-(i + 2)]

            ah_t = alpha_hat[t]
            ah_t_prev = alpha_hat[t_prev]
            
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            
            x_in = x.permute(0, 3, 2, 1)
            noise_pred = self.model(x_in, t_batch, cond_info, self_cond=None)
            noise_pred = noise_pred.permute(0, 3, 2, 1)

            x0_pred = (x - torch.sqrt(1 - ah_t) * noise_pred) / torch.sqrt(ah_t)

            # deterministic update
            x_det = torch.sqrt(ah_t_prev) * x0_pred + torch.sqrt(1 - ah_t_prev) * noise_pred

            # stochastic noise term
            if eta > 0 and t_prev > 0:
                sigma_t = eta * torch.sqrt((1 - ah_t_prev) / (1 - ah_t)) * torch.sqrt(1 - ah_t)
                noise = torch.randn_like(x)
                x = x_det + sigma_t * noise
            else:
                x = x_det

        return x.view(num_samples, B, T, N, D)  # [1, B, T, N, D]
