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
        
        x_in = x_t.permute(0, 3, 2, 1).contiguous()  # [B, D, N, T]
        
        if cond_info is not None:
            graph_rep, hist_rep = cond_info
            cond = (graph_rep, hist_rep)
        else:
            cond = None
        
        noise_pred = self.model(x_in, t, cond, self_cond)
        noise_pred = noise_pred.permute(0, 3, 2, 1).contiguous()
        noise_loss = F.mse_loss(noise_pred, noise)
        
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
        alphas = self.alphas
        alpha_hat = self.alpha_hat

        if cond_info is not None:
            graph_rep, hist_rep = cond_info
            graph_rep = graph_rep.to(device).repeat(num_samples, 1)
            hist_rep = hist_rep.to(device).repeat(num_samples, 1)
            cond = (graph_rep, hist_rep)
        else:
            cond = None

        x = torch.randn(num_samples * B, T, N, D, device=device)
        x = x.permute(0, 3, 2, 1).contiguous()  # [1, D, N, T]

        for i, t in enumerate(reversed(timesteps)):
            t_prev = 0 if i == ddim_steps - 1 else timesteps[-(i + 2)]

            ah_t = alpha_hat[t]
            ah_t_prev = alpha_hat[t_prev]
            
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_batch, cond_info=cond, self_cond=None)

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

        x = x.permute(0, 3, 2, 1).contiguous()
        return x.view(num_samples, B, T, N, D)  # [1, B, T, N, D]
