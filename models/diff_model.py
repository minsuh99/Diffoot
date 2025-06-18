import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import per_player_fde_loss

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

    def forward(self, x_0, t=None, cond_info=None, self_cond=None):
        B = x_0.size(0)
        device = x_0.device
        
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        x_t, noise = self.q_sample(x_0, t)
        x_t_in = x_t.permute(0, 3, 2, 1)

        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)               # [B,1,1,1]
        sqrt_a = torch.sqrt(a_hat)                               # √α̂_t
        sqrt_one_minus = torch.sqrt(1 - a_hat)                          # √(1−α̂_t)
        v_target = sqrt_a * noise - sqrt_one_minus * x_0                 # v_t = √α ε_t − √(1−α) x₀
        
        x_t_in = x_t.permute(0, 3, 2, 1)
        z = self.model(x_t_in, t, cond_info, self_cond)
        z = z.permute(0, 3, 2, 1)

        v_pred, log_var = z[..., :2], z[..., 2:]
        log_var = log_var.clamp(-10, 2)
        var = log_var.exp()

        v_loss = F.mse_loss(v_pred, v_target)
        
        # NLL loss computing
        nll = 0.5 * ((v_target - v_pred) ** 2 / var + log_var + math.log(2 * math.pi))
        noise_nll = nll.mean()
        
        # Relative Coords FDE
        x0_pred = sqrt_a * x_t - sqrt_one_minus * v_pred  # x₀ = √α̂_t x_t − √(1−α̂_t) v_pred
        x0_fde = per_player_fde_loss(x0_pred, x_0)
    
        return v_loss, noise_nll, x0_fde
    
    # DDIM Sampling
    @torch.no_grad()
    def generate(self, shape, cond_info=None, ddim_steps=50, eta=0.0, num_samples=1, self_conditioning_ratio=0.0):
        B, T, N, D = shape
        device = next(self.parameters()).device

        timesteps = torch.linspace(0, self.num_steps - 1, ddim_steps, device=device).long()
        alpha_hat = self.alpha_hat

        if cond_info is not None:
            cond_info = cond_info.to(device).unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
            cond_info = cond_info.view(num_samples * B, *cond_info.shape[2:])

        x = torch.randn(num_samples * B, T, N, D, device=device)
        self_cond = None

        for i, t in enumerate(reversed(timesteps)):
            t_prev = 0 if i == ddim_steps - 1 else timesteps[-(i + 2)]

            ah_t = alpha_hat[t]
            ah_t_prev = alpha_hat[t_prev]
            
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            
            x_in = x.permute(0, 3, 2, 1)
            use_self_cond = self_cond if torch.rand(1).item() < self_conditioning_ratio else None
        
            z = self.model(x_in, t_batch, cond_info, use_self_cond).permute(0, 3, 2, 1)
            v_pred = z[..., :2]
            
            sqrt_ah_t  = torch.sqrt(ah_t)           # √α̂_t
            sqrt_one_minus_t  = torch.sqrt(1 - ah_t)       # √(1−α̂_t)
            x0_pred =  sqrt_ah_t * x - sqrt_one_minus_t * v_pred
            eps_pred = (x - sqrt_ah_t * x0_pred) / sqrt_one_minus_t

            # Self-conditioning 업데이트 (ratio > 0일 때만)
            if self_conditioning_ratio > 0:
                self_cond = x0_pred.permute(0, 3, 2, 1).detach()
            
            if t_prev > 0:
                if eta > 0:
                    sigma_t = eta * torch.sqrt((1 - ah_t_prev) / (1 - ah_t)) * torch.sqrt(1 - ah_t / ah_t_prev)
                    noise = torch.randn_like(x)
                else:
                    sigma_t = 0.0
                    noise = torch.zeros_like(x)

                c1 = torch.sqrt(ah_t_prev)
                c2 = torch.sqrt(1 - ah_t_prev - sigma_t**2)
                x_new = x.clone()
                x_new = c1 * x0_pred + c2 * eps_pred + sigma_t * noise
                x = x_new
            else:
                x = x0_pred

        result = x.view(num_samples, B, T, N, D)
        return result