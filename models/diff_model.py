import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import per_player_mse_loss

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

    def forward(self, x_0, t=None, cond_info=None, self_cond=None, initial_pos=None):
        B = x_0.size(0)
        device = x_0.device
        
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        x_t, noise = self.q_sample(x_0, t)
        x_t_in = x_t.permute(0, 3, 2, 1)

        z = self.model(x_t_in, t, cond_info, self_cond)
        z = z.permute(0, 3, 2, 1)

        eps_pred, log_var = z[..., :6], z[..., 6:]
        log_var = log_var.clamp(-5, 5)
        var = log_var.exp()
        
        eps_pred_rel = eps_pred[..., 2:4]  # 상대좌표 부분만
        noise_rel = noise[..., 2:4]
        log_var_rel = log_var[..., 2:4]
        var_rel = var[..., 2:4]
        
        noise_mse = F.mse_loss(eps_pred_rel, noise_rel)
        
        # NLL loss computing
        nll = 0.5 * ((noise_rel - eps_pred_rel) ** 2 / var_rel + log_var_rel + math.log(2 * math.pi))
        noise_nll = nll.mean()

        # Player-wise MSE
        if initial_pos is not None:
            target_abs = x_0[..., :2]

            a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
            x0_pred_rel = (x_t[..., 2:4] - torch.sqrt(1 - a_hat) * eps_pred_rel) / torch.sqrt(a_hat)

            initial_pos_expanded = initial_pos.unsqueeze(1).expand(-1, x_0.size(1), -1, -1)  # [B, T, N, 2]
            pred_abs = x0_pred_rel + initial_pos_expanded  # [B, T, N, 2]

            player_mse = per_player_mse_loss(pred_abs, target_abs)
        
        return noise_mse, noise_nll, player_mse
    
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
            eps_pred, log_var = z[..., :6], z[..., 6:]
            
            # 훈련과 동일하게 상대좌표 부분만 사용
            noise_pred = eps_pred
            
            x0_pred = x.clone()
            x0_pred[..., 2:4] = (x[..., 2:4] - torch.sqrt(1 - ah_t) * noise_pred[..., 2:4]) / torch.sqrt(ah_t)
            
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
                x_new[..., 2:4] = c1 * x0_pred[..., 2:4] + c2 * noise_pred[..., 2:4] + sigma_t * noise[..., 2:4]
                x = x_new
            else:
                x = x0_pred

        result = x.view(num_samples, B, T, N, D)
        return result