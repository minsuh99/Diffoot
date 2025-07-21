import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffoot(nn.Module):    
    def __init__(self, model, num_steps=1000, s=0.008):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, dtype=torch.float64)

        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()
        
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

    def forward(self, target_abs, reference_point, zscore_stats, t=None, cond_info=None):
        B = target_abs.size(0)
        T = target_abs.size(1)
        device = target_abs.device
        
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)

        N = reference_point.size(1) // 2
        target_abs_4d = target_abs.view(B, T, N, 2)
        ref_raw = reference_point.view(B, N, 2)
        
        if zscore_stats is not None:
            px_s = zscore_stats['player_x_std']
            py_s = zscore_stats['player_y_std']
            rx_m, rx_s = zscore_stats['rel_x_mean'], zscore_stats['rel_x_std']
            ry_m, ry_s = zscore_stats['rel_y_mean'], zscore_stats['rel_y_std']

            abs_raw = torch.zeros_like(target_abs_4d)
            abs_raw[..., 0] = target_abs_4d[..., 0] * px_s + zscore_stats['player_x_mean']
            abs_raw[..., 1] = target_abs_4d[..., 1] * py_s + zscore_stats['player_y_mean']

            rel_raw = abs_raw - ref_raw.unsqueeze(1)  # [B, T, N, 2]

            rel_norm = torch.zeros_like(rel_raw)
            rel_norm[..., 0] = (rel_raw[..., 0] - rx_m) / rx_s
            rel_norm[..., 1] = (rel_raw[..., 1] - ry_m) / ry_s
            
            x_0 = rel_norm
        else:
            x_0 = target_abs_4d - ref_raw.unsqueeze(1)

        x_t, noise = self.q_sample(x_0, t)
        x_t_in = x_t.permute(0, 3, 2, 1)

        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_a = torch.sqrt(a_hat)
        sqrt_o = torch.sqrt(1 - a_hat)
        v_target = sqrt_a * noise - sqrt_o * x_0
        
        z = self.model(x_t_in, t, cond_info)
        z = z.permute(0, 3, 2, 1)

        v_pred, log_var = z[..., :2], z[..., 2:]
        log_var = log_var.clamp(-10, 2)
        var = log_var.exp()

        v_loss = F.mse_loss(v_pred, v_target)
        
        # NLL loss computing
        nll = 0.5 * ((v_target - v_pred) ** 2 / var + log_var + math.log(2 * math.pi))
        noise_nll = nll.mean()

        return v_loss, noise_nll
    
    # DDIM Sampling
    @torch.no_grad()
    def generate(self, shape, reference_point, cond_info=None, ddim_steps=50, eta=0.0, num_samples=1):
        B, T, N, D = shape
        device = next(self.parameters()).device

        timesteps = torch.linspace(0, self.num_steps - 1, ddim_steps, device=device).long()
        alpha_hat = self.alpha_hat

        if cond_info is not None:
            cond_info = cond_info.to(device).unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
            cond_info = cond_info.view(num_samples * B, *cond_info.shape[2:])

        ref_raw = reference_point.view(B, N, D)
        ref_raw = ref_raw.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        ref_raw = ref_raw.view(num_samples * B, N, D)

        x = torch.randn(num_samples * B, T, N, D, device=device)

        for i, t in enumerate(reversed(timesteps)):
            t_prev = 0 if i == ddim_steps - 1 else timesteps[-(i + 2)]

            ah_t = alpha_hat[t]
            ah_t_prev = alpha_hat[t_prev]
            
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            
            x_in = x.permute(0, 3, 2, 1)
        
            z = self.model(x_in, t_batch, cond_info).permute(0, 3, 2, 1)
            v_pred = z[..., :2]
            
            sqrt_ah_t = torch.sqrt(ah_t)
            sqrt_o_t = torch.sqrt(1 - ah_t)
            x0_pred = sqrt_ah_t * x - sqrt_o_t * v_pred
            eps_pred = (x - sqrt_ah_t * x0_pred) / sqrt_o_t

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

        rel_norm = x.view(num_samples, B, T, N, D)
        return rel_norm