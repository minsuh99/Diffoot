import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionTrajectoryModel(nn.Module):
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

        ts = torch.linspace(0, 1, num_steps)
        betas = beta_start + (beta_end - beta_start) * (ts ** 2)
        
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_hat = alpha_hat

    def q_sample(self, x_0, t, noise=None):
        device = x_0.device
        alpha_hat = self.alpha_hat.to(device)
        
        if noise is None:
            noise = torch.randn_like(x_0)
        t = t.to(device)
        a_hat = alpha_hat[t].view(-1, 1, 1, 1)
        one_minus = 1.0 - a_hat
        x_t = torch.sqrt(a_hat) * x_0 + torch.sqrt(one_minus) * noise
        return x_t, noise

    def forward(self, x_0, cond_info=None, self_cond=None):
        device = x_0.device
        B = x_0.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # x_0 -> x_t
        x_t, noise = self.q_sample(x_0, t)
        x_t = x_t.permute(0, 3, 2, 1)   # [B, 2, 11, T]
        noise = noise.permute(0, 3, 2, 1)

        if cond_info is not None:
            cond_info = cond_info.to(device)

        # predict noise
        noise_pred = self.model(x_t, t, cond_info, self_cond)
        noise_loss = F.mse_loss(noise_pred, noise)

        # x_t -> x_0
        alpha_hat = self.alpha_hat.to(device)
        a_hat = alpha_hat[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - torch.sqrt(1.0 - a_hat) * noise_pred) / torch.sqrt(a_hat)
        x_0_pred = x_0_pred.permute(0, 3, 2, 1)  # [B, T, 11, 2]

        x_0 = x_0.to(device)

        player_loss = F.mse_loss(x_0_pred, x_0, reduction='none')
        player_loss = player_loss.mean(dim=[1, 3]).mean()

        return noise_loss, player_loss

    @torch.no_grad()
    def generate(self, shape, cond_info=None, num_samples=10):
        B, T, N, D = shape
        device = next(self.parameters()).device
        
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alpha_hat = self.alpha_hat.to(device)

        if cond_info is not None:
            cond_info = cond_info.to(device)
            cond_info = cond_info.repeat(num_samples, 1, 1, 1)

        x_t = torch.randn((num_samples * B, T, N, D), device=device)
        x_t = x_t.permute(0, 3, 2, 1)  # [N*B, 2, 11, T]
        s = torch.zeros_like(x_t) # Self-Conditioning initialization

        for t_i in reversed(range(self.num_steps)):
            t_tensor = torch.full((num_samples * B,), t_i, device=device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor, cond_info, self_cond = s)

            a = alphas[t_i]
            a_hat = alpha_hat[t_i]
            b = betas[t_i]

            if t_i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            x_prev = (1 / torch.sqrt(a)) * (x_t - ((1 - a) / torch.sqrt(1 - a_hat)) * noise_pred) + torch.sqrt(b) * noise
            
            s = x_prev.clone()
            x_t = x_prev

        x_t = x_t.permute(0, 3, 2, 1)  # [N*B, T, 11, 2]
        x_t = x_t.view(num_samples, B, T, N, D)
        return x_t
