import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import per_player_frechet_loss, per_player_fde_loss

class DiffusionTrajectoryModel(nn.Module):
    # def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
    def __init__(self, model, num_steps=1000, s=0.008):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
    
        # ts = torch.linspace(0, 1, num_steps)
        # betas = beta_start + (beta_end - beta_start) * (ts ** 2)
        # alphas = 1.0 - betas
        # alpha_hat = torch.cumprod(alphas, dim=0)
    
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps, dtype=torch.float32) / num_steps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(min=0.0, max=0.999)

        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_hat', alpha_hat)
        
    def q_posterior(self, x0, xt, t):
        # compute posterior q(x_{t-1}|x_t, x0) parameters
        betas_t = self.betas[t].view(-1,1,1,1)
        a_hat_t = self.alpha_hat[t].view(-1,1,1,1)
        a_hat_prev = torch.cat([torch.tensor([1.0], device=x0.device), self.alpha_hat[:-1]])[t].view(-1,1,1,1)
        posterior_var = betas_t * (1 - a_hat_prev) / (1 - a_hat_t)
        posterior_mean = (
            torch.sqrt(a_hat_prev) * betas_t / (1 - a_hat_t) * x0 +
            torch.sqrt(self.alphas[t].view(-1,1,1,1)) * (1 - a_hat_prev) / (1 - a_hat_t) * xt
        )
        return posterior_mean, posterior_var

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

        z = self.model(x_t_in, t, cond_info, self_cond)
        z = z.permute(0, 3, 2, 1)
        
        eps_pred, log_var = z[..., :2], z[..., 2:]
        log_var = log_var.clamp(-5, 5)
        var = log_var.exp()
        
        noise_mse = F.mse_loss(eps_pred, noise)
        
        # NLL loss computing
        nll = 0.5 * ((noise - eps_pred) ** 2 / var + log_var + math.log(2 * math.pi))
        noise_nll = nll.mean()
        
        # VLB term L_t (KL between q and p_theta)
        post_mean, post_var = self.q_posterior(x_0, x_t, t)
        # p_theta has mean=predicted_mean, var=var
        pred_mean = torch.sqrt(self.alpha_hat[t].view(-1,1,1,1)) * x_0 - \
                     torch.sqrt(1 - self.alpha_hat[t].view(-1,1,1,1)) * eps_pred
        # KL divergence between two Gaussians
        kl = 0.5 * (
            torch.log(post_var) - log_var + post_var/var + (post_mean - pred_mean)**2/var - 1
        )
        vlb_loss = kl.mean()
        
        return noise_mse, vlb_loss
    
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
        final_x0 = None

        for i, t in enumerate(reversed(timesteps)):
            t_prev = 0 if i == ddim_steps - 1 else timesteps[-(i + 2)]

            ah_t = alpha_hat[t]
            ah_t_prev = alpha_hat[t_prev]
            
            t_batch = torch.full((num_samples * B,), t, device=device, dtype=torch.long)
            
            x_in = x.permute(0, 3, 2, 1)
            z = self.model(x_in, t_batch, cond_info, self_cond=None).permute(0, 3, 2, 1)
            noise_pred = z[..., :2]
            
            x0_pred = (x - torch.sqrt(1 - ah_t) * noise_pred) / torch.sqrt(ah_t)
            
            if t_prev == 0:
                final_x0 = x0_pred.view(num_samples, B, T, N, D)
            
            if eta > 0 and t_prev > 0:
                sigma_t = eta * torch.sqrt((1 - ah_t_prev) / (1 - ah_t)) * torch.sqrt(1 - ah_t)
            else:
                sigma_t = 0.0
                
            c = torch.sqrt(torch.clamp(1 - ah_t_prev - sigma_t**2, min=0.0))
                
            noise = torch.randn_like(x)  
              
            x = torch.sqrt(ah_t_prev) * x0_pred + c * noise_pred + sigma_t * noise

        return final_x0