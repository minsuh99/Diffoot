import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from models.diff_modules import diff_CSDI
from models.diff_model import DiffusionTrajectoryModel
from models.encoder import InteractionGraphEncoder, TargetTrajectoryEncoder
from make_dataset import CustomDataset, organize_and_process
from utils.utils import set_everything, worker_init_fn, generator, plot_trajectories_on_pitch, log_graph_stats
from utils.data_utils import split_dataset_indices, compute_train_zscore_stats, custom_collate_fn
from utils.graph_utils import build_graph_sequence_from_condition

# SEED Fix
SEED = 42
set_everything(SEED)

# Save Log / Logger Setting
model_save_path = './results/logs/'
os.makedirs(model_save_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=os.path.join(model_save_path, 'train.log'),
    filemode='w'
)
logger = logging.getLogger()

# 1. Model Config & Hyperparameter Setting
csdi_config = {
    "num_steps": 1000,
    "channels": 512,
    "diffusion_embedding_dim": 512,
    "nheads": 8,
    "layers": 10,
    "side_dim": 512,
    
    "time_seq_len": 50,      # Target 시간 길이
    "feature_seq_len": 11,    # 선수 수
    "compressed_dim": 32     # 압축 차원
}
hyperparams = {
    'raw_data_path': "idsse-data", # raw_data_path = "Download raw file path"
    'data_save_path': "match_data",
    'train_batch_size': 16,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'num_workers': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'self_conditioning_ratio': 0.0,
    'num_samples': 10,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',

    'ddim_step': 50,
    'eta': 0.0,
    **csdi_config
}
raw_data_path = hyperparams['raw_data_path']
data_save_path = hyperparams['data_save_path']
train_batch_size = hyperparams['train_batch_size']
val_batch_size = hyperparams['val_batch_size']
test_batch_size = hyperparams['test_batch_size']
num_workers = hyperparams['num_workers']
epochs = hyperparams['epochs']
learning_rate = hyperparams['learning_rate']
self_conditioning_ratio = hyperparams['self_conditioning_ratio']
num_samples = hyperparams['num_samples']
device = hyperparams['device']
ddim_step = hyperparams['ddim_step']
eta = hyperparams['eta']
side_dim = hyperparams['side_dim']

logger.info(f"Hyperparameters: {hyperparams}")

# 2. Data Loading
print("---Data Loading---")
if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
    organize_and_process(raw_data_path, data_save_path)
else:
    print("Skip organize_and_process")

temp_dataset = CustomDataset(data_root=data_save_path)
train_idx, val_idx, test_idx = split_dataset_indices(temp_dataset, val_ratio=1/6, test_ratio=1/6, random_seed=SEED)

zscore_stats = compute_train_zscore_stats(temp_dataset, train_idx, save_path="./train_zscore_stats.pkl")
del temp_dataset
gc.collect()
dataset = CustomDataset(data_root=data_save_path, zscore_stats=zscore_stats)

train_dataloader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
    generator=generator(SEED)
)

val_dataloader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
)

test_dataloader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")
print(f"Train: {len(train_dataloader.dataset)} | Val: {len(val_dataloader.dataset)} | Test: {len(test_dataloader.dataset)}")

# 3. Model Define
# Extract node feature dimension
sample = dataset[0]
graph = build_graph_sequence_from_condition({
    "condition": sample["condition"],
    "condition_columns": sample["condition_columns"],
    "pitch_scale": sample["pitch_scale"],
    "zscore_stats": zscore_stats
}).to(device)

log_graph_stats(graph, logger, prefix="InitGraphSample")

in_dim = graph['Node'].x.size(1)

# history trajectories from condition
condition_columns = sample["condition_columns"]
target_idx = [i for i, col in enumerate(condition_columns) if col.endswith(("_x", "_y"))]

# graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=128, out_dim=128, heads = 2).to(device)
graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=side_dim // 2, out_dim=side_dim // 2).to(device)
history_encoder = TargetTrajectoryEncoder(num_layers=5, hidden_dim = side_dim // 4, bidirectional=True).to(device)
denoiser = diff_CSDI(csdi_config)
diff_model = DiffusionTrajectoryModel(denoiser, num_steps=csdi_config["num_steps"]).to(device)

optimizer = torch.optim.AdamW(list(diff_model.parameters()) + list(graph_encoder.parameters()) + list(history_encoder.parameters()), lr=learning_rate, weight_decay=0.0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, threshold=1e-5, min_lr=learning_rate*0.01)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs * 0.7, eta_min=learning_rate * 0.01)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = int(epochs * 0.5), T_mult = 2, eta_min=learning_rate * 0.1)
scaler = GradScaler()

logger.info(f"Device: {device}")
logger.info(f"GraphEncoder: {graph_encoder}")
logger.info(f"HistoryEncoder: {history_encoder}")
logger.info(f"Denoiser (diff_CSDI): {denoiser}")
logger.info(f"DiffusionTrajectoryModel: {diff_model}")

# 4. Train
best_state_dict = None
best_val_loss = float("inf")

train_losses = []
val_losses   = []

first_batch_debug = True

for epoch in tqdm(range(1, epochs + 1), desc="Training...", leave=True):
    diff_model.train()
    graph_encoder.train()
    history_encoder.train()
    
    train_noise_mse = 0
    train_noise_nll = 0
    train_player_mse = 0
    train_loss = 0

    for batch in tqdm(train_dataloader, desc = "Batch Training...", leave=False):
        optimizer.zero_grad()
        cond = batch["condition"].to(device)
        B, T_cond, _ = cond.shape
        _, T_target, _ = batch["target"].shape
        target_columns = batch["target_columns"][0]
        condition_columns = batch["condition_columns"][0]
        
        target_x_indices = []
        target_y_indices = []
        
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]
            y_col = target_columns[i + 1]
            
            if x_col in condition_columns and y_col in condition_columns:
                target_x_indices.append(condition_columns.index(x_col))
                target_y_indices.append(condition_columns.index(y_col))

        last_past_cond = cond[:, -1]
        # initial_pos: [B, 11, 2] - 각 수비수의 마지막 위치
        # initial_pos = torch.stack([
        #     last_past_cond[:, target_x_indices],  # [B, 11]
        #     last_past_cond[:, target_y_indices]   # [B, 11]
        # ], dim=-1)  # [B, 11, 2]

        target_abs = batch["target"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
        target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
        target_vel = batch["target_velocity"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]

        model_input = torch.cat([target_abs, target_rel, target_vel], dim=-1).to(device)  # [B, T, 11, 6]
        graph_batch = batch["graph"].to(device)    
        # graph → H
        H = graph_encoder(graph_batch)                                       # [B, 256]
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        
        # All player + ball history trajectories
        hist = cond[:, :, target_idx].to(device) 
        hist_rep = history_encoder(hist)  # [B, 256]
        cond_hist = hist_rep.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        
        # Concat conditions
        cond_info = torch.cat([cond_H, cond_hist], dim=1)
        # Preparing Self-conditioning data
        # timestep (consistency)
        t = torch.randint(0, diff_model.num_steps, (model_input.size(0),), device=device)
        
        s = None
        if torch.rand(1, device=device) < self_conditioning_ratio:
            with torch.no_grad():
                # 첫 번째 denoising step으로 x0 예측
                x_t, noise = diff_model.q_sample(model_input, t)
                x_t_input = x_t.permute(0, 3, 2, 1)
                
                # 첫 번째 예측 (self_cond=None)
                z1 = diff_model.model(x_t_input, t, cond_info, self_cond=None)
                eps_pred1 = z1[:, :6, :, :]
                
                # x0 예측값 계산 (다음 step의 self-conditioning input)
                a_hat = diff_model.alpha_hat[t].view(-1, 1, 1, 1)
                x0_hat = (x_t_input - (1 - a_hat).sqrt() * eps_pred1) / a_hat.sqrt()
                x0_hat = x0_hat.permute(0, 3, 2, 1)
                
                s = x0_hat.detach()  # [B, T, 11, 6]
                
                del x_t, noise, x_t_input, z1, eps_pred1, a_hat, x0_hat

        noise_mse, noise_nll = diff_model(model_input, t=t, cond_info=cond_info, self_cond=s)
        loss = noise_mse + noise_nll * 0.001
            
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_noise_mse += (noise_mse).item()
        train_noise_nll += (noise_nll * 0.001).item()
        # train_player_mse += (player_mse * 0.05).item()

        train_loss += loss.item()

        del cond, last_past_cond, target_abs, target_rel, target_vel, graph_batch, H, cond_H, hist, hist_rep, cond_hist
        del cond_info, t, s, noise_mse, noise_nll

    num_batches = len(train_dataloader)
    
    avg_train_noise_mse = train_noise_mse / num_batches
    avg_train_noise_nll = train_noise_nll / num_batches
    # avg_train_player_mse = train_player_mse / num_batches
    avg_train_loss = train_loss / num_batches


    # --- Validation ---
    diff_model.eval()
    graph_encoder.eval()
    history_encoder.eval()
    
    val_noise_mse = 0
    val_noise_nll = 0
    val_player_mse = 0
    val_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            cond = batch["condition"].to(device)
            B, T_cond, _ = cond.shape
            _, T_target, _ = batch["target"].shape
            target_columns = batch["target_columns"][0]
            condition_columns = batch["condition_columns"][0]
            
            target_x_indices = []
            target_y_indices = []
            
            for i in range(0, len(target_columns), 2):
                x_col = target_columns[i]
                y_col = target_columns[i + 1]
                
                if x_col in condition_columns and y_col in condition_columns:
                    target_x_indices.append(condition_columns.index(x_col))
                    target_y_indices.append(condition_columns.index(y_col))
            
            last_past_cond = cond[:, -1]
            # initial_pos: [B, 11, 2] - 각 수비수의 마지막 위치
            # initial_pos = torch.stack([
            #     last_past_cond[:, target_x_indices],  # [B, 11]
            #     last_past_cond[:, target_y_indices]   # [B, 11]
            # ], dim=-1)  # [B, 11, 2]
            
            target_abs = batch["target"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
            target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
            target_vel = batch["target_velocity"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]

            model_input = torch.cat([target_abs, target_rel, target_vel], dim=-1).to(device)  # [B, T, 11, 6]
            graph_batch = batch["graph"].to(device)                        # HeteroData batch

            # graph → H
            H = graph_encoder(graph_batch)                                      # [B, 256]
            cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
            
            # Target's history trajectories
            hist = cond[:, :, target_idx].to(device)  # [B,256,11,T]
            hist_rep = history_encoder(hist)  # [B, 256]
            cond_hist = hist_rep.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
            
            # Concat conditions
            cond_info = torch.cat([cond_H, cond_hist], dim=1)
            
            t = torch.randint(0, diff_model.num_steps, (B,), device=device)
            
            s = None
            if torch.rand(1, device=device) < self_conditioning_ratio:
                # 첫 번째 denoising step으로 x0 예측
                x_t, noise = diff_model.q_sample(model_input, t)
                x_t_input = x_t.permute(0, 3, 2, 1)
                
                # 첫 번째 예측 (self_cond=None)
                z1 = diff_model.model(x_t_input, t, cond_info, self_cond=None)
                eps_pred1 = z1[:, :6, :, :]
                
                # x0 예측값 계산 (다음 step의 self-conditioning input)
                a_hat = diff_model.alpha_hat[t].view(-1, 1, 1, 1)
                x0_hat = (x_t_input - (1 - a_hat).sqrt() * eps_pred1) / a_hat.sqrt()
                x0_hat = x0_hat.permute(0, 3, 2, 1)
                
                s = x0_hat.detach() # [B, T, 11, 6]
                
                del x_t, noise, x_t_input, z1, eps_pred1, a_hat, x0_hat

            noise_mse, noise_nll = diff_model(model_input, t=t, cond_info=cond_info, self_cond=s)
            val_loss = noise_mse + noise_nll * 0.001

            val_noise_mse += (noise_mse).item()
            val_noise_nll += (noise_nll * 0.001).item()
            # val_player_mse += (player_mse * 0.05).item()
            val_total_loss += val_loss.item()

        del cond, last_past_cond, target_abs, target_rel, target_vel, graph_batch, H, cond_H, hist, hist_rep, cond_hist
        del cond_info, t, s, noise_mse, noise_nll

    num_batches = len(val_dataloader)

    avg_val_noise_mse = val_noise_mse / num_batches
    avg_val_noise_nll = val_noise_nll / num_batches
    # avg_val_player_mse = val_player_mse / num_batches
    avg_val_loss = val_total_loss / num_batches

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"[Epoch {epoch}/{epochs}] Train Loss={avg_train_loss:.6f} (Noise simple={avg_train_noise_mse:.6f}, Noise NLL={avg_train_noise_nll:.6f}) | "
                f"Val Loss={avg_val_loss:.6f} (Noise simple={avg_val_noise_mse:.6f}, Noise NLL={avg_val_noise_nll:.6f}) | LR={current_lr:.6e}")

    tqdm.write(f"[Epoch {epoch}]\n"
               f"[Train] Cost: {avg_train_loss:.6f} | Noise Loss: {avg_train_noise_mse:.6f} | NLL Loss: {avg_train_noise_nll:.6f} | LR: {current_lr:.6f}\n"
               f"[Validation] Val Loss: {avg_val_loss:.6f} | Noise Loss: {avg_val_noise_mse:.6f} | NLL Loss: {avg_val_noise_nll:.6f}")

    scheduler.step(avg_val_loss)
    # scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state_dict = {
            'diff_model': {k: v.cpu().clone() for k, v in diff_model.state_dict().items()},
            'graph_encoder': {k: v.cpu().clone() for k, v in graph_encoder.state_dict().items()},
            'history_encoder': {k: v.cpu().clone() for k, v in history_encoder.state_dict().items()},
            'zscore_stats': zscore_stats
        }
    
    torch.cuda.empty_cache()
    gc.collect()

logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

if epoch == epochs:
    for loader in (train_dataloader, val_dataloader):
        ds = loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if hasattr(ds, "graph_cache"):
            ds.graph_cache.clear()
            
# 4-1. Plot learning_curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Train & Validation Loss, {csdi_config['num_steps']} steps, {csdi_config['channels']} channels,\n"
          f"{csdi_config['diffusion_embedding_dim']} embedding dim, {csdi_config['nheads']} heads, {csdi_config['layers']} layers "
          f"self-conditioning ratio: {self_conditioning_ratio}")
plt.legend()
plt.tight_layout()

# plt.savefig('results/0603_diffusion_lr_curve.png')
timestamp = datetime.now().strftime('%m%d')
plt.savefig(f'results/{timestamp}_diffusion_lr_curve.png')

plt.show()
plt.close()

final_state = {
    'diff_model': {k: v.cpu().clone() for k, v in diff_model.state_dict().items()},
    'graph_encoder': {k: v.cpu().clone() for k, v in graph_encoder.state_dict().items()},
    'history_encoder': {k: v.cpu().clone() for k, v in history_encoder.state_dict().items()},
    'zscore_stats': zscore_stats,
    'train_losses': train_losses,
    'val_losses': val_losses
}

torch.save(final_state, './results/final_model_latest.pth')

# 5. Inference (Best-of-N Sampling) & Visualization
diff_model.load_state_dict(best_state_dict['diff_model'])
graph_encoder.load_state_dict(best_state_dict['graph_encoder'])
history_encoder.load_state_dict(best_state_dict['history_encoder'])

diff_model.eval()
graph_encoder.eval()
history_encoder.eval()

all_ades = []
all_fdes = []

visualize_samples = 5
visualized = False # If you want to visualize
# visualized = True

px_mean = torch.tensor(zscore_stats['player_x_mean'], device=device)
px_std = torch.tensor(zscore_stats['player_x_std'], device=device)
py_mean = torch.tensor(zscore_stats['player_y_mean'], device=device)
py_std = torch.tensor(zscore_stats['player_y_std'],  device=device)

bx_mean = torch.tensor(zscore_stats['ball_x_mean'], device=device)
bx_std = torch.tensor(zscore_stats['ball_x_std'], device=device)
by_mean = torch.tensor(zscore_stats['ball_y_mean'], device=device)
by_std = torch.tensor(zscore_stats['ball_y_std'], device=device)

rel_x_mean = torch.tensor(zscore_stats['rel_x_mean'], device=device)
rel_x_std = torch.tensor(zscore_stats['rel_x_std'], device=device)
rel_y_mean = torch.tensor(zscore_stats['rel_y_mean'], device=device)
rel_y_std = torch.tensor(zscore_stats['rel_y_std'], device=device)

with torch.no_grad():        
    for batch in tqdm(test_dataloader, desc="Test Streaming Inference", leave=True):
        cond = batch["condition"].to(device)
        B, T_cond, _ = cond.shape
        _, T_target, _ = batch["target"].shape
        target_columns = batch["target_columns"][0]
        condition_columns = batch["condition_columns"][0]
        
        target_x_indices = []
        target_y_indices = []
        
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]
            y_col = target_columns[i + 1]
            
            if x_col in condition_columns and y_col in condition_columns:
                target_x_indices.append(condition_columns.index(x_col))
                target_y_indices.append(condition_columns.index(y_col))
        
        last_past_cond = cond[:, -1]
        # initial_pos: [B, 11, 2] - 각 수비수의 마지막 위치
        initial_pos = torch.stack([
            last_past_cond[:, target_x_indices],  # [B, 11]
            last_past_cond[:, target_y_indices]   # [B, 11]
        ], dim=-1)  # [B, 11, 2]
        
        target_abs = batch["target"].to(device).view(-1, T_target, 11, 2)  # [B, T_target, 11, 2]
        target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T_target, 11, 2]
        target_vel = batch["target_velocity"].to(device).view(-1, T_target, 11, 2)  # [B, T_target, 11, 2]

        model_input = torch.cat([target_abs, target_rel, target_vel], dim=-1).to(device)  # [B, T_target, 11, 6]
        graph_batch = batch["graph"].to(device)

        # condition 준비
        H = graph_encoder(batch["graph"].to(device))
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        
        hist = cond[:, :, target_idx].to(device)
        hist_rep = history_encoder(hist)
        cond_hist = hist_rep.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        cond_info = torch.cat([cond_H, cond_hist], dim=1)  # (B, C, 11, T)
        
        # Target Denormalization
        target_den = target_abs.clone()
        target_den[...,0] = target_abs[...,0] * px_std + px_mean
        target_den[...,1] = target_abs[...,1] * py_std + py_mean

        pred = diff_model.generate(shape=model_input.shape, cond_info=cond_info, ddim_steps=ddim_step, eta=eta, num_samples=1)[0]  # (B, T, 11, 6)

        pred_rel_norm = pred[..., 2:4]  # 정규화된 상대좌표
        pred_rel_raw = pred_rel_norm * torch.stack([rel_x_std, rel_y_std]) + torch.stack([rel_x_mean, rel_y_mean])

        initial_pos_raw = initial_pos * torch.stack([px_std, py_std]) + torch.stack([px_mean, py_mean])

        pred_abs_raw = pred_rel_raw + initial_pos_raw.unsqueeze(1)
        pred_den = pred_abs_raw
        
        # ADE, FDE Calculation (Meters)
        ade = ((pred_den[...,:2] - target_den[...,:2])**2).sum(-1).sqrt().mean((1,2))
        fde = ((pred_den[:,-1,:,:2] - target_den[:,-1,:,:2])**2).sum(-1).sqrt().mean(1)
            
        all_ades.extend(ade.cpu().tolist())
        all_fdes.extend(fde.cpu().tolist())
        
        # Visualization (1st batch)
        if not visualized:
            base_dir = "results/test_trajs"
            os.makedirs(base_dir, exist_ok=True)

            for i in range(min(B, visualize_samples)):
                other_cols = batch["other_columns"][i]
                target_cols = batch["target_columns"][i]
                
                # Other Trajectories Denormalization
                other_seq = batch["other"][i].view(T_target, -1, 2).to(device)          # [T, 12, 2]
                other_den = torch.zeros_like(other_seq)
                for j in range(other_seq.size(1)):
                    x_col = other_cols[2 * j]
                    if x_col == "ball_x":
                        x_mean, x_std = bx_mean, bx_std
                        y_mean, y_std = by_mean, by_std
                    else:
                        x_mean, x_std = px_mean, px_std
                        y_mean, y_std = py_mean, py_std

                    other_den[:, j, 0] = other_seq[:, j, 0] * x_std + x_mean
                    other_den[:, j, 1] = other_seq[:, j, 1] * y_std + y_mean
                
                pred_traj = pred_den[i, :, :, :2].cpu().numpy()
                target_traj = target_den[i, :, :, :2].cpu().numpy()
                other_traj  = other_den.cpu().numpy()
                
                folder = os.path.join(base_dir, f"sample{i:02d}")
                os.makedirs(folder, exist_ok=True)
                
                defender_nums = [int(col.split('_')[1]) for col in target_cols[::2]]
                for idx, jersey in enumerate(defender_nums):
                    save_path = os.path.join(folder, f"player_{jersey:02d}.png")
                    plot_trajectories_on_pitch(
                        other_traj, target_traj, pred_traj,
                        other_columns=other_cols, target_columns=target_cols, player_idx=idx,
                        annotate=True, save_path=save_path
                    )

            visualized = True
        
        del pred, pred_den, target_den, ade, fde
        del cond, target_abs, target_rel, target_vel, H, cond_H, hist, hist_rep, cond_hist, cond_info, initial_pos
        torch.cuda.empty_cache()
        gc.collect()
            
print(f"Best-of-{num_samples} Sampling:")
print(f"ADE: {np.mean(all_ades):.3f} ± {np.std(all_ades):.3f} meters")
print(f"FDE: {np.mean(all_fdes):.3f} ± {np.std(all_fdes):.3f} meters")