import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.lstm_model import DefenseTrajectoryPredictorLSTM
from make_dataset import CustomDataset, organize_and_process, ApplyAugmentedDataset
from utils.utils import set_everything, worker_init_fn, generator, plot_trajectories_on_pitch, calc_frechet_distance
from utils.data_utils import split_dataset_indices, compute_train_zscore_stats, custom_collate_fn

# SEED Fix
SEED = 42
set_everything(SEED)

# Save Log / Logger Setting
model_save_path = './results/logs/'
os.makedirs(model_save_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=os.path.join(model_save_path, 'lstm_absolute_train.log'),
    filemode='w'
)
logger = logging.getLogger()

# 1. Model Config & Hyperparameter Setting
lstm_config = {
    "input_dim": 22,  # 11 defenders * 2 coordinates (x, y)
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.5
}

hyperparams = {
    'raw_data_path': "idsse-data",
    'data_save_path': "match_data",
    'train_batch_size': 16,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'num_workers': 8,
    'epochs': 30,
    'learning_rate': 1e-4,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    **lstm_config
}

raw_data_path = hyperparams['raw_data_path']
data_save_path = hyperparams['data_save_path']
train_batch_size = hyperparams['train_batch_size']
val_batch_size = hyperparams['val_batch_size']
test_batch_size = hyperparams['test_batch_size']
num_workers = hyperparams['num_workers']
epochs = hyperparams['epochs']
learning_rate = hyperparams['learning_rate']
device = hyperparams['device']

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
    ApplyAugmentedDataset(Subset(dataset, train_idx)),
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
model = DefenseTrajectoryPredictorLSTM(**lstm_config).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, threshold=1e-5, min_lr=learning_rate*0.01)

logger.info(f"Device: {device}")
logger.info(f"LSTM Model (Absolute Coords): {model}")

# 4. Train
best_val_loss = float("inf")
best_model_path = None
timestamp = datetime.now().strftime('%m%d')

train_losses = []
val_losses = []

for epoch in tqdm(range(1, epochs + 1), desc="Training...", leave=True):
    model.train()
    
    train_loss = 0

    for batch in tqdm(train_dataloader, desc="Batch Training...", leave=False):
        # Extract defender coordinates from condition
        condition = batch["condition"].to(device)  # [B, T_cond, F]
        condition_columns = batch["condition_columns"][0]
        target_columns = batch["target_columns"][0]
        
        # Extract defender absolute coordinates from condition
        B, T_cond, _ = condition.shape
        past_abs = torch.zeros(B, T_cond, 22, device=condition.device)
        
        # Create mapping from condition columns to indices
        col_to_idx = {col: i for i, col in enumerate(condition_columns)}
        
        # Extract defender coordinates
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]     # e.g., 'Away_15_x'  
            y_col = target_columns[i + 1] # e.g., 'Away_15_y'
            
            if x_col in col_to_idx and y_col in col_to_idx:
                x_idx = col_to_idx[x_col]
                y_idx = col_to_idx[y_col]
                
                past_abs[:, :, i] = condition[:, :, x_idx]     # x coordinate
                past_abs[:, :, i + 1] = condition[:, :, y_idx] # y coordinate
        
        # Target is already in absolute coordinates
        target_abs = batch["target"].to(device)  # [B, T_target, 22]
        
        # Predict absolute coordinates
        pred_abs = model(past_abs)  # [B, T_target, 22]
        
        loss = criterion(pred_abs, target_abs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del condition, past_abs, target_abs, pred_abs, loss

    num_batches = len(train_dataloader)
    avg_train_loss = train_loss / num_batches

    # --- Validation ---
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            # Extract defender coordinates from condition
            condition = batch["condition"].to(device)  # [B, T_cond, F]
            condition_columns = batch["condition_columns"][0]
            target_columns = batch["target_columns"][0]
            
            # Extract defender absolute coordinates from condition
            B, T_cond, _ = condition.shape
            past_abs = torch.zeros(B, T_cond, 22, device=condition.device)
            
            # Create mapping from condition columns to indices
            col_to_idx = {col: i for i, col in enumerate(condition_columns)}
            
            # Extract defender coordinates
            for i in range(0, len(target_columns), 2):
                x_col = target_columns[i]     # e.g., 'Away_15_x'  
                y_col = target_columns[i + 1] # e.g., 'Away_15_y'
                
                if x_col in col_to_idx and y_col in col_to_idx:
                    x_idx = col_to_idx[x_col]
                    y_idx = col_to_idx[y_col]
                    
                    past_abs[:, :, i] = condition[:, :, x_idx]     # x coordinate
                    past_abs[:, :, i + 1] = condition[:, :, y_idx] # y coordinate
            
            # Target is already in absolute coordinates
            target_abs = batch["target"].to(device)  # [B, T_target, 22]
            
            # Predict absolute coordinates
            pred_abs = model(past_abs)  # [B, T_target, 22]
            
            loss = criterion(pred_abs, target_abs)
            val_loss += loss.item()

            del condition, past_abs, target_abs, pred_abs, loss
            
    avg_val_loss = val_loss / len(val_dataloader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"[Epoch {epoch}/{epochs}] Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | LR={current_lr:.6e}")

    tqdm.write(f"[Epoch {epoch}]\n"
               f"[Train] MSE Loss: {avg_train_loss:.6f} | LR: {current_lr:.6f}\n"
               f"[Validation] MSE Loss: {avg_val_loss:.6f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        best_model_path = os.path.join(model_save_path, f'{timestamp}_best_lstm_absolute_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'zscore_stats': zscore_stats,
            'hyperparams': hyperparams
        }, best_model_path)
    
    torch.cuda.empty_cache()
    gc.collect()

logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

if epoch == epochs:
    for loader in (train_dataloader, val_dataloader):
        ds = loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if hasattr(ds, "graph_cache"):
            ds.graph_cache.clear()

# 4-1. Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"LSTM Baseline (Absolute Coords) - Train & Validation Loss\n"
          f"Hidden dim: {lstm_config['hidden_dim']}, Layers: {lstm_config['num_layers']}")
plt.legend()
plt.tight_layout()
plt.savefig(f'results/{timestamp}_lstm_absolute_lr_curve.png')
plt.show()
plt.close()

# 5. Inference & Visualization
if best_model_path and os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

model.eval()

all_ades = []
all_fdes = []
all_frechet_dist = []
all_DE = []

px_mean = torch.tensor(zscore_stats['player_x_mean'], device=device)
px_std = torch.tensor(zscore_stats['player_x_std'], device=device)
py_mean = torch.tensor(zscore_stats['player_y_mean'], device=device)
py_std = torch.tensor(zscore_stats['player_y_std'], device=device)

bx_mean = torch.tensor(zscore_stats['ball_x_mean'], device=device)
bx_std = torch.tensor(zscore_stats['ball_x_std'], device=device)
by_mean = torch.tensor(zscore_stats['ball_y_mean'], device=device)
by_std = torch.tensor(zscore_stats['ball_y_std'], device=device)

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Test Inference", leave=True)):
        # Extract defender coordinates from condition
        condition = batch["condition"].to(device)  # [B, T_cond, F]
        condition_columns = batch["condition_columns"][0]
        target_columns = batch["target_columns"][0]
        
        # Extract defender absolute coordinates from condition
        B, T_cond, _ = condition.shape
        past_abs = torch.zeros(B, T_cond, 22, device=condition.device)
        
        # Create mapping from condition columns to indices
        col_to_idx = {col: i for i, col in enumerate(condition_columns)}
        
        # Extract defender coordinates
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]     # e.g., 'Away_15_x'  
            y_col = target_columns[i + 1] # e.g., 'Away_15_y'
            
            if x_col in col_to_idx and y_col in col_to_idx:
                x_idx = col_to_idx[x_col]
                y_idx = col_to_idx[y_col]
                
                past_abs[:, :, i] = condition[:, :, x_idx]     # x coordinate
                past_abs[:, :, i + 1] = condition[:, :, y_idx] # y coordinate
        
        # Target is already in absolute coordinates
        target_abs = batch["target"].to(device)  # [B, T_target, 22]
        
        B, T_cond, _ = past_abs.shape
        _, T_target, _ = target_abs.shape
        
        # Predict absolute coordinates
        pred_abs = model(past_abs)  # [B, T_target, 22]
        
        # Reshape for evaluation: [B, T, 11, 2]
        pred_abs = pred_abs.view(B, T_target, 11, 2)
        target_abs = target_abs.view(B, T_target, 11, 2)
        
        # Denormalize coordinates for evaluation
        pred_abs_denorm = pred_abs.clone()
        pred_abs_denorm[..., 0] = pred_abs[..., 0] * px_std + px_mean
        pred_abs_denorm[..., 1] = pred_abs[..., 1] * py_std + py_mean
        
        target_abs_denorm = target_abs.clone()
        target_abs_denorm[..., 0] = target_abs[..., 0] * px_std + px_mean
        target_abs_denorm[..., 1] = target_abs[..., 1] * py_std + py_mean

        # Calculate ADE and FDE
        ade = ((pred_abs_denorm - target_abs_denorm)**2).sum(-1).sqrt().mean((1,2))  # [B]
        fde = ((pred_abs_denorm[:,-1,:,:] - target_abs_denorm[:,-1,:,:])**2).sum(-1).sqrt().mean(1)  # [B]

        # Calculate DE (Direction Error: overall movement direction)
        eps = 1e-6
        overall_pred = pred_abs_denorm[:, -1] - pred_abs_denorm[:, 0]
        overall_gt = target_abs_denorm[:, -1] - target_abs_denorm[:, 0]
        
        norm_pred = overall_pred.norm(dim=-1, keepdim=True).clamp(min=eps)  # [B, N, 1]
        norm_gt = overall_gt.norm(dim=-1, keepdim=True).clamp(min=eps)  # [B, N, 1]
        
        u = overall_pred / norm_pred
        v = overall_gt / norm_gt
        
        cosine = (u * v).sum(dim=-1).clamp(-1.0, 1.0)
        theta = cosine.acos()
        DE = theta.mean(dim=1)

        all_ades.extend(ade.cpu().tolist())
        all_fdes.extend(fde.cpu().tolist())
        all_DE.extend(DE.cpu().tolist())
        
        # Calculate Fréchet distance
        pred_np = pred_abs_denorm.cpu().numpy()      # [B,T,N,2]
        target_np = target_abs_denorm.cpu().numpy()
        B_, T, N, _ = pred_np.shape
        batch_frechet = []
        for b in range(B_):
            per_player_frechet = []
            for j in range(N):
                pred_traj = pred_np[b, :, j, :]
                target_traj = target_np[b, :, j, :]
                frechet_dist = calc_frechet_distance(pred_traj, target_traj)
                per_player_frechet.append(frechet_dist)
            batch_frechet.append(np.mean(per_player_frechet))
        
        all_frechet_dist.extend(batch_frechet)
        
        # Debug print
        print(f"[Batch {batch_idx}] "
              f"ADE={ade.mean():.3f}, FDE={fde.mean():.3f}, "
              f"Frechet={np.mean(batch_frechet):.3f}, DE={torch.rad2deg(DE.mean()):.2f}°")

        # Visualization
        base_dir = f"results/{timestamp}_test_trajs_lstm_absolute"
        os.makedirs(base_dir, exist_ok=True)

        for i in range(B):
            other_cols = batch["other_columns"][i]
            target_cols = batch["target_columns"][i]
            
            # Other Trajectories Denormalization
            other_seq = batch["other"][i].view(T_target, -1, 2).to(device)
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

            pred_traj = pred_abs_denorm[i].cpu().numpy()
            target_traj = target_abs_denorm[i].cpu().numpy()
            other_traj = other_den.cpu().numpy()

            current_ade = ade[i].item()
            current_fde = fde[i].item()
            current_frechet = batch_frechet[i]

            defender_nums = [int(col.split('_')[1]) for col in target_cols[::2]]

            folder = os.path.join(base_dir, f"batch_{batch_idx:03d}")
            os.makedirs(folder, exist_ok=True)

            save_path = os.path.join(folder, f"sample_{i:02d}.png")
            plot_trajectories_on_pitch(
                other_traj, target_traj, pred_traj, other_columns=other_cols, 
                defenders_num=defender_nums, annotate=True, save_path=save_path
            )

        del condition, past_abs, pred_abs, target_abs
        del pred_abs_denorm, target_abs_denorm, ade, fde, DE
        torch.cuda.empty_cache()
        gc.collect()

print(f"LSTM Baseline (Absolute Coordinates)")
print(f"ADE: {np.mean(all_ades):.3f} ± {np.std(all_ades):.3f} meters")
print(f"FDE: {np.mean(all_fdes):.3f} ± {np.std(all_fdes):.3f} meters")
print(f"Fréchet: {np.mean(all_frechet_dist):.3f} ± {np.std(all_frechet_dist):.3f} meters")
print(f"Direction Error (DE): {np.rad2deg(np.mean(all_DE)):.2f} ± {np.rad2deg(np.std(all_DE)):.2f}°")