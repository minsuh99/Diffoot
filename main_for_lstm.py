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
from torch.amp import autocast, GradScaler
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
    filename=os.path.join(model_save_path, 'lstm_train.log'),
    filemode='w'
)
logger = logging.getLogger()

# 1. Model Config & Hyperparameter Setting
lstm_config = {
    "input_dim": 202,
    "hidden_dim": 128,
    "num_layers": 2,
    "output_dim": 2200,
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

# 2. Data Loading (diffusion과 동일)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, threshold=1e-5, min_lr=learning_rate*0.01)
scaler = GradScaler()

logger.info(f"Device: {device}")
logger.info(f"LSTM Model: {model}")

# 4. Train
best_val_loss = float("inf")
best_model_path = None
timestamp = datetime.now().strftime('%m%d_%H%M%S')

train_losses = []
val_losses = []

for epoch in tqdm(range(1, epochs + 1), desc="Training...", leave=True):
    model.train()
    
    train_loss = 0
    num_batches = len(train_dataloader)

    for batch in tqdm(train_dataloader, desc="Batch Training...", leave=False):
        optimizer.zero_grad()
        
        with autocast('cuda'):
            condition = batch["condition"].to(device)
            target_relative = batch["target_relative"].to(device).view(-1, 100, 22)

            pred = model(condition)

            loss = criterion(pred, target_relative)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()

    avg_train_loss = train_loss / num_batches

    # --- Validation ---
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            with autocast('cuda'):
                condition = batch["condition"].to(device)
                target_relative = batch["target_relative"].to(device).view(-1, 100, 22)
                
                pred = model(condition)
                loss = criterion(pred, target_relative)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"[Epoch {epoch}/{epochs}] Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | LR={current_lr:.6e}")

    tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        
        best_model_path = os.path.join(model_save_path, f'{timestamp}_best_lstm_epoch_{epoch}.pth')
        
        torch.save({
            'model': model.state_dict(),
            'zscore_stats': zscore_stats,
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'epoch': epoch,
            'hyperparams': hyperparams
        }, best_model_path)
    
    torch.cuda.empty_cache()
    gc.collect()

logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

# 4-1. Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"LSTM Baseline - Train & Validation Loss\n"
          f"Hidden dim: {lstm_config['hidden_dim']}, Layers: {lstm_config['num_layers']}")
plt.legend()
plt.tight_layout()
plt.savefig(f'results/{timestamp}_lstm_lr_curve.png')
plt.show()
plt.close()

# 5. Inference
if best_model_path and os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

model.eval()

all_ades = []
all_fdes = []
all_frechet_dist = []

visualize_samples = 5
visualized = False  # If you want to visualize

px_mean = torch.tensor(zscore_stats['player_x_mean'], device=device)
px_std = torch.tensor(zscore_stats['player_x_std'], device=device)
py_mean = torch.tensor(zscore_stats['player_y_mean'], device=device)
py_std = torch.tensor(zscore_stats['player_y_std'], device=device)

bx_mean = torch.tensor(zscore_stats['ball_x_mean'], device=device)
bx_std = torch.tensor(zscore_stats['ball_x_std'], device=device)
by_mean = torch.tensor(zscore_stats['ball_y_mean'], device=device)
by_std = torch.tensor(zscore_stats['ball_y_std'], device=device)

rel_x_mean = torch.tensor(zscore_stats['rel_x_mean'], device=device)
rel_x_std = torch.tensor(zscore_stats['rel_x_std'], device=device)
rel_y_mean = torch.tensor(zscore_stats['rel_y_mean'], device=device)
rel_y_std = torch.tensor(zscore_stats['rel_y_std'], device=device)

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Test Inference", leave=True):
        condition = batch["condition"].to(device)
        
        with autocast('cuda'):
            B, T_cond, _ = condition.shape
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
            
            last_past_cond = condition[:, -1]
            # initial_pos: [B, 11, 2] - 각 수비수의 마지막 위치
            initial_pos = torch.stack([
                last_past_cond[:, target_x_indices],  # [B, 11]
                last_past_cond[:, target_y_indices]   # [B, 11]
            ], dim=-1)  # [B, 11, 2]
            
            target_abs = batch["target"].to(device).view(-1, T_target, 11, 2)
            target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)

            pred = model(condition)
            pred = pred.view(B, T_target, 11, 2)

        # Denormalization
        pred_rel_denorm = pred.clone()
        pred_rel_denorm[..., 0] = pred[..., 0] * rel_x_std + rel_x_mean
        pred_rel_denorm[..., 1] = pred[..., 1] * rel_y_std + rel_y_mean
        
        # 기준점 비정규화
        ref_denorm = initial_pos.clone()
        ref_denorm[..., 0] = initial_pos[..., 0] * px_std + px_mean
        ref_denorm[..., 1] = initial_pos[..., 1] * py_std + py_mean

        pred_absolute = pred_rel_denorm + ref_denorm.unsqueeze(1)  # [B, T, N, 2]

        target_abs_denorm = target_abs.clone()
        target_abs_denorm[..., 0] = target_abs[..., 0] * px_std + px_mean
        target_abs_denorm[..., 1] = target_abs[..., 1] * py_std + py_mean
        
        # ADE, FDE Calculation (diffusion과 동일)
        ade = ((pred_absolute[...,:2] - target_abs_denorm[...,:2])**2).sum(-1).sqrt().mean((1,2))
        fde = ((pred_absolute[:,-1,:,:2] - target_abs_denorm[:,-1,:,:2])**2).sum(-1).sqrt().mean(1)
            
        all_ades.extend(ade.cpu().tolist())
        all_fdes.extend(fde.cpu().tolist())
        
        pred_np = pred_absolute.cpu().numpy()      # [B,T,N,2]
        target_np = target_abs_denorm.cpu().numpy()
        B_, T, N, _ = pred_np.shape
        batch_fres = []
        for b in range(B_):
            per_player = []
            for j in range(N):
                pred_ = pred_np[b, :, j, :]
                target_ = target_np[b, :, j, :]
                per_player.append(calc_frechet_distance(pred_, target_))
            batch_fres.append(np.mean(per_player))
        all_frechet_dist.extend(batch_fres)
        
        # Visualization
        if not visualized:
            base_dir = "results/test_trajs_lstm"
            os.makedirs(base_dir, exist_ok=True)

            for i in range(min(B, visualize_samples)):
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
                
                pred_traj = pred_absolute[i].cpu().numpy()
                target_traj = target_abs_denorm[i].cpu().numpy()
                other_traj = other_den.cpu().numpy()
                
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
        
        del pred, pred_rel_denorm, pred_absolute, target_abs_denorm, ref_denorm, ade, fde
        torch.cuda.empty_cache()
        gc.collect()

print(f"LSTM Baseline Results:")
print(f"ADE: {np.mean(all_ades):.3f} ± {np.std(all_ades):.3f} meters")
print(f"FDE: {np.mean(all_fdes):.3f} ± {np.std(all_fdes):.3f} meters")
print(f"Fréchet: {np.mean(all_frechet_dist):.3f} ± {np.std(all_frechet_dist):.3f} meters")