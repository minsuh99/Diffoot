import os
import random
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.transformer_model import DefenseTrajectoryTransformer
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_evertyhing, worker_init_fn, generator, plot_trajectories_on_pitch
from utils.data_utils import split_dataset_indices, custom_collate_fn

# 1. Hyperparameter Setting
raw_data_path = "idsse-data"
data_save_path = "match_data"
batch_size = 64
num_workers = 8
epochs = 100
learning_rate = 1e-4
SEED = 42

set_evertyhing(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading
print("---Data Loading---")
if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
    organize_and_process(raw_data_path, data_save_path)
else:
    print("Skip organize_and_process")

dataset = MultiMatchSoccerDataset(data_root=data_save_path)
train_idx, val_idx, test_idx = split_dataset_indices(dataset, val_ratio=1/6, test_ratio=1/6, random_seed=SEED)

train_dataloader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
    generator=generator(SEED)
)

val_dataloader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
)

test_dataloader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")

# 3. Model Define
model = DefenseTrajectoryTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4)

# 4. Train
best_state_dict = None
best_val_loss = float("inf")

for epoch in tqdm(range(1, epochs + 1)):
    model.train()
    train_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        condition = batch['condition'].to(device)  # [B, T, 158]
        target = batch['target'].to(device)        # [B, T, 22]
        pred = model(condition, target=target)     # Teacher forcing with GT

        pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)      # [B, T, 11, 2]
        target = target.view(target.shape[0], target.shape[1], 11, 2)

        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
        loss = mse.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)

            pred = model(condition, target=target)
            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)
            target = target.view(target.shape[0], target.shape[1], 11, 2)

            mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
            loss = mse.mean()
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f} Current LR: {scheduler.get_last_lr()[0]:.6f}, target_mean: {target.mean():.4f}, pred_mean: {pred.mean():.4f}")
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state_dict = model.state_dict()


# 5. Inference ＆ Visualization
model.load_state_dict(best_state_dict)
model.eval()
all_ade = []
all_fde = []

visualize_samples = 5
visualization_done = False

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Inference"):
        condition = batch['condition'].to(device)
        target = batch['target'].to(device)
        pred = model(condition, target = None) # Teacher forcing without GT

        pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)
        target = target.view(target.shape[0], target.shape[1], 11, 2)

        # Denormalize
        x_scales = torch.tensor([s[0] for s in batch["pitch_scale"]], device=device).view(-1, 1, 1)
        y_scales = torch.tensor([s[1] for s in batch["pitch_scale"]], device=device).view(-1, 1, 1)

        pred = pred.clone()
        target = target.clone()

        pred[..., 0] *= x_scales
        pred[..., 1] *= y_scales
        target[..., 0] *= x_scales
        target[..., 1] *= y_scales

        # Player-wise ADE
        ade = ((pred - target) ** 2).sum(-1).sqrt().mean(1).mean(1)  # [B]
        all_ade.extend(ade.cpu().numpy())

        # Player-wise FDE
        fde = ((pred[:, -1] - target[:, -1]) ** 2).sum(-1).sqrt().mean(1)  # [B]
        all_fde.extend(fde.cpu().numpy())

        # Visualization
        os.makedirs("results", exist_ok=True)
        if not visualization_done:
            B = pred.shape[0]
            for i in range(min(B, visualize_samples)):
                others = batch["other"][i].view(-1, 12, 2).cpu()
                target_vis = target[i].cpu()
                pred_vis = pred[i].cpu()
                pitch_scale = batch["pitch_scale"][i]

                save_path = f"results/Transformer_sample_{i:02d}.png"
                plot_trajectories_on_pitch(others, target_vis, pred_vis, pitch_scale, save_path=save_path)

            visualization_done = True

avg_ade = np.mean(all_ade)
avg_fde = np.mean(all_fde)
print(f"[Inference] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")
