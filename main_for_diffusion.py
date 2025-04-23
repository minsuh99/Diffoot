import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.diff_modules import diff_CSDI
from models.diff_model import DiffusionTrajectoryModel
from models.encoder import InteractionGraphEncoder
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_evertyhing, worker_init_fn, generator, plot_trajectories_on_pitch
from utils.data_utils import split_dataset_indices, custom_collate_fn
from utils.graph_utils import build_graph_sequence_from_condition

# 1. Hyperparameter Setting
# raw_data_path = "Download raw file path"
raw_data_path = "idsse-data"
data_save_path = "match_data"
batch_size = 32
num_workers = 8
epochs = 50
learning_rate = 1e-4
num_samples = 10
SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"]   = "1"

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
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")

# 3. Model Define
# Extract node feature dimension
sample = dataset[0]
graph = build_graph_sequence_from_condition({
    "condition": sample["condition"],
    "condition_columns": sample["condition_columns"],
    "pitch_scale": sample["pitch_scale"]
}).to(device)
in_dim = graph['Node'].x.size(1)

csdi_config = {
    "num_steps": 500,
    "channels": 256,
    "diffusion_embedding_dim": 128,
    "nheads": 4,
    "layers": 10,
    "side_dim": 128
}

graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=128, out_dim=128, heads = 2).to(device)
denoiser = diff_CSDI(csdi_config)
model = DiffusionTrajectoryModel(denoiser, num_steps=csdi_config["num_steps"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4)

# 4. Train
best_state_dict = None
best_val_loss = float("inf")

train_losses = []
val_losses   = []

for epoch in tqdm(range(1, epochs + 1), desc="Training..."):
    model.train()
    train_noise_loss = 0
    train_player_loss = 0
    train_loss = 0

    for batch in tqdm(train_dataloader, desc = "Batch Training..."):
        cond = batch["condition"]                            # [B, T, F]
        target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)  # [B, T, 11, 2]
        graph_batch = batch["graph"].to(device)                              # HeteroData batch

        # graph → H
        H = graph_encoder(graph_batch)                                       # [B, 128]
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))                    # [B,128,11,T]

        # Preparing Self-conditioning data
        if torch.rand(1, device=device) < 0.5:
            s = torch.zeros_like(target)
        else:
            with torch.no_grad():
                t = torch.randint(0, model.num_steps, (target.size(0),), device=device)
                x_t, noise = model.q_sample(target, t)
                x_t = x_t.permute(0,3,2,1)
                eps_pred1 = model.model(x_t, t, cond_H, self_cond=None)
                a_hat = model.alpha_hat.to(device)[t].view(-1,1,1,1)
                x0_hat = (x_t - (1 - a_hat).sqrt() * eps_pred1) / a_hat.sqrt()
                x0_hat = x0_hat.permute(0,3,2,1)
            s = x0_hat
        
        noise_loss, player_loss = model(target, cond_info=cond_H, self_cond=s)
        loss = noise_loss + player_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_noise_loss += noise_loss.item()
        train_player_loss += player_loss.item()
        train_loss += loss.item()

    num_batches = len(train_dataloader)
    
    avg_noise_loss = train_noise_loss / num_batches
    avg_player_loss = train_player_loss / num_batches
    avg_train_loss = train_loss / num_batches

    # --- Validation ---
    model.eval()
    val_noise_loss = 0
    val_player_loss = 0
    val_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            cond = batch["condition"]
            target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)
            graph_batch = batch["graph"].to(device)

            H = graph_encoder(graph_batch)
            cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))
            
            s = torch.zeros_like(target)
            
            noise_loss, player_loss = model(target, cond_info=cond_H, self_cond=s)
            val_noise_loss += noise_loss.item()
            val_player_loss += player_loss.item()
            val_total_loss += (noise_loss + player_loss).item()

    avg_val_loss = val_total_loss / len(val_dataloader)
  
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    tqdm.write(f"[Epoch {epoch}]\nCost: {avg_train_loss:.6f} | Noise Loss: {avg_noise_loss:.6f} | Player Loss: {avg_player_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}\n"
               f"Val Loss: {avg_val_loss:.6f} | Noise: {val_noise_loss/len(val_dataloader):.6f} | Player: {val_player_loss/len(val_dataloader):.6f}")
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state_dict = model.state_dict()
        
# 4-1. Plot learning_curve
plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train & Validation Loss, 500 steps, 128 channels, 128 embedding dim, 4 heads, 6 layers')
plt.legend()
plt.tight_layout()

plt.savefig('results/diffusion_lr_curve.png')

plt.show()

# 5. Inference (Best-of-N Sampling) & Visualization
model.load_state_dict(best_state_dict)
model.eval()
all_ade, all_fde = [], []
visualize_samples = 5
visualization_done = False

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Inference"):
        cond = batch["condition"]
        target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)
        graph_batch = batch["graph"].to(device)
        B = cond.size(0)

        H = graph_encoder(graph_batch)
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))

        generated = model.generate(shape=target.shape, cond_info=cond_H, num_samples=num_samples)  # [N, B, T, 11, 2]
        target = target.unsqueeze(0).expand(num_samples, -1, -1, -1, -1).clone()

        # Denormalize
        x_scales = torch.tensor([s[0] for s in batch["pitch_scale"]], device=device, dtype=torch.float32).view(1, B, 1, 1).expand(num_samples, B, 1, 1)
        y_scales = torch.tensor([s[1] for s in batch["pitch_scale"]], device=device, dtype=torch.float32).view(1, B, 1, 1).expand(num_samples, B, 1, 1)

        generated[..., 0] *= x_scales
        generated[..., 1] *= y_scales
        target[..., 0] *= x_scales
        target[..., 1] *= y_scales
        
        # generated/[0,1] → [−1,1] → 실제[m]
        # generated[..., 0] = (generated[..., 0] - 0.5) * 2 * x_scales
        # generated[..., 1] = (generated[..., 1] - 0.5) * 2 * y_scales

        # # target도 동일하게 복원
        # target[..., 0] = (target[..., 0] - 0.5) * 2 * x_scales
        # target[..., 1] = (target[..., 1] - 0.5) * 2 * y_scales

        ade = ((generated - target) ** 2).sum(-1).sqrt().mean(2).mean(2)  # [N, B]
        best_idx = ade.argmin(dim=0)                                    # [B]
        best_pred = generated[best_idx, torch.arange(B)]                # [B, T, 11, 2]
        best_target = target[0, torch.arange(B)]                        # [B, T, 11, 2]

        ade_final = ((best_pred - best_target) ** 2).sum(-1).sqrt() \
                        .mean(1).mean(0)                               # [B]
        fde_final = ((best_pred[:, -1] - best_target[:, -1]) ** 2)\
                        .sum(-1).sqrt()                               # [B]

        all_ade.extend(ade_final.cpu().numpy())
        all_fde.extend(fde_final.cpu().numpy())
        

        if not visualization_done:
            os.makedirs("results", exist_ok=True)
            for i in range(min(B, visualize_samples)):
                others = batch["other"][i].view(-1,12,2).cpu().numpy()
                target = best_target[i].cpu().numpy()   # (T,11,2)
                pred = best_pred[i].cpu().numpy()     # (T,11,2)
                
                save_path = f"results/Diff_sample_{i:02d}.png"
                
                os.makedirs('results/player_trajs', exist_ok=True)
                for p in range(11):
                    save_path = f'results/player_trajs/sample{i:02d}_def{p:02d}.png'
                    plot_trajectories_on_pitch(others, target, pred, player_idx=p, save_path=save_path)
            visualization_done = True

avg_ade = np.mean(all_ade)
avg_fde = np.mean(all_fde)
print(f"[Inference - Best of {num_samples}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")