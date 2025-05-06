import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_WARN_ONCE"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging

fx_logger = logging.getLogger("torch.fx.experimental.symbolic_shapes")
class OnceFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()
    def filter(self, record):
        msg = record.getMessage()
        if msg in self.seen:
            return False
        self.seen.add(msg)
        return True
fx_logger.addFilter(OnceFilter())

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from models.diff_modules import DiffCSDI
from models.diff_model import DiffusionTrajectoryModel
from models.encoder import InteractionGraphEncoder, TargetTrajectoryEncoder
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_evertyhing, worker_init_fn, generator, plot_trajectories_on_pitch, log_graph_stats
from utils.data_utils import split_dataset_indices, custom_collate_fn
from utils.graph_utils import build_graph_sequence_from_condition

# SEED Fix
SEED = 42
set_evertyhing(SEED)
torch.set_float32_matmul_precision('high')

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
    'num_steps': 500,
    'channels': 128,
    'diffusion_embedding_dim': 128,
    'nheads': 4,
    'layers': 5,
    'side_dim': 256
}

hyperparams = {
    'raw_data_path': "idsse-data",
    'data_save_path': "match_data",
    'train_batch_size': 64,
    'val_batch_size': 64,
    'test_batch_size': 64,
    'num_workers': 8,
    
    'epochs': 50,
    'learning_rate': 5e-4,
    'self_conditioning_ratio': 0.5,
    
    'num_steps': 500,
    'ddim_steps': 200,
    'eta': 0.0,
    'embedding_dim': 128,
    'base_channels': 128,
    'depth': 4,
    'heads': 4,
    'feature_dim': 2,
    
    'num_samples': 10,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
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

num_steps = hyperparams['num_steps']
ddim_steps = hyperparams['ddim_steps']
eta = hyperparams['eta']

embedding_dim = hyperparams['embedding_dim']
base_channels = hyperparams['base_channels']
depth = hyperparams['depth']
heads = hyperparams['heads']
feature_dim = hyperparams['feature_dim']

num_samples = hyperparams['num_samples']
device = hyperparams['device']

logger.info(f"Hyperparameters: {hyperparams}")

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
    batch_size=train_batch_size,
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
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
)

test_dataloader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")
print(f"Train Dataset: {len(train_dataloader.dataset)} samples | "
      f"Validation Dataset: {len(val_dataloader.dataset)} samples | "
      f"Test Dataset: {len(test_dataloader.dataset)} samples")

# 3. Model Define
# Extract node feature dimension
sample = dataset[0]
_, T, N = sample["target"].unsqueeze(0).shape
graph = build_graph_sequence_from_condition({
    "condition": sample["condition"],
    "condition_columns": sample["condition_columns"],
    "pitch_scale": sample["pitch_scale"]
}).to(device)
log_graph_stats(graph, logger, prefix="InitGraphSample")

in_dim = graph['Node'].x.size(1)

# Extract target's history trajectories from condition
condition_columns = sample["condition_columns"]
target_columns = sample["target_columns"]
target_idx = [condition_columns.index(col) for col in target_columns if col in condition_columns]

# graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=128, out_dim=128).to(device)
# history_encoder = TargetTrajectoryEncoder(num_layers=5).to(device)
denoiser = DiffCSDI(csdi_config, input_dim=feature_dim).to(device)

model = DiffusionTrajectoryModel(denoiser, num_steps=num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 2, threshold = 1e-4)

logger.info(f"Device: {device}")
# logger.info(f"GraphEncoder: {graph_encoder}")
# logger.info(f"HistoryEncoder: {history_encoder}")
logger.info(f"DiffusionTrajectoryModel: {model}")


# 4. Train
best_state_dict = None
best_val_loss = float("inf")

train_losses = []
val_losses   = []

model.train()
scaler = GradScaler()

for epoch in tqdm(range(1, epochs + 1), desc="Training..."):
    train_noise_loss = 0
    # train_mse_loss = 0
    train_frechet_loss = 0
    train_fde_loss = 0
    train_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} Train"):
        cond_all=batch['condition'].to(device)  # (B,T,C_all)
        B, T, _ = cond_all.shape
        target = batch['target'].to(device).view(B,T,11, feature_dim)
        graph_batch = batch['graph'].to(device)

        # other trajectories -> cond_info
        cols = batch['condition_columns'][0]
        others_cols = batch['other_columns'][0]

        idxs = [cols.index(c) for c in others_cols]
        tmp = cond_all[:,:,idxs].view(B, T, len(others_cols)//2, feature_dim)
        cond_info = tmp.permute(0,3,2,1)     # (B,2,M,T)

        # Self-conditioning
        if torch.rand(1, device=device) < self_conditioning_ratio:
            s = None
        else:
            t0 = torch.randint(0, model.num_steps, (B,), device=device)
            x_t, _ = model.q_sample(target, t0)
            x_t_in = x_t.permute(0, 3, 2, 1)  # [B, C, N, T] → [B, T, N, C]
            with torch.no_grad():
                eps1 = denoiser(x_t_in, cond_info, t0, graph_batch, self_cond=None)
            eps1 = eps1.permute(0, 3, 2, 1)  # [B, T, N, C] → [B, C, N, T]
            a_hat0 = model.alpha_hat[t0].view(-1, 1, 1, 1)
            s = (x_t - torch.sqrt(1 - a_hat0) * eps1) / torch.sqrt(a_hat0)

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            noise_loss, player_loss_frechet, player_loss_fde = model(target, cond_info=cond_info, graph_batch=graph_batch, self_cond=s)
            loss = noise_loss + player_loss_frechet * 0.2 + player_loss_fde

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_noise_loss += (noise_loss).item()
        train_frechet_loss += (player_loss_frechet * 0.2).item()
        train_fde_loss += player_loss_fde.item()
        train_loss += loss.item()

    avg_noise_loss = train_noise_loss / len(train_dataloader)
    avg_frechet_loss = train_frechet_loss / len(train_dataloader)
    avg_fde_loss = train_fde_loss / len(train_dataloader)
    avg_train_loss = train_loss / len(train_dataloader)


    # --- Validation ---
    model.eval()
    val_noise_loss = 0
    # val_mse_loss = 0
    val_frechet_loss = 0
    val_fde_loss = 0
    val_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} Val"):
            cond_all=batch['condition'].to(device)
            B, T, _ = cond_all.shape
            target = batch['target'].to(device).view(B,T,11,feature_dim)
            graph_batch = batch['graph'].to(device)

            cols = batch['condition_columns'][0]
            others_cols = batch['other_columns'][0]

            idxs = [cols.index(c) for c in others_cols]

            tmp = cond_all[:,:,idxs].view(B, T, len(others_cols)//2, feature_dim)
            cond_info = tmp.permute(0,3,2,1)

            noise_loss, player_loss_frechet, player_loss_fde = model(target, cond_info=cond_info, graph_batch=graph_batch, self_cond=None)
            val_loss = noise_loss + player_loss_frechet * 0.2 + player_loss_fde
            
            val_noise_loss += (noise_loss).item()
            val_frechet_loss += (player_loss_frechet * 0.2).item()
            val_fde_loss += player_loss_fde.item()
            val_total_loss += val_loss.item()

    avg_val_noise_loss = val_noise_loss / len(val_dataloader)
    avg_val_frechet_loss = val_frechet_loss / len(val_dataloader)
    avg_val_fde_loss = val_fde_loss / len(val_dataloader)
    avg_val_loss = val_total_loss / len(val_dataloader)
  
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"[Epoch {epoch}/{epochs}] Train Loss={avg_train_loss:.6f} (Noise={avg_noise_loss:.6f}, Frechet={avg_frechet_loss:.6f}) , FDE={avg_fde_loss:.6f})| Val Loss={avg_val_loss:.6f} | LR={current_lr:.6e}")
    
    tqdm.write(f"[Epoch {epoch}]\n"
               f"[Train] Cost: {avg_train_loss:.6f} | Noise Loss: {avg_noise_loss:.6f} | Frechet Loss: {avg_frechet_loss:.6f} | FDE Loss: {avg_fde_loss:.6f} LR: {current_lr:.6f}\n"
               f"[Validation] Val Loss: {avg_val_loss:.6f} | Noise: {avg_val_noise_loss:.6f} | Frechet: {avg_val_frechet_loss:.6f} | FDE: {avg_val_fde_loss:.6f}")
    
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state_dict = model.state_dict()

logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        
# 4-1. Plot learning_curve
plt.figure(figsize=(12, 8))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Train & Validation Loss, {num_steps} steps, {base_channels} base channels\n"
          f"{embedding_dim} embedding dim, {depth} depth "
          f"self-conditioning ratio: {self_conditioning_ratio}")
plt.legend()
plt.tight_layout()

plt.savefig('results/0505_diffusion_lr_curve.png')

# 5. Inference (Best-of-N Sampling) & Visualization
model.load_state_dict(best_state_dict)
model.eval()
all_best_ades_test = []
all_best_fdes_test = []
visualize_samples = 5
visualized = False

with torch.no_grad():
    for batch in tqdm(test_dataloader,desc="Test"):
        cond_all = batch['condition'].to(device)
        B, T, _ = cond_all.shape
        target = batch['target'].to(device).view(B, T, 11, feature_dim)
        graph_batch = batch['graph'].to(device)
        
        cols = batch['condition_columns'][0]
        others_cols = batch['other_columns'][0]

        idxs = [cols.index(c) for c in others_cols]

        tmp = cond_all[:,:,idxs].view(B, T, len(others_cols)//2, feature_dim)
        cond_info = tmp.permute(0,3,2,1)
        
        best_ade_t = torch.full((B,),float('inf'),device=device)
        best_pred_t = torch.zeros_like(target)
        best_fde_t = torch.full((B,),float('inf'),device=device)
        
        preds=model.generate(target.shape, cond_info=cond_info, graph_batch=graph_batch, 
                             ddim_steps=hyperparams['ddim_steps'], eta=hyperparams['eta'],num_samples=hyperparams['num_samples'])
        scales=torch.tensor(batch['pitch_scale'],device=device).view(B,1,1,2)
        target_den = target * scales

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            pred_i_den = pred_i * scales
            
            ade_i = ((pred_i_den - target_den)**2).sum(-1).sqrt().mean((1,2))
            fde_i = ((pred_i_den[:,-1] - target_den[:,-1])**2).sum(-1).sqrt().mean(1)
            
            better = ade_i < best_ade_t
            
            best_pred_t[better] = pred_i_den[better]
            best_ade_t[better] = ade_i[better]
            best_fde_t[better] = fde_i[better]

        all_best_ades_test.extend(best_ade_t.cpu().tolist())
        all_best_fdes_test.extend(best_fde_t.cpu().tolist())

        # Visualization
        if not visualized:
            base_dir = "results/test_trajs"
            os.makedirs(base_dir, exist_ok=True)
            for i in range(min(B, visualize_samples)):
                sample_dir = os.path.join(base_dir, f"sample{i:02d}")
                os.makedirs(sample_dir, exist_ok=True)
                
                other_cols  = batch["other_columns"][i]
                target_cols = batch["target_columns"][i]
                defender_nums = [int(col.split('_')[1]) for col in target_cols[::2]]

                others_seq = batch["other"][i].view(T, 12, 2).cpu().numpy()
                target_traj = target_den[i].cpu().numpy()
                pred_traj = best_pred_t[i].cpu().numpy()

                for idx, jersey in enumerate(defender_nums):
                    save_path = os.path.join(sample_dir, f"player_{jersey:02d}.png")
                    plot_trajectories_on_pitch(others_seq, target_traj, pred_traj,
                                               other_columns=other_cols, target_columns=target_cols,
                                               player_idx=idx, annotate=True, save_path=save_path)

            visualized = True

avg_test_ade = np.mean(all_best_ades_test)
avg_test_fde = np.mean(all_best_fdes_test)
print(f"[Test Best-of-{num_samples}] Average ADE: {avg_test_ade:.4f} | Average FDE: {avg_test_fde:.4f}")
print(f"[Test Best-of-{num_samples}] Best ADE overall: {min(all_best_ades_test):.4f} | Best FDE overall: {min(all_best_fdes_test):.4f}")