import os
import random
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.diff_modules import diff_CSDI
from models.diff_model import DiffusionTrajectoryModel
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_seed, plot_trajectories_on_pitch
from utils.data_utils import split_dataset_indices, custom_collate_fn

def main():
    # 0. Set seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(42)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(42)  
    
    # 1. Hyperparameter Setting
    # raw_data_path = "Download raw file path"
    raw_data_path = "idsse-data"
    data_save_path = "match_data"
    batch_size = 32
    num_workers = 8
    epochs = 100
    learning_rate = 1e-4
    num_samples = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    print("---Data Loading---")
    if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
        organize_and_process(raw_data_path, data_save_path)
    else:
        print("Skip organize_and_process")

    dataset = MultiMatchSoccerDataset(data_root=data_save_path, use_condition_graph=False)
    train_idx, test_idx, _, _ = split_dataset_indices(dataset)
    train_dataloader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_dataloader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker
    )
    print("---Data Load!---")

    # 3. Model Define
    csdi_config = {
        "num_steps": 1000,
        "channels": 64,
        "diffusion_embedding_dim": 128,
        "nheads": 4,
        "layers": 4,
        "side_dim": 158
    }
    denoiser = diff_CSDI(csdi_config)
    model = DiffusionTrajectoryModel(denoiser, num_steps=csdi_config["num_steps"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Train
    print("--- Train ---")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            cond = batch["condition"].to(device)  # [B, T, 158]
            target = batch["target"].to(device).view(-1, cond.shape[1], 11, 2)  # [B, T, 11, 2]
            
            cond = cond.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, 11, -1)  # [B, 158, 11, T]
            
            loss = model(target, cond_info=cond)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, Current LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step(avg_loss)

    print("---Train finished!---")

    # 5. Inference (Best-of-N Sampling) & Visualization
    print("--- Inference ---")
    model.eval()
    all_ade = []
    all_fde = []
    visualize_samples = 5
    visualization_done = False

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Inference
            cond = batch["condition"].to(device)   # [B, T, 158]
            target = batch["target"].to(device).view(-1, cond.shape[1], 11, 2)  # [B, T, 11, 2]

            cond_for_gen = cond.permute(0, 2, 1).unsqueeze(2)  # [B, 158, 1, T]
            cond_for_gen = cond_for_gen.expand(-1, -1, 11, -1)

            generated = model.generate(shape=target.shape, cond_info=cond_for_gen, num_samples=num_samples)  # [N, B, T, 11, 2]
            target_exp = target.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)  # [N, B, T, 11, 2]
            
            # batch_size
            B = generated.shape[1]

            # Denormalize            
            x_scales = torch.tensor([s[0] for s in batch["pitch_scale"]], device=device , dtype=torch.float32).view(1, B, 1, 1)
            y_scales = torch.tensor([s[1] for s in batch["pitch_scale"]], device=device, dtype=torch.float32).view(1, B, 1, 1)
            
            x_scales = x_scales.expand(num_samples, B, 1, 1)
            y_scales = y_scales.expand(num_samples, B, 1, 1)

            generated = generated.clone()
            target_exp = target_exp.clone()
            
            generated[..., 0] = (generated[..., 0] + 1.0) * x_scales
            generated[..., 1] = (generated[..., 1] + 1.0) * y_scales

            target_exp[..., 0] = (target_exp[..., 0] + 1.0) * x_scales
            target_exp[..., 1] = (target_exp[..., 1] + 1.0) * y_scales

            ade = ((generated - target_exp) ** 2).sum(-1).sqrt().mean(2)  # [N, B, 11]
            ade = ade.mean(dim=2)  # [N, B]

            best_idx = ade.argmin(dim=0)  # [B]
            best_pred = generated[best_idx, torch.arange(generated.shape[1])]     # [B, T, 11, 2]
            best_target = target_exp[0, torch.arange(generated.shape[1])]         # [B, T, 11, 2]

            ade_final = ((best_pred - best_target) ** 2).sum(-1).sqrt().mean(1).mean(1)  # [B]
            fde_final = ((best_pred[:, -1] - best_target[:, -1]) ** 2).sum(-1).sqrt()    # [B]

            all_ade.extend(ade_final.cpu().numpy())
            all_fde.extend(fde_final.cpu().numpy())
            
            # Visualization
            os.makedirs("results", exist_ok=True)
            
            if not visualization_done:
                for i in range(min(B, visualize_samples)):
                    others = batch["other"][i].view(-1, 12, 2).cpu()       # [T, 12, 2]
                    target_vis = target[i].cpu()                           # [T, 11, 2]
                    pred_vis = best_pred[i].cpu()                          # [T, 11, 2]
                    pitch_scale = batch["pitch_scale"][i]

                    save_path = f"results/sample_{i:02d}.png"
                    plot_trajectories_on_pitch(others, target_vis, pred_vis, pitch_scale, save_path=save_path)

                visualization_done = True
                
    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)
    print(f"[Inference - Best of {num_samples}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")

if __name__ == "__main__":
    main()