import os
import random
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.lstm_model import DefenseTrajectoryPredictor
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_seed, plot_trajectories_on_pitch
from utils.data_utils import split_dataset_indices, custom_collate_fn

# 0331: LSTM 기본 모델 사용중
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
    # epochs = 100
    epochs=1
    learning_rate = 1e-4
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
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker
        )
    
    print("---Data Load!---")
    
    # 3. Model Define
    model = DefenseTrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Train
    print("--- Train ---")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            condition = batch['condition'].to(device)  # [B, T, 158]
            target = batch['target'].to(device)        # [B, T, 22]
            pred = model(condition)                    # [B, T, 22]

            # 선수별 경로 MSE 구한 후 평균
            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)      # [B, T, 11, 2]
            target = target.view(target.shape[0], target.shape[1], 11, 2)  # [B, T, 11, 2]

            mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
            loss = mse.mean()  # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, Current LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step(avg_loss)

    print("---Train finished!---")

    # 5. Inference ＆ Visualization
    print("--- Inference ---")
    model.eval()
    all_ade = []
    all_fde = []
    
    visualize_samples = 5
    visualization_done = False

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Inference
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)  # [B, T, 22]
            pred = model(condition)              # [B, T, 22]

            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)      # [B, T, 11, 2]
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
                    
                    save_path = f"results/LSTM_sample_{i:02d}.png"
                    plot_trajectories_on_pitch(others, target_vis, pred_vis, pitch_scale, save_path=save_path)

                visualization_done = True
            
    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)
    print(f"[Inference] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")

if __name__ == "__main__":
    main()
