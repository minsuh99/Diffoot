import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from models.diff_modules import diff_CSDI
from models.diff_model import DiffusionTrajectoryModel
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.data_utils import split_dataset_indices, custom_collate_fn
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. Hyperparameter Setting
    # raw_data_path = "Download raw file path"
    raw_data_path = "kim-internship/Minsuh/SoccerTrajPredict/idsse-data"
    data_save_path = "kim-internship/Minsuh/SoccerTrajPredict/match_data"
    batch_size = 16
    num_workers = 4
    epochs = 30
    learning_rate = 1e-4
    num_samples = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # 2. Data Loading
    print("---Data Loading---")
    organize_and_process(raw_data_path, data_save_path)

    dataset = MultiMatchSoccerDataset(data_root=data_save_path, use_condition_graph=False)
    train_idx, test_idx, _, _ = split_dataset_indices(dataset)

    train_dataloader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
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

    # 4. Train
    print("--- Train ---")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            cond = batch["condition"].to(device)  # [B, T, 158]
            target = batch["target"].to(device).view(-1, cond.shape[1], 11, 2)  # [B, T, 11, 2]

            cond = cond.permute(0, 2, 1).unsqueeze(2)  # [B, 158, 1, T]

            loss = model(target, cond_info=cond)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    print("---Train finished!---")

    # 5. Inference with Best-of-N Sampling
    print("--- Inference ---")
    model.eval()
    all_ade = []
    all_fde = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            cond = batch["condition"].to(device)  # [B, T, 158]
            target = batch["target"].to(device).view(-1, cond.shape[1], 11, 2)

            cond_for_gen = cond.permute(0, 2, 1).unsqueeze(2)  # [B, 158, 1, T]

            # Generate multiple samples
            generated = model.generate(shape=target.shape, cond_info=cond_for_gen, num_samples=num_samples)  # [N, B, T, 11, 2]

            # Find best sample (lowest ADE)
            target_exp = target.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
            ade = ((generated - target_exp) ** 2).sum(-1).sqrt().mean(2)  # [N, B, 11]
            ade = ade.mean(dim=2)  # [N, B]

            best_idx = ade.argmin(dim=0)  # [B]
            best_pred = generated[best_idx, torch.arange(generated.shape[1])]  # [B, T, 11, 2]

            # Final ADE & FDE
            ade_final = ((best_pred - target) ** 2).sum(-1).sqrt().mean(1).mean(1)  # [B]
            fde_final = ((best_pred[:, -1] - target[:, -1]) ** 2).sum(-1).sqrt().mean(1)  # [B]

            all_ade.extend(ade_final.cpu().numpy())
            all_fde.extend(fde_final.cpu().numpy())

    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)
    print(f"[Inference - Best of {num_samples}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")

if __name__ == "__main__":
    main()