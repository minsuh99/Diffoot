import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from models.lstm_model import DefenseTrajectoryPredictor
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

# 0331: LSTM 기본 모델 사용중
def main():
    # 1. Hyperparameter Setting
    raw_data_path = "kim-internship/Minsuh/SoccerTrajPredict/idsse-data"
    data_save_path = "kim-internship/Minsuh/SoccerTrajPredict/match_data"
    batch_size = 64
    num_workers = 4
    epochs = 30
    learning_rate = 1e-3

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
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )
    print("---Data Load!---")
    
    # 3. Model Define
    model = DefenseTrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(device)

    # 4. Train
    print("--- Train ---")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            condition = batch['condition'].to(device)  # [B, T, 158]
            target = batch['target'].to(device)        # [B, T, 22] (11 defenders × (x, y))
            pred = model(condition)                    # [B, T, 22]

            # 선수별 (x, y) MSE → 평균
            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)      # [B, T, 11, 2]
            target = target.view(target.shape[0], target.shape[1], 11, 2)  # [B, T, 11, 2]

            mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
            loss = mse.mean()  # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    print("---Train finished!---")

    # 5. Inference
    print("--- Inference ---")
    model.eval()
    all_ade = []
    all_fde = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)  # [B, T, 22] (11 defenders × (x, y))

            pred = model(condition)             # [B, T, 22]
            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)        # [B, T, 11, 2]
            target = target.view(target.shape[0], target.shape[1], 11, 2)

            # 선수별 ADE: [B, 11]
            ade = ((pred - target) ** 2).sum(-1).sqrt().mean(1)  # [B, 11]
            ade = ade.mean(dim=1)  # 배치별 선수 평균 → [B]
            all_ade.extend(ade.cpu().numpy())

            # 선수별 FDE: [B, 11]
            fde = ((pred[:, -1] - target[:, -1]) ** 2).sum(-1).sqrt()  # [B, 11]
            fde = fde.mean(dim=1)  # [B]
            all_fde.extend(fde.cpu().numpy())

    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)

    print(f"[Inference] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")


if __name__ == "__main__":
    main()
