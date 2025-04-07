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

# 데이터 로더 워커 시드 고정을 위한 함수 (diffusion 코드와 동일)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # 1. Hyperparameter Setting
    raw_data_path = "idsse-data"
    data_save_path = "match_data"
    batch_size = 32
    num_workers = 8
    epochs = 30
    learning_rate = 5e-4
    num_samples = 10  # 추론 시 Best-of-N 샘플링 횟수

    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        collate_fn=custom_collate_fn
    )
    print("---Data Load!---")
    
    # 3. Model Define
    model = DefenseTrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Train
    print("--- Train ---")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            condition = batch["condition"].to(device)  # [B, T, 158]
            target = batch["target"].to(device)          # [B, T, 22] (11명 × (x, y))
            
            # 모델 예측 및 (x, y) 2차원으로 재구성
            pred = model(condition)                      # [B, T, 22]
            pred = pred.view(pred.shape[0], pred.shape[1], 11, 2)        # [B, T, 11, 2]
            target = target.view(target.shape[0], target.shape[1], 11, 2)  # [B, T, 11, 2]

            loss = ((pred - target) ** 2).mean(dim=(1, 2, 3)).mean()
            
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
            condition = batch["condition"].to(device)  # [B, T, 158]
            target = batch["target"].to(device)          # [B, T, 22]
            
            # LSTM 모델은 결정론적이므로, 여러 번 추론해도 결과가 동일할 수 있음.
            # Best-of-N 프로세스를 모방하기 위해 num_samples 번 예측하여 스택함.
            predictions = []
            for _ in range(num_samples):
                pred = model(condition)  # [B, T, 22]
                predictions.append(pred)
            predictions = torch.stack(predictions, dim=0)  # [N, B, T, 22]
            
            # (x, y) 2차원으로 재구성
            predictions = predictions.view(num_samples, predictions.shape[1], predictions.shape[2], 11, 2)
            target = target.view(target.shape[0], target.shape[1], 11, 2)
            target_exp = target.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)  # [N, B, T, 11, 2]
            
            # Denormalize (pitch_scale 적용)
            B = predictions.shape[1]
            x_scales = torch.tensor([s[0] for s in batch["pitch_scale"]], device=device, dtype=torch.float32).view(1, B, 1, 1)
            y_scales = torch.tensor([s[1] for s in batch["pitch_scale"]], device=device, dtype=torch.float32).view(1, B, 1, 1)
            x_scales = x_scales.expand(num_samples, B, 1, 1)
            y_scales = y_scales.expand(num_samples, B, 1, 1)

            predictions = predictions.clone()
            target_exp = target_exp.clone()
            predictions[..., 0] = (predictions[..., 0] + 1.0) * x_scales
            predictions[..., 1] = (predictions[..., 1] + 1.0) * y_scales
            target_exp[..., 0] = (target_exp[..., 0] + 1.0) * x_scales
            target_exp[..., 1] = (target_exp[..., 1] + 1.0) * y_scales
            
            # 각 샘플별 ADE 계산
            ade = ((predictions - target_exp) ** 2).sum(-1).sqrt().mean(dim=2)  # [N, B, 11]
            ade = ade.mean(dim=2)  # [N, B]
            
            # 각 배치별 최적 샘플 선택
            best_idx = ade.argmin(dim=0)  # [B]
            best_pred = predictions[best_idx, torch.arange(B)]     # [B, T, 11, 2]
            best_target = target_exp[0, torch.arange(B)]             # [B, T, 11, 2]
            
            ade_final = ((best_pred - best_target) ** 2).sum(-1).sqrt().mean(dim=1).mean(dim=1)  # [B]
            fde_final = ((best_pred[:, -1] - best_target[:, -1]) ** 2).sum(-1).sqrt()             # [B]
            
            all_ade.extend(ade_final.cpu().numpy())
            all_fde.extend(fde_final.cpu().numpy())

    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)
    print(f"[Inference - Best of {num_samples}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")

if __name__ == "__main__":
    main()
