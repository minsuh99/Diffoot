import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from make_dataset import MultiMatchSoccerDataset
from Minsuh.SoccerTrajPredict.code.models.encoder import GraphAutoencoder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dataset
dataset = MultiMatchSoccerDataset(
    data_root="../kim-internship/Minsuh/SoccerTrajPredict/data",
    segment_length=250,
    condition_length=125,
    framerate=25,
    stride=25
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

# 2. Input feature dimension 자동 추출
sample = dataset[0]
first_frame = sample["condition_graph_seq"][0]
in_dim_dict = {
    node_type: [len(first_frame[node_type].x)]  # e.g., {'x': [11,1], 'y': [11,1], ...}
    for node_type in first_frame.node_types
}

# 3. 모델 초기화
model = GraphAutoencoder(
    in_dim_dict=in_dim_dict,
    hidden_dim=64,
    temporal_hidden_dim=64,
    out_dim=128  # 최종 H 크기
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for batch in dataloader:
    print("Batch keys:", batch.keys())

    condition_seq = batch["condition_graph_seq"]  # batch_size=1 가정
    print("Type of condition_seq:", type(condition_seq))  # 리스트여야 함
    print("Length of condition_seq (frames):", len(condition_seq))

    first_frame = condition_seq[0]
    print("First frame type:", type(first_frame))

    # HeteroData 속성 체크
    print(" - Node types:", first_frame.node_types)
    print(" - Edge types:", first_frame.edge_types)

    for node_type in first_frame.node_types:
        print(f"   {node_type} - x keys:", first_frame[node_type].x.keys())

    break  # 첫 배치만 확인



# # 4. 학습 루프
# print("Training Autoencoder...")
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

#     for batch in pbar:
#         graph_seq = batch["condition_graph_seq"][0]
#         graph_seq = [g.to(device) for g in graph_seq]

#         last_frame = graph_seq[-1]  # 복원 대상 frame
#         last_frame = last_frame.to(device)

#         optimizer.zero_grad()
#         H = model(graph_seq)  # H ∈ ℝ^d
#         loss = model.decode_from_H(H, last_frame)  # self-supervised loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         pbar.set_postfix({"loss": loss.item()})

#     avg_loss = total_loss / len(dataloader)
#     print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

# # 5. 저장
# torch.save(model.state_dict(), "pretrained_graph_autoencoder.pt")
