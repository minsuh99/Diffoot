import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import math
from tqdm import tqdm


def frame_tensor_to_df(frame_tensor, column_names):
    np_array = frame_tensor.cpu().numpy()
    df = pd.DataFrame([np_array], columns=column_names)
    return df


def get_global_dist_min_max(dataset):
    all_dists = []
    for sample in dataset:
        cond = sample["condition"]
        cols = sample["condition_columns"]
        for t in range(cond.shape[0]):
            for col in cols:
                if col.endswith("_dist"):
                    val = cond[t, cols.index(col)].item()
                    all_dists.append(val)
    all_dists_tensor = torch.tensor(all_dists)
    return all_dists_tensor.min().item(), all_dists_tensor.max().item()


def extract_node_features(condition_tensor, condition_columns):
    column_index_map = {col: idx for idx, col in enumerate(condition_columns)}
    device = condition_tensor.device
    dtype  = condition_tensor.dtype
    player_bases = sorted(list(set(col.rsplit("_", 1)[0] for col in condition_columns if "ball" not in col)))
    attk_bases = [base for base in player_bases if "Attk" in base or any(f"{base}_position" in col for col in condition_columns[:22*7])][:11]
    def_bases = [base for base in player_bases if base not in attk_bases][:11]

    unified_feats = []

    def get_feat(base, node_type_idx):
        feats = []
        for feat in ["x", "y", "vx", "vy"]:
            col = f"{base}_{feat}"
            if col in column_index_map:
                val = condition_tensor[column_index_map[col]]
            else:
                val = torch.tensor(0.0, device=device, dtype=dtype)
            feats.append(val)
        col = f"{base}_dist"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_position"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_starter"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_possession_duration"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_neighbor_count"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        
        feats.append(torch.tensor(float(node_type_idx), device=device, dtype=dtype))
        return torch.stack(feats)

    for base in attk_bases:
        unified_feats.append(get_feat(base, 0))
    for base in def_bases:
        unified_feats.append(get_feat(base, 1))

    # Ball
    ball_feats = []
    for feat in ["x", "y", "vx", "vy"]:
        col = f"ball_{feat}"
        val = condition_tensor[column_index_map[col]] if col in column_index_map else torch.tensor(0.0)
        ball_feats.append(val)
    ball_feats += [torch.tensor(-1.0, device=device, dtype=dtype)] * 4  # dist, position, starter, possession
    ball_feats.append(torch.tensor(2.0, device=device, dtype=dtype))    # node_type
    unified_feats.append(torch.stack(ball_feats))

    return {"Node": torch.stack(unified_feats)}

# Edge
def build_edges_based_on_interactions(node_features, pitch_scale):
    edge_index_dict, edge_attr_dict = {}, {}
    x_scale, y_scale = pitch_scale

    nodes = node_features["Node"]        # (N, F)
    node_type = nodes[:, -1]             # 0=Attk, 1=Def, 2=Ball
    poss_dur = nodes[:, 7]
    neighbor_count = nodes[:, 8]         # (N,)
    masks = {t: (node_type == t) for t in (0, 1, 2)}

    weight_thr = 0.05   # 필터 임계값

    def make_edges(s_t, d_t, rel):
        s_idx = torch.where(masks[s_t])[0]          # (Ns,)
        d_idx = torch.where(masks[d_t])[0]          # (Nd,)
        if s_idx.numel() == 0 or d_idx.numel() == 0:
            edge_index_dict[("Node", rel, "Node")] = torch.empty((2, 0), dtype=torch.long)
            edge_attr_dict [("Node", rel, "Node")] = torch.empty((0, 1), dtype=torch.float32)
            return

        # 거리 계산 시 실제 거리 사용
        s_pos = nodes[s_idx, :2] * torch.tensor([x_scale, y_scale], device=nodes.device)
        d_pos = nodes[d_idx, :2] * torch.tensor([x_scale, y_scale], device=nodes.device)
        dist = (s_pos.unsqueeze(1) - d_pos.unsqueeze(0)).norm(dim=-1)  # (Ns, Nd)
        
        # possession 플래그
        poss_s = (poss_dur[s_idx].unsqueeze(1) > 0.0)  # (Ns,1)
        poss_d = (poss_dur[d_idx].unsqueeze(0) > 0.0)  # (1,Nd)
        
        if rel == "attk_and_attk" or rel == "def_and_def":
            W_dist = 1.0 / (1.0 + dist)
            
            Nopp_s = neighbor_count[s_idx].unsqueeze(1)
            Nopp_d = neighbor_count[d_idx].unsqueeze(0)
            
            base_sit = torch.exp(-(Nopp_s + Nopp_d) / (dist + 1e-6))  # (Ns,Nd)
            W_situation = base_sit * (poss_s | poss_d).float()
            
            weight = W_dist + W_situation

        elif rel == "attk_and_def":
            W_dist = 1.0 / (1.0 + dist)
            
            dir_vec = (s_pos.unsqueeze(1) - d_pos.unsqueeze(0)) / (dist.unsqueeze(-1) + 1e-6)  # (Ns,Nd,2)
            v_real = nodes[d_idx, 2:4] * torch.tensor([x_scale, y_scale], device=nodes.device) 
            v_def = v_real.unsqueeze(0).expand(dist.size(0), -1, -1)                         # (Ns,Nd,2)
            W_situation = (v_def * dir_vec).sum(dim=-1)
            
            weight = W_dist + W_situation

        elif rel == "attk_and_ball" or rel == "def_and_ball":
            W_dist = torch.exp(-dist * 0.15)
            
            dir_vec = (s_pos.unsqueeze(1) - d_pos.unsqueeze(0)) / (dist.unsqueeze(-1) + 1e-6)  # (Ns,Nd,2)
            v_s = nodes[s_idx, 2:4] * torch.tensor([x_scale, y_scale], device=nodes.device)
            v_s = v_s.unsqueeze(1)                  # (Ns,1,2)
            W_approach = (v_s * dir_vec).sum(dim=-1)
            
            if rel == "attk_and_ball":
                t_pos = poss_dur[s_idx].unsqueeze(1)
                sigma = (t_pos > 0).float()
                W_possess = torch.log1p(t_pos)
        
                W_situation = sigma * W_possess + (1 - sigma) * W_approach
            else:
                W_situation = W_approach
                
            weight = W_dist + W_situation

        else:
            # temporal 등 기타 엣지
            weight = torch.ones_like(dist)
        

        # weight = torch.exp(-dist * 0.15) if d_t == 2 else 1.0 / (1.0 + dist)   # (Ns, Nd)
        mask = weight > weight_thr
        
        # # denormalization + 거리 계산
        # real_xy = (nodes[:, :2] - 0.5) * 2 * torch.tensor([x_scale, y_scale], device=nodes.device)
        # s_real = real_xy[s_idx]
        # d_real = real_xy[d_idx]
        # dist = torch.cdist(s_real, d_real)          # (Ns, Nd)
        # weight = torch.exp(-dist * 0.25) if d_t == 2 else 1.0 / (1.0 + dist)
        # mask = weight > weight_thr

        if not mask.any():
            edge_index_dict[("Node", rel, "Node")] = torch.empty((2, 0), dtype=torch.long)
            edge_attr_dict [("Node", rel, "Node")] = torch.empty((0, 1), dtype=torch.float32)
            return

        src_ids, dst_ids = mask.nonzero(as_tuple=True)          # (E,), (E,)
        edge_index = torch.stack([s_idx[src_ids], d_idx[dst_ids]], dim=0)  # (2, E)
        edge_attr  = weight[mask].unsqueeze(1).float()                        # **(E, 1)**

        if rel != "temporal":
            rev_index = edge_index.flip(0)
            rev_attr = edge_attr.clone()
            edge_index = torch.cat([edge_index, rev_index], dim=1)
            edge_attr = torch.cat([edge_attr, rev_attr], dim=0)

        edge_index_dict[("Node", rel, "Node")] = edge_index
        edge_attr_dict[("Node", rel, "Node")] = edge_attr

    make_edges(0, 0, "attk_and_attk")
    make_edges(0, 1, "attk_and_def")
    make_edges(1, 1, "def_and_def")
    make_edges(0, 2, "attk_and_ball")
    make_edges(1, 2, "def_and_ball")

    return edge_index_dict, edge_attr_dict


def convert_to_hetero_graph(node_features, edge_index_dict, edge_attr_dict):
    data = HeteroData()
    for node_type, feats in node_features.items():
        data[node_type].x = feats
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr_dict[edge_type]
    return data


def build_graph_sequence_from_condition(sample):
    condition = sample["condition"]     # [T, F]
    T = condition.shape[0]

    full_graph = HeteroData()
    node_offset = 0

    added_rels = set()

    for t in range(T):
        # print(f"[Frame {t+1}/{T}] node_offset_before={node_offset}")
        node_feats = extract_node_features(condition[t], sample["condition_columns"])
        edge_index_dict, edge_attr_dict = build_edges_based_on_interactions(node_feats, sample["pitch_scale"])

        node_count = node_feats["Node"].size(0)
        if t == 0:
            full_graph["Node"].x = node_feats["Node"]
        else:
            full_graph["Node"].x = torch.cat(
                [full_graph["Node"].x, node_feats["Node"]], dim=0)

        for rel, eidx in edge_index_dict.items():
            offset_eidx = eidx + node_offset
            # rel is tuple like ("Node","attk_and_def","Node")
            if rel in added_rels:
                full_graph[rel].edge_index = torch.cat(
                    [full_graph[rel].edge_index, offset_eidx], dim=1)
                full_graph[rel].edge_attr = torch.cat(
                    [full_graph[rel].edge_attr, edge_attr_dict[rel]], dim=0)
            else:
                full_graph[rel].edge_index = offset_eidx
                full_graph[rel].edge_attr = edge_attr_dict[rel]
                added_rels.add(rel)

        if t > 0:
            prev_offset = node_offset - node_count
            tem_edges = torch.stack([
                torch.arange(node_count, device=node_feats["Node"].device) + prev_offset,
                torch.arange(node_count, device=node_feats["Node"].device) + node_offset
            ])
            rel = ("Node", "temporal", "Node")
            ones_attr = torch.ones((node_count,1), device=tem_edges.device)
            if rel in added_rels:
                full_graph[rel].edge_index = torch.cat(
                    [full_graph[rel].edge_index, tem_edges], dim=1)
                full_graph[rel].edge_attr = torch.cat(
                    [full_graph[rel].edge_attr, ones_attr], dim=0)
            else:
                full_graph[rel].edge_index = tem_edges
                full_graph[rel].edge_attr = ones_attr
                added_rels.add(rel)

        node_offset += node_count

    return full_graph

