import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch as GeoBatch

# Return related feature columns for given x/y columns
def get_related_features(columns, all_cols):
        related = set()
        for col in columns:
            if col.endswith("_x") or col.endswith("_y"):
                prefix = col.rsplit("_", 1)[0]  # 예: Home_3
                for suffix in ["_x", "_y", "_vx", "_vy", "_dist"]:
                    extended_col = f"{prefix}{suffix}"
                    if extended_col in all_cols:
                        related.add(extended_col)
        return related
    
# Sort given columns based on their original order in the dataset
def sort_columns_by_original_order(columns, original_order):
    return [col for col in original_order if col in columns]


# Return x/y columns of valid players for a given team
def get_valid_player_columns_in_order(segment, team_prefix, original_order):
    valid_columns = []
    for col in original_order:
        if col.startswith(team_prefix) and (col.endswith("_x") or col.endswith("_y")):
            pid = col.rsplit("_", 1)[0]
            col_x = f"{pid}_x"
            col_y = f"{pid}_y"
            if (col_x in segment.columns and col_y in segment.columns and
                not np.isnan(segment[col_x].values).any() and 
                not np.isnan(segment[col_y].values).any()):
                if col_x not in valid_columns:
                    valid_columns.append(col_x)
                if col_y not in valid_columns:
                    valid_columns.append(col_y)
    return valid_columns


# Compute cumulative distance traveled for each player
def compute_cumulative_distances(df, team_prefix):
    result = pd.DataFrame(index=df.index)
    player_ids = sorted(set(int(col.split("_")[1]) for col in df.columns if col.startswith(team_prefix) and col.endswith("_x")))

    for pid in player_ids:
        x = df[f"{team_prefix}_{pid}_x"].values
        y = df[f"{team_prefix}_{pid}_y"].values

        valid_mask = ~np.isnan(x) & ~np.isnan(y)

        cumulative_d = np.full_like(x, np.nan)

        if valid_mask.sum() > 1:
            valid_indices = np.where(valid_mask)[0]
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]

            dx = np.diff(x_valid)
            dy = np.diff(y_valid)
            d = np.sqrt(dx**2 + dy**2)
            cum_d = np.insert(np.cumsum(d), 0, 0)

            cumulative_d[valid_indices] = cum_d

        result[f"{team_prefix}_{pid}_dist"] = cumulative_d

    return result


# Infer starting players by checking if their first frame has valid x/y
def infer_starters_from_tracking(df_tracking, team_prefix, num_players, offset=0):
    starters = []
    for i in range(offset + 1, num_players + offset + 1):
        col_x = f"{team_prefix}_{i}_x"
        col_y = f"{team_prefix}_{i}_y"
        is_starter = not (pd.isna(df_tracking.iloc[0][col_x]) or pd.isna(df_tracking.iloc[0][col_y]))
        starters.append(is_starter)
    return starters


# Split dataset indices into train/test sets by match ID
def split_dataset_indices(dataset, val_ratio=(1/6), test_ratio=(1/6), random_seed=42):
    match_to_indices = defaultdict(list)
    for idx, (match_id, _, _, _) in enumerate(dataset.samples):
        match_to_indices[match_id].append(idx)

    match_ids = list(match_to_indices.keys())
    match_ids.sort()
    random.seed(random_seed)
    random.shuffle(match_ids)

    num_matches = len(match_ids)
    num_val_matches = max(1, int(num_matches * val_ratio))
    num_test_matches = max(1, int(num_matches * test_ratio))
    num_train_matches = num_matches - num_val_matches - num_test_matches

    train_match_ids = set(match_ids[:num_train_matches])
    val_match_ids = set(match_ids[num_train_matches:num_train_matches + num_val_matches])
    test_match_ids = set(match_ids[num_train_matches + num_val_matches:])

    train_indices = sorted([i for m in train_match_ids for i in match_to_indices[m]])
    val_indices = sorted([i for m in val_match_ids for i in match_to_indices[m]])
    test_indices = sorted([i for m in test_match_ids for i in match_to_indices[m]])

    return train_indices, val_indices, test_indices

def compute_train_zscore_stats(dataset, train_indices, save_path="train_zscore_stats.pkl"):
    # 있으면 쓰고
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            stats = pickle.load(f)
        return stats
    
    # 없으면 새로 계산
    train_subset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    
    px, py, pvx, pvy = [], [], [], []
    bx, by, bvx, bvy = [], [], [], []
    dists = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Calculate Z-score...")):
        target = batch["target"]
        target_cols = batch["target_columns"][0]
        for i, col in enumerate(target_cols):
            arr = target[..., i].flatten().cpu().numpy()
            arr = arr[~np.isnan(arr)]
            if col.endswith('_x'):
                px.extend(arr.tolist())
            elif col.endswith('_y'):
                py.extend(arr.tolist())


        cond = batch["condition"]
        cond_cols = batch["condition_columns"][0]
        for i, col in enumerate(cond_cols):
            arr = cond[..., i].flatten().cpu().numpy()
            arr = arr[~np.isnan(arr)]
            if col.endswith('_x') and col != 'ball_x':
                px.extend(arr.tolist())
            elif col == 'ball_x':
                bx.extend(arr.tolist())
            elif col.endswith('_y') and col != 'ball_y':
                py.extend(arr.tolist())
            elif col == 'ball_y':
                by.extend(arr.tolist())
            elif col.endswith('_vx') and col != 'ball_vx':
                pvx.extend(arr.tolist())
            elif col == 'ball_vx':
                bvx.extend(arr.tolist())
            elif col.endswith('_vy') and col != 'ball_vy':
                pvy.extend(arr.tolist())
            elif col == 'ball_vy':
                bvy.extend(arr.tolist())
            elif col.endswith('_dist'):
                dists.extend(arr.tolist())
    
    stats = {}

    stats['player_x_mean'], stats['player_x_std'] = float(np.mean(px)), float(np.std(px))
    stats['player_y_mean'], stats['player_y_std'] = float(np.mean(py)), float(np.std(py))

    stats['ball_x_mean'], stats['ball_x_std'] = float(np.mean(bx)), float(np.std(bx))
    stats['ball_y_mean'], stats['ball_y_std'] = float(np.mean(by)), float(np.std(by))

    if pvx and pvy:
        stats['player_vx_mean'], stats['player_vx_std'] = float(np.mean(pvx)), float(np.std(pvx))
        stats['player_vy_mean'], stats['player_vy_std'] = float(np.mean(pvy)), float(np.std(pvy))

    if bvx and bvy:
        stats['ball_vx_mean'], stats['ball_vx_std'] = float(np.mean(bvx)), float(np.std(bvx))
        stats['ball_vy_mean'], stats['ball_vy_std'] = float(np.mean(bvy)), float(np.std(bvy))
        
    stats['dist_mean'], stats['dist_std'] = float(np.mean(dists)), float(np.std(dists))
    
    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Z-score statistics saved to {save_path}")
    print(f"Player X: mean={stats['player_x_mean']:.2f}m, std={stats['player_x_std']:.2f}m")
    print(f"Player Y: mean={stats['player_y_mean']:.2f}m, std={stats['player_y_std']:.2f}m")
    print(f"Ball X: mean={stats['ball_x_mean']:.2f}m, std={stats['ball_x_std']:.2f}m")
    print(f"Ball Y: mean={stats['ball_y_mean']:.2f}m, std={stats['ball_y_std']:.2f}m")
    if 'player_vx_mean' in stats:
        print(f"Player Vx: mean={stats['player_vx_mean']:.2f}m/s, std={stats['player_vx_std']:.2f}m/s")
        print(f"Player Vy: mean={stats['player_vy_mean']:.2f}m/s, std={stats['player_vy_std']:.2f}m/s")
    if 'ball_vx_mean' in stats:
        print(f"Ball Vx: mean={stats['ball_vx_mean']:.2f}m/s, std={stats['ball_vx_std']:.2f}m/s")
        print(f"Ball Vy: mean={stats['ball_vy_mean']:.2f}m/s, std={stats['ball_vy_std']:.2f}m/s")
    print(f"Dist: mean={stats['dist_mean']:.2f}m, std={stats['dist_std']:.2f}m")

    return stats


# Custom collate function to batch
def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if key in ("other_columns", "target_columns", "condition_columns"):
            collated[key] = [b[key] for b in batch]
        elif key == "graph":
            collated[key] = GeoBatch.from_data_list([b["graph"] for b in batch])
        elif key == "pitch_scale":
            collated[key] = [b["pitch_scale"] for b in batch]
        else:
            try:
                collated[key] = default_collate([b[key] for b in batch])
            except Exception:
                collated[key] = [b[key] for b in batch]
    return collated



