import os
import ast
import random
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import warnings
warnings.filterwarnings("ignore", message="The 'gameclock' column does not match the defined value range.*", category=UserWarning, module=r"floodlight\.core\.events")

from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_pitch_from_mat_info_xml
from utils.utils import calc_velocites, correct_nan_velocities_and_positions
from utils.data_utils import (
    infer_starters_from_tracking,
    sort_columns_by_original_order,
    get_valid_player_columns_in_order,
    compute_cumulative_distances
)
from utils.graph_utils import build_graph_sequence_from_condition


# .xml files in DFL -> .csv with Metrica_sports format
def convert_dfl_to_df(xy_objects, team, half, offset):
    tracking = xy_objects[half][team].xy
    ball = xy_objects[half]["Ball"].xy
    framerate = xy_objects[half][team].framerate
    n_frames, n_coords = tracking.shape
    n_players = n_coords // 2
    player_ids = np.arange(1, n_players + 1) + offset
    time = np.arange(n_frames) / framerate
    period = 1 if half == "firstHalf" else 2

    # Metrica_sports format Baseline
    df = pd.DataFrame(np.hstack([tracking, ball]), columns=[
        *[f"{team}_{i}_{ax}" for i in player_ids for ax in ["x", "y"]],
        "ball_x", "ball_y"
    ])
    df.insert(0, "Period", period)
    df.insert(1, "Time [s]", time)
    df.index.name = "Frame"
    
    # Sorting player columns
    player_cols = [col for col in df.columns if col.startswith(f"{team}_")]
    player_cols = sorted(player_cols, key=lambda x: (int(x.split("_")[1]), x.split("_")[2]))
    df = df[["Period", "Time [s]"] + player_cols + ["ball_x", "ball_y"]]
    
    return df

# Convert data to Series
def get_series(obj, key, half, offset=0, name="data"):
    data = obj[half].code.flatten()
    return pd.Series(data, index=np.arange(len(data)) + offset, name=name)


# Concat 1st half, 2nd half
def process_match(xy, possession, ballstatus):
    df_home_1 = convert_dfl_to_df(xy, "Home", "firstHalf", 0)
    player_cols = [col for col in df_home_1.columns if col.startswith("Home_") and (col.endswith("_x") or col.endswith("_y"))]
    num_players = len(player_cols) // 2
    df_away_1 = convert_dfl_to_df(xy, "Away", "firstHalf", num_players)
    df_home_2 = convert_dfl_to_df(xy, "Home", "secondHalf", 0)
    df_away_2 = convert_dfl_to_df(xy, "Away", "secondHalf", num_players)

    offset = df_home_1.index.max() + 1
    time_offset = df_home_1["Time [s]"].iloc[-1]
    for df in [df_home_2, df_away_2]:
        df.index += offset
        df["Time [s]"] += time_offset

    home = pd.concat([df_home_1, df_home_2])
    away = pd.concat([df_away_1, df_away_2])
    
    # Calculate Match time
    home["match_time"] = away["match_time"] = home["Time [s]"]
    home.loc[home["Period"] == 2, "match_time"] -= time_offset
    away.loc[away["Period"] == 2, "match_time"] -= time_offset

    # Add 'ball_active', 'ball_possession' (for team)
    active = pd.concat([
        get_series(ballstatus, "active", "firstHalf"),
        get_series(ballstatus, "active", "secondHalf", offset)
    ])
    poss = pd.concat([
        get_series(possession, "possession", "firstHalf"),
        get_series(possession, "possession", "secondHalf", offset)
    ])
    for df in [home, away]:
        df["active"] = active
        df["possession"] = poss

    return home, away

def make_ball_holder_series(event_objects, half, framerate, n_frames, offset=0):
    holder = [None] * (n_frames + offset)
    all_events = pd.concat(
        [ev_obj.events for ev_obj in event_objects[half].values()],
        ignore_index=True
    ).sort_values(['minute','second']).reset_index(drop=True)

    current = None
    for _, ev in all_events.iterrows():
        qd = ast.literal_eval(ev['qualifier']) if isinstance(ev['qualifier'], str) else ev['qualifier']
        eid = ev['eID']
        if 'Recipient' in qd:
            newp = qd['Recipient']
        elif eid == 'TacklingGame' and 'PossessionChange' in qd:
            if qd['PossessionChange']==1 and 'Winner' in qd:
                newp = qd['Winner']
            elif 'Loser' in qd:
                newp = qd['Loser']
            else:
                newp = current
        elif 'Player' in qd and eid not in ('Delete','FinalWhistle','VideoAssistantAction'):
            newp = qd['Player']
        else:
            newp = None

        if newp is not None:
            current = newp
        frm = int((ev['minute']*60 + ev['second']) * framerate) + offset
        if 0 <= frm < len(holder):
            holder[frm] = current

    for i in range(offset+1, offset+n_frames):
        if holder[i] is None:
            holder[i] = holder[i-1]

    durations = [0.0] * (n_frames + offset)
    durations[offset] = 1.0 / framerate
    for i in range(offset+1, offset+n_frames):
        if holder[i] == holder[i-1]:
            durations[i] = durations[i-1] + 1.0 / framerate
        else:
            durations[i] = 1.0/framerate

    idx = list(range(offset, offset+n_frames))
    holder_s = pd.Series(holder[offset:offset+n_frames], index=idx, name="holder")
    dur_s = pd.Series(np.log1p(durations[offset:offset+n_frames]), index=idx, name="possession_duration")
    
    return holder_s, dur_s


# Save DFL .xml files as .csv format
def organize_and_process(data_path, save_path):
    # Searching Folder
    files = [f for f in os.listdir(data_path) if f.endswith(".xml")]
    for f in files:
        match_id = f.split("_")[-1].split(".")[0]
        match_dir = os.path.join(data_path, match_id)
        os.makedirs(match_dir, exist_ok=True)
        shutil.move(os.path.join(data_path, f), os.path.join(match_dir, f))

    # Preprocessing for each folder
    def _convert_match(match_id):
        match_dir = os.path.join(data_path, match_id)
        if not os.path.isdir(match_dir): 
            return

        pos, info, events = None, None, None
        for fname in os.listdir(match_dir):
            if "positions_raw" in fname: pos = fname
            elif "matchinformation" in fname: info = fname
            elif "events_raw" in fname: events = fname

        if not (pos and info and events):
            return

        xy, poss, ball, teamsheets, _ = read_position_data_xml(
            os.path.join(match_dir, pos),
            os.path.join(match_dir, info)
        )
        home, away = process_match(xy, poss, ball)

        save_match_dir = os.path.join(save_path, match_id)
        os.makedirs(save_match_dir, exist_ok=True)
        home.to_csv(os.path.join(save_match_dir, "tracking_home.csv"))
        away.to_csv(os.path.join(save_match_dir, "tracking_away.csv"))

        # player_info.csv 생성
        position_mapping = {
            "TW": 1, "LV": 2, "IVL": 3, "IVZ": 4, "IVR": 5, "RV": 6,
            "DML": 7, "DMZ": 8, "DMR": 9,
            "LM": 10, "HL": 11, "MZ": 12, "HR": 13, "RM": 14,
            "OLM": 15, "ZO": 16, "ORM": 17,
            "LA": 18, "STL": 19, "HST": 20, "STZ": 21, "STR": 22, "RA": 23
        }
        player_info_rows = []
        for team in ["Home", "Away"]:
            df_team = teamsheets[team].teamsheet.reset_index(drop=True)
            tracking_df = home if team == "Home" else away
            base_offset = 1 if team == "Home" else 21
            num_players = len(df_team)
            starters = infer_starters_from_tracking(
                tracking_df, team, num_players, offset=base_offset - 1
            )
            for i, row in df_team.iterrows():
                col_name = f"{team}_{base_offset + i}"
                pos_num = position_mapping.get(row["position"], 0)
                is_start = 1 if starters[i] else 0

                if f"{col_name}_x" in tracking_df.columns and f"{col_name}_y" in tracking_df.columns:
                    pts = tracking_df[[f"{col_name}_x", f"{col_name}_y"]].dropna()
                    start_f = int(pts.index.min()) if not pts.empty else None
                    end_f   = int(pts.index.max()) if not pts.empty else None
                else:
                    start_f = end_f = None

                player_info_rows.append({
                    "col_name": col_name,
                    "position": pos_num,
                    "starter": is_start,
                    "pID": row["pID"],
                    "start_frame": start_f,
                    "end_frame": end_f
                })

        df_pi = pd.DataFrame(player_info_rows)
        df_pi.to_csv(os.path.join(save_match_dir, "player_info.csv"), index=False)

        # 매치정보 XML 복사
        shutil.copy(
            os.path.join(match_dir, info),
            os.path.join(save_match_dir, "matchinformation.xml")
        )
        
        shutil.copy(
            os.path.join(match_dir, events),
            os.path.join(save_match_dir, "events_raw.xml")
        )
        
    match_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for mid in match_ids:
        _convert_match(mid)


class CustomDataset(Dataset):
    def __init__(self, data_root, segment_length=200, condition_length=100, framerate=25, stride=25, zscore_stats = None):
        self.data_root = data_root
        self.segment_length = segment_length
        self.condition_length = condition_length
        self.framerate = framerate
        self.stride = stride
        self.zscore_stats = zscore_stats
        
        self.match_events = {}
        self.match_player_pid_map = {}
        self.samples = []
        self.match_data = {}
        self.column_order = None
        self.load_all_matches(data_root)
        self.graph_cache = {}
    
    # Preprocess raw match data and extract valid trajectory segments
    def load_all_matches(self, data_root):
        match_ids = os.listdir(data_root)
        skip_ids = {"DFL-MAT-J03WN1"}  # Skip matches with insufficient data
        match_ids = [m for m in match_ids if m not in skip_ids]

        for match_id in tqdm(match_ids, desc="Loading Matches"):
            folder = os.path.join(data_root, match_id)
            # CSV 로드
            home = pd.read_csv(os.path.join(folder, "tracking_home.csv"), index_col="Frame")
            away = pd.read_csv(os.path.join(folder, "tracking_away.csv"), index_col="Frame")
            
            # Event Data 로드
            events_fname = next(f for f in os.listdir(folder) if "events" in f and f.endswith(".xml"))
            info_fname = next(f for f in os.listdir(folder) if "matchinformation" in f and f.endswith(".xml"))
            events_path = os.path.join(folder, events_fname)
            info_path = os.path.join(folder, info_fname)
            
            events_objects, teamsheets, _ = read_event_data_xml(events_path, info_path)
            self.match_events[match_id] = events_objects
            
            pid_map = {}

            home_sheet = teamsheets["Home"].teamsheet.reset_index(drop=True)
            for i, row in home_sheet.iterrows():
                base = f"Home_{i+1}"
                pid_map[base] = row["pID"]

            offset = len(home_sheet)
            away_sheet = teamsheets["Away"].teamsheet.reset_index(drop=True)
            for i, row in away_sheet.iterrows():
                base = f"Away_{offset + i + 1}"
                pid_map[base] = row["pID"]

            self.match_player_pid_map[match_id] = pid_map
            
            # 전처리
            home = calc_velocites(home)
            away = calc_velocites(away)
            home = correct_nan_velocities_and_positions(home, self.framerate)
            away = correct_nan_velocities_and_positions(away, self.framerate)
            home_dist = compute_cumulative_distances(home, "Home")
            away_dist = compute_cumulative_distances(away, "Away")

            # 공통/팀별 컬럼 합치기
            common_cols = ['Period', 'Time [s]', 'match_time', 'active', 'possession']
            common = home[common_cols]
            home_only = home.drop(columns=common_cols).drop(
                columns=['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_speed']
            )
            away_only = away.drop(columns=common_cols)
            df = pd.concat([common, home_only, away_only, home_dist, away_dist], axis=1)

            # 세그먼트 정보 추출
            segs = self.extract_segments_info(df, match_id)
            if not segs:
                continue

            # 최초 한 번만 컬럼 순서 기록
            if self.column_order is None:
                self.column_order = df.columns.tolist()

            # 데이터 저장
            self.match_data[match_id] = df
            self.samples.extend(segs)

    def extract_segments_info(self, df, match_id):
        if self.column_order is None:
            self.column_order = df.columns.tolist()
            
        segments_info = []
        num_frames = len(df)
        possession_array = df["possession"].values
        active_array = df["active"].values
        ball_x_valid = ~np.isnan(df["ball_x"].values)
        ball_y_valid = ~np.isnan(df["ball_y"].values)
        valid_mask = (active_array == 1) & ball_x_valid & ball_y_valid
        segments = []
        current_start = None
        for i in range(num_frames):
            if not valid_mask[i] or pd.isna(possession_array[i]):
                if current_start is not None:
                    segments.append((current_start, i - 1))
                    current_start = None
            else:
                if current_start is None or possession_array[i] != possession_array[current_start]:
                    if current_start is not None:
                        segments.append((current_start, i - 1))
                    current_start = i
        if current_start is not None and current_start < num_frames - self.segment_length:
            segments.append((current_start, num_frames - 1))
        for start, end in segments:
            if end - start + 1 < self.segment_length:
                continue
            i = start
            while i <= end - self.segment_length:
                segment = df.iloc[i:i + self.segment_length]
                possession_team = possession_array[i]
                # Distinguishing Attk team / Def team
                if possession_team == 1:
                    atk_prefix, def_prefix = "Home", "Away"
                elif possession_team == 2:
                    atk_prefix, def_prefix = "Away", "Home"
                else:
                    i += 1
                    continue
                atk_cols = get_valid_player_columns_in_order(segment, atk_prefix, self.column_order)
                def_cols = get_valid_player_columns_in_order(segment, def_prefix, self.column_order)
                if len(atk_cols) != 22 or len(def_cols) != 22:
                    i += 1
                    continue
                    
                input_feats = sort_columns_by_original_order(["ball_x", "ball_y"] + atk_cols, self.column_order)
                target_feats = sort_columns_by_original_order(def_cols, self.column_order)
                segments_info.append((match_id, i, input_feats, target_feats))
                i += self.stride
                
        return segments_info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        match_id, start_idx, other_columns, target_columns = self.samples[idx]
        df = self.match_data[match_id]
        
        first_len = (df["Period"] == 1).sum()
        second_len = len(df) - first_len
        offset = first_len

        holder_first, duration_first = make_ball_holder_series(self.match_events[match_id], "firstHalf", self.framerate, first_len, offset=0)
        holder_second, duration_second = make_ball_holder_series(self.match_events[match_id], "secondHalf", self.framerate, second_len, offset=offset)
        holder = pd.concat([holder_first, holder_second])
        poss_time = pd.concat([duration_first, duration_second])
        
        full_seq = df.iloc[start_idx:start_idx + self.segment_length]
        condition_seq = full_seq.iloc[:self.condition_length].copy()
        future_seq = full_seq.iloc[self.condition_length:].copy()
        
        # Determine team roles (attacking or defending) based on possession
        possession_team = df.iloc[start_idx]["possession"]
        atk_prefix, def_prefix = ("Home", "Away") if possession_team == 1 else ("Away", "Home")

        atk_cols = get_valid_player_columns_in_order(full_seq, atk_prefix, self.column_order)
        def_cols = get_valid_player_columns_in_order(full_seq, def_prefix, self.column_order)

        if len(atk_cols) != 22 or len(def_cols) != 22:
            raise ValueError("Invalid number of valid players in segment")

        atk_bases = sorted({c.rsplit("_",1)[0] for c in atk_cols}, key=lambda b: int(b.split("_")[1]))
        def_bases = sorted({c.rsplit("_",1)[0] for c in def_cols}, key=lambda b: int(b.split("_")[1]))
        player_bases = atk_bases + def_bases
        ball_feats = ["ball_x", "ball_y", "ball_vx", "ball_vy"]
        
        # Get pitch scale
        if not hasattr(self, "pitch_cache"):
            self.pitch_cache = {}
        if match_id not in self.pitch_cache:
            info_path = os.path.join(self.data_root, match_id, "matchinformation.xml")
            pitch = read_pitch_from_mat_info_xml(info_path)
            self.pitch_cache[match_id] = (pitch.length / 2, pitch.width / 2)
            
        x_scale, y_scale = self.pitch_cache[match_id]
        
        # Collect feature columns
        condition_columns = set()
        for base in player_bases:
            for feat in ["x", "y", "vx", "vy", "dist"]:
                col = f"{base}_{feat}"
                if col in df.columns:
                    condition_columns.add(col)
            condition_columns.add(f"{base}_possession_duration")
            condition_columns.add(f"{base}_neighbor_count")
        for col in ball_feats:
            if col in df.columns:
                condition_columns.add(col)

        # Sort columns
        if not hasattr(self, "column_order"):
            self.column_order = df.columns.tolist()
        
        condition_columns = sort_columns_by_original_order(condition_columns, self.column_order)
        condition_seq = condition_seq[condition_columns]

        poss_seq = poss_time.iloc[start_idx:start_idx + self.condition_length].reset_index(drop=True)
        # Load player metadata
        if not hasattr(self, "player_info_cache"):
            self.player_info_cache = {}

        if match_id not in self.player_info_cache:
            player_info_path = os.path.join(self.data_root, match_id, "player_info.csv")
            self.player_info_cache[match_id] = pd.read_csv(player_info_path)

        player_info = self.player_info_cache[match_id]
        player_info_map = player_info.set_index("col_name")[["position", "starter", "pID"]].to_dict("index")

        # other: Attk + ball
        # target: Def
        other_seq = future_seq[other_columns]
        target_seq = future_seq[target_columns]
        
        # --- 1. CONDITION & OTHER 정규화 먼저 수행 ---
        if self.zscore_stats is not None:
            # Condition 정규화
            condition_seq_normalized = condition_seq.copy()
            for col in condition_columns:
                base, feat = col.rsplit("_", 1)
                if feat in ("x", "y", "vx", "vy"):
                    key = "ball" if base == "ball" else "player"
                    mean = self.zscore_stats[f"{key}_{feat}_mean"]
                    std  = self.zscore_stats[f"{key}_{feat}_std"]
                    condition_seq_normalized[col] = (condition_seq[col] - mean) / std
                elif feat == "dist":
                    condition_seq_normalized[col] = (condition_seq[col] - self.zscore_stats["dist_mean"]) / self.zscore_stats["dist_std"]
            
            # Other 정규화
            other_array = other_seq.values.copy()
            for i, col in enumerate(other_columns):
                base, feat = col.rsplit("_", 1)
                if feat in ("x", "y"):
                    key = "ball" if base == "ball" else "player"
                    mean = self.zscore_stats[f"{key}_{feat}_mean"]
                    std = self.zscore_stats[f"{key}_{feat}_std"]
                    other_array[:, i] = (other_array[:, i] - mean) / std
                elif feat in ("vx", "vy"):
                    key = "ball" if base == "ball" else "player"
                    mean = self.zscore_stats[f"{key}_{feat}_mean"]
                    std = self.zscore_stats[f"{key}_{feat}_std"]
                    other_array[:, i] = (other_array[:, i] - mean) / std
        else:
            # 피치 스케일 정규화 (기본값)
            condition_seq_normalized = condition_seq
            other_array = other_seq.values
        
        other_tensor = torch.tensor(other_array, dtype=torch.float32)

        reference_frame = condition_seq_normalized.iloc[-1]

        target_abs_data = []
        target_rel_data = []
        target_vel_data = []
        
        num_players = len(target_columns) // 2  # 11명

        if self.zscore_stats is not None:
            target_abs_normalized = np.zeros((len(target_seq), len(target_columns)))
            for i in range(0, len(target_columns), 2):
                col_x = target_columns[i]
                col_y = target_columns[i + 1]

                target_abs_normalized[:, i] = (target_seq[col_x].values - self.zscore_stats['player_x_mean']) / self.zscore_stats['player_x_std']
                target_abs_normalized[:, i+1] = (target_seq[col_y].values - self.zscore_stats['player_y_mean']) / self.zscore_stats['player_y_std']
        else:
            target_abs_normalized = target_seq.values

        for i in range(num_players):
            abs_x = target_abs_normalized[:, i * 2]
            abs_y = target_abs_normalized[:, i * 2 + 1]
            target_abs_data.append(np.column_stack([abs_x, abs_y]))

            col_x = target_columns[i * 2]
            col_y = target_columns[i * 2 + 1]
            ref_x = reference_frame[col_x]
            ref_y = reference_frame[col_y] 
            rel_x = (abs_x - ref_x)
            rel_y = (abs_y - ref_y)
            
            target_rel_data.append(np.column_stack([rel_x, rel_y]))

            # 첫 번째 프레임 속도: (첫 프레임 위치 - 기준점) * framerate
            v0_x = (abs_x[0] - ref_x) * self.framerate
            v0_y = (abs_y[0] - ref_y) * self.framerate
            
            # 나머지 프레임 속도: 연속 프레임 간 차이 * framerate  
            v_rest_x = np.diff(abs_x) * self.framerate
            v_rest_y = np.diff(abs_y) * self.framerate
            # 전체 속도 결합
            vx = np.concatenate([[v0_x], v_rest_x])
            vy = np.concatenate([[v0_y], v_rest_y])
            target_vel_data.append(np.column_stack([vx, vy]))

        target_abs = np.concatenate(target_abs_data, axis=1)  # [T, 22] (11명 * 2차원)
        target_rel = np.concatenate(target_rel_data, axis=1)  # [T, 22]
        target_vel = np.concatenate(target_vel_data, axis=1)  # [T, 22]
        
        # 텐서로 변환
        target_abs_tensor = torch.tensor(target_abs, dtype=torch.float32)
        target_rel_tensor = torch.tensor(target_rel, dtype=torch.float32)
        target_vel_tensor = torch.tensor(target_vel, dtype=torch.float32)
                
        # Calculate possession duration & neighbor opposite player count
        Na = len(atk_bases)
        Nd = len(def_bases)

        xs = condition_seq[[f"{b}_x" for b in player_bases]].values
        ys = condition_seq[[f"{b}_y" for b in player_bases]].values
    
        coords = np.stack([xs, ys], axis=-1)
        diff2  = ((coords[:, :, None, :] - coords[:, None, :, :])**2).sum(-1)

        neighbor_radius = 5.0
        r2 = neighbor_radius**2

        T, N = diff2.shape[:2]
        neighbor_counts = np.zeros((T, N), dtype=int)

        neighbor_counts[:, :Na] = (diff2[:, :Na, Na:] <= r2).sum(axis=2)
        neighbor_counts[:, Na:] = (diff2[:, Na:, :Na] <= r2).sum(axis=2)
        
        T = self.condition_length
        start = start_idx
        holder_slice = holder[start:start+T].reset_index(drop=True)
        poss_slice = poss_time[start:start+T].reset_index(drop=True)
        
        bases = player_bases
        N = len(bases)
        
        # features
        num_feats_list = []
        for f in ["x", "y", "vx", "vy", "dist"]:
            cols = [f"{base}_{f}" for base in bases]
            num_feats_list.append(condition_seq_normalized[cols].fillna(0).values[..., None])
        num_feats = np.concatenate(num_feats_list, axis=2)

        # position, starter
        pos_arr = np.array([player_info_map[b]["position"] for b in bases], dtype=np.float32)
        starter_arr = np.array([player_info_map[b]["starter"]  for b in bases], dtype=np.float32)
        pos_feats = np.broadcast_to(pos_arr, (T, N))[..., None]
        starter_feats = np.broadcast_to(starter_arr, (T, N))[..., None]

        # possession
        pid_arr = np.array([self.match_player_pid_map[match_id][b] for b in bases])
        holder_v = holder_slice.values
        poss_v = poss_slice.values
        mask = (holder_v[:, None] == pid_arr[None, :])
        poss_feats = (mask * poss_v[:, None])[..., None]

        # N_opp
        neigh_feats = (neighbor_counts / 11.0)[..., None]

        # Ball features
        ball_cols = ["ball_x","ball_y","ball_vx","ball_vy"]
        ball_arr = condition_seq_normalized[ball_cols].fillna(0).values

        # Concat
        player_feats = np.concatenate([num_feats, pos_feats, starter_feats, poss_feats, neigh_feats], axis=2)
        player_flat = player_feats.reshape(T, N * player_feats.shape[2])
        cond_arr = np.concatenate([player_flat, ball_arr], axis=1)
        
        condition_tensor = torch.tensor(cond_arr, dtype=torch.float32)

        # 컬럼명 생성 (분리된 형태로)
        target_abs_columns = []
        target_rel_columns = []
        target_vel_columns = []
        
        for i in range(num_players):
            base_name = target_columns[i * 2].rsplit('_', 1)[0]  # e.g., "Away_15"
            target_abs_columns.extend([f"{base_name}_x", f"{base_name}_y"])
            target_rel_columns.extend([f"{base_name}_rel_x", f"{base_name}_rel_y"])
            target_vel_columns.extend([f"{base_name}_vx", f"{base_name}_vy"])

        sample = {
            "match_id": match_id,
            "condition": condition_tensor,
            "other": other_tensor,
            "target": target_abs_tensor,           # 절대좌표
            "target_relative": target_rel_tensor,  # 상대좌표
            "target_velocity": target_vel_tensor,  # 속도
            "condition_columns": [
                f"{base}_{f}" for base in player_bases for f in ["x", "y", "vx", "vy", "dist", "position", "starter", "possession_duration", "neighbor_count"]
            ] + ball_feats,
            "other_columns": other_columns,
            "target_columns": target_abs_columns,           # 절대좌표 컬럼명
            "target_relative_columns": target_rel_columns,  # 상대좌표 컬럼명
            "target_velocity_columns": target_vel_columns,  # 속도 컬럼명
            "condition_frames": list(condition_seq.index),
            "target_frames": list(future_seq.index),
            "pitch_scale": (x_scale, y_scale),
            "zscore_stats": self.zscore_stats
        }
        if idx not in self.graph_cache:
            self.graph_cache[idx] = build_graph_sequence_from_condition({
                "condition": condition_tensor,
                "condition_columns": sample["condition_columns"],
                "pitch_scale": sample["pitch_scale"],
                "zscore_stats": self.zscore_stats
            })
            
        sample["graph"] = self.graph_cache[idx]
        
        return sample
    

class ApplyAugmentedDataset(Dataset):
    def __init__(self, base_dataset, flip_prob = 0.5):
        self.base = base_dataset
        if isinstance(base_dataset, Subset):
            self.zscore_stats = base_dataset.dataset.zscore_stats
        else:
            self.zscore_stats = base_dataset.zscore_stats
        self.N = len(base_dataset)
        self.flip_N = int(self.N * flip_prob)
        self.total = self.N + self.flip_N
        self.flip_indices = random.sample(range(self.N), self.flip_N)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.N:
            return self.base[idx]

        t = idx - self.N
        base_sample = self.base[self.flip_indices[t]]
        
        cond = base_sample["condition"].clone()
        cond_x_indices = [i for i, col in enumerate(base_sample["condition_columns"]) if col.endswith("_x")]
        cond_vx_indices = [i for i, col in enumerate(base_sample["condition_columns"]) if col.endswith("_vx")]
        cond[:, cond_x_indices + cond_vx_indices] *= -1

        other = base_sample["other"].clone()
        other_x_indices = [i for i, col in enumerate(base_sample["other_columns"]) if col.endswith("_x")]
        other_vx_indices = [i for i, col in enumerate(base_sample["other_columns"]) if col.endswith("_vx")]
        other[:, other_x_indices + other_vx_indices] *= -1

        # 분리된 target 데이터들 각각 augmentation
        target = base_sample["target"].clone()
        target_x_indices = [i for i, col in enumerate(base_sample["target_columns"]) if col.endswith("_x")]
        target[:, target_x_indices] *= -1

        target_relative = base_sample["target_relative"].clone()
        target_rel_x_indices = [i for i, col in enumerate(base_sample["target_relative_columns"]) if col.endswith("_rel_x")]
        target_relative[:, target_rel_x_indices] *= -1

        target_velocity = base_sample["target_velocity"].clone()
        target_vx_indices = [i for i, col in enumerate(base_sample["target_velocity_columns"]) if col.endswith("_vx")]
        target_velocity[:, target_vx_indices] *= -1

        sample = {
            "match_id": base_sample["match_id"],
            "condition": cond,
            "other": other,
            "target": target,
            "target_relative": target_relative,
            "target_velocity": target_velocity,
            "condition_columns": base_sample["condition_columns"],
            "other_columns": base_sample["other_columns"],
            "target_columns": base_sample["target_columns"],
            "target_relative_columns": base_sample["target_relative_columns"],
            "target_velocity_columns": base_sample["target_velocity_columns"],
            "condition_frames": base_sample["condition_frames"],
            "target_frames": base_sample["target_frames"],
            "pitch_scale": base_sample["pitch_scale"],
            "zscore_stats": base_sample["zscore_stats"]
        }

        sample["graph"] = build_graph_sequence_from_condition({
            "condition": sample["condition"],
            "condition_columns": sample["condition_columns"],
            "pitch_scale": sample["pitch_scale"],
            "zscore_stats": self.zscore_stats
        })

        return sample

if __name__ == "__main__":
    raw_data_path = "idsse-data" # Raw Data Downloaded Path
    data_save_path = "match_data" # Saving path for preprocessed data

    if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
        organize_and_process(raw_data_path, data_save_path)
    else:
        print("Skip organize_and_process")
    
    import pickle
    with open('./train_zscore_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    dataset = CustomDataset(data_root=data_save_path, zscore_stats=stats)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    sample = dataset[0]
    
    print(len(dataset), "samples loaded.")
    print("Match id:", sample["match_id"])
    print("Condition columns:", sample["condition_columns"])
    print("Condition shape:", sample["condition"].shape)
    print("Other columns:", sample["other_columns"])
    print("Other shape:", sample["other"].shape)
    print("Target columns:", sample["target_columns"])
    print("Target shape:", sample["target"].shape)
    print("Target relative columns:", sample["target_relative_columns"])
    print("Target relative shape:", sample["target_relative"].shape)
    print("Target velocity columns:", sample["target_velocity_columns"])
    print("Target velocity shape:", sample["target_velocity"].shape)
    print("Condition frames:", sample["condition_frames"])
    print("Using frames:", sample["target_frames"])
    
    print("Target (absolute):", sample["target"])
    print("Target relative:", sample["target_relative"])
    print("Target velocity:", sample["target_velocity"])
    
    print("Target relative mean:", sample["target_relative"].mean())
    print("Target relative min:", sample["target_relative"].min())
    print("Target relative max:", sample["target_relative"].max())
    print("Target relative std:", sample["target_relative"].std())
    
    print("Target mean:", sample["target"].mean())
    print("Target min:", sample["target"].min())
    print("Target max:", sample["target"].max())
    print("Target std:", sample["target"].std())
    
    print("Target vel mean:", sample["target_velocity"].mean())
    print("Target vel min:", sample["target_velocity"].min())
    print("Target vel max:", sample["target_velocity"].max())
    print("Target vel std:", sample["target_velocity"].std())