import os
import ast
import random
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", message="The 'gameclock' column does not match the defined value range.*", category=UserWarning, module=r"floodlight\.core\.events")

from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_pitch_from_mat_info_xml
from utils.utils import calc_velocites, correct_all_player_jumps_adjacent
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

    # forward‐fill
    for i in range(offset+1, offset+n_frames):
        if holder[i] is None:
            holder[i] = holder[i-1]

    # 2) durations 계산
    durations = [0.0] * (n_frames + offset)
    durations[offset] = 1.0 / framerate
    for i in range(offset+1, offset+n_frames):
        if holder[i] == holder[i-1]:
            durations[i] = durations[i-1] + 1.0 / framerate
        else:
            durations[i] = 1.0/framerate

    # 3) 두 시리즈를 튜플로 반환
    idx = list(range(offset, offset+n_frames))
    holder_s = pd.Series(holder[offset:offset+n_frames], index=idx, name="holder")
    dur_s    = pd.Series(durations[offset:offset+n_frames], index=idx, name="possession_duration")
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


class MultiMatchSoccerDataset(Dataset):
    def __init__(self, data_root, segment_length=250, condition_length=125, framerate=25, stride=25):
        self.data_root = data_root
        self.segment_length = segment_length
        self.condition_length = condition_length
        self.framerate = framerate
        self.stride = stride
        
        self.match_events = {}
        self.match_player_pid_map = {}
        self.samples = []
        self.match_data = {}
        self.column_order = None
        self.load_all_matches(data_root)
    
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
            home = correct_all_player_jumps_adjacent(home, self.framerate)
            away = correct_all_player_jumps_adjacent(away, self.framerate)
            home = calc_velocites(home)
            away = calc_velocites(away)
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
        off = first_len

        holder_first, duration_first = make_ball_holder_series(self.match_events[match_id], "firstHalf", self.framerate, first_len, offset=0)
        holder_second, duration_second = make_ball_holder_series(self.match_events[match_id], "secondHalf", self.framerate, second_len, offset=off)
        holder = pd.concat([holder_first, holder_second])
        poss_time = pd.concat([duration_first, duration_second])
        
        full_seq = df.iloc[start_idx:start_idx + self.segment_length]
        condition_seq = full_seq.iloc[:self.condition_length].copy()
        target_seq = full_seq.iloc[self.condition_length:].copy()
        
        # Determine team roles (attacking or defending) based on possession
        possession_team = df.iloc[start_idx]["possession"]
        atk_prefix, def_prefix = ("Home", "Away") if possession_team == 1 else ("Away", "Home")

        # Extract 22 valid players (11 per team) from current segment
        atk_cols = get_valid_player_columns_in_order(full_seq, atk_prefix, self.column_order)
        def_cols = get_valid_player_columns_in_order(full_seq, def_prefix, self.column_order)

        if len(atk_cols) != 22 or len(def_cols) != 22:
            raise ValueError("Invalid number of valid players in segment")

        atk_bases = sorted({c.rsplit("_",1)[0] for c in atk_cols}, key=lambda b: int(b.split("_")[1]))
        def_bases = sorted({c.rsplit("_",1)[0] for c in def_cols}, key=lambda b: int(b.split("_")[1]))
        player_bases = atk_bases + def_bases
        ball_feats = ["ball_x", "ball_y", "ball_vx", "ball_vy"]
        
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
        other_seq = target_seq[other_columns]
        target_seq = target_seq[target_columns]
        
        # Normalization
        if not hasattr(self, "pitch_cache"):
            self.pitch_cache = {}
        if match_id not in self.pitch_cache:
            info_path = os.path.join(self.data_root, match_id, "matchinformation.xml")
            pitch = read_pitch_from_mat_info_xml(info_path)
            self.pitch_cache[match_id] = (pitch.length / 2, pitch.width / 2)
            
        x_scale, y_scale = self.pitch_cache[match_id]

        target_seq[target_columns[0::2]] /= x_scale
        target_seq[target_columns[1::2]] /= y_scale

        # other_seq[other_columns[0::2]] = other_seq[other_columns[0::2]] / x_scale
        # other_seq[other_columns[1::2]] = other_seq[other_columns[1::2]] / y_scale

        condition_x_cols = [col for col in condition_seq.columns if col.endswith("_x")]
        condition_y_cols = [col for col in condition_seq.columns if col.endswith("_y")]
        condition_seq[condition_x_cols] /= x_scale
        condition_seq[condition_y_cols] /= y_scale

        # Normalization for other columns
        # suffixes 는 기존과 동일
        for col in condition_seq.columns:
            if col.endswith("_vx") or col.endswith("ball_vx"):
                condition_seq[col] /= x_scale                  # m/s → 1/s
            elif col.endswith("_vy") or col.endswith("ball_vy"):
                condition_seq[col] /= y_scale
            elif col.endswith("_dist"):
                condition_seq[col] /= (x_scale**2 + y_scale**2) ** 0.5 
                
        # Calculate possession duration & neighbor opposite player count
        Na = len(atk_bases)
        Nd = len(def_bases)

        xs = condition_seq[[f"{b}_x" for b in player_bases]].values
        ys = condition_seq[[f"{b}_y" for b in player_bases]].values
        
        xs *= x_scale
        ys *= y_scale
        
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
        holder_slice = holder[start : start+T].reset_index(drop=True)
        poss_slice = poss_time[start : start+T].reset_index(drop=True)
        
        # Add player's position, starter feature
        final_condition = []
        
        for i in range(len(condition_seq)):
            row = condition_seq.iloc[i].to_dict()
            final_condition_row = []
            for j, base in enumerate(player_bases):
                feats = [f"{base}_{f}" for f in ["x", "y", "vx", "vy", "dist"]]
                final_condition_row.extend([float(row[f]) if f in row and pd.notna(row[f]) else 0.0 for f in feats])
                meta = player_info_map.get(base, {"position": -1, "starter": 0})
                final_condition_row.append(float(meta["position"]))
                final_condition_row.append(float(meta["starter"]))
                
                # possession
                pid = self.match_player_pid_map[match_id][base]
                final_condition_row.append(poss_slice.iloc[i] if holder_slice.iloc[i] == pid else 0.0)
                
                # neighbor count
                final_condition_row.append(float(neighbor_counts[i, j]))
                
            final_condition_row.extend([float(row[f]) if f in row and pd.notna(row[f]) else 0.0 for f in ball_feats])
            final_condition.append(final_condition_row)

        condition_tensor = torch.tensor(final_condition, dtype=torch.float32)
        other_tensor = torch.tensor(other_seq.values, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq.values, dtype=torch.float32)

        sample = {
            "match_id": match_id,
            "condition": condition_tensor,
            "other": other_tensor,
            "target": target_tensor,
            "condition_columns": [
                f"{base}_{f}" for base in player_bases for f in ["x", "y", "vx", "vy", "dist", "position", "starter", "possession_duration", "neighbor_count"]
            ] + ball_feats,
            "other_columns": other_columns,
            "target_columns": target_columns,
            "condition_frames": list(condition_seq.index),
            "target_frames": list(target_seq.index),
            "pitch_scale": (x_scale, y_scale)
        }
        
        sample["graph"] = build_graph_sequence_from_condition({
            "condition": sample["condition"],
            "condition_columns": sample["condition_columns"],
            "pitch_scale": sample["pitch_scale"]
        })
        
        return sample


if __name__ == "__main__":
    raw_data_path = "idsse-data" # Raw Data Downloaded Path
    data_save_path = "match_data" # Saving path for preprocessed data

    organize_and_process(raw_data_path, data_save_path)

    dataset = MultiMatchSoccerDataset(data_root=data_save_path)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    sample = dataset[0]
    
    print(len(dataset), "samples loaded.")
    sample = dataset[0]
    print("Match id:", sample["match_id"])
    print("Condition columns:", sample["condition_columns"])
    print("Condition shape:", sample["condition"].shape)
    print("Other columns:", sample["other_columns"])
    print("Other shape:", sample["other"].shape)
    print("Target columns:", sample["target_columns"])
    print("Target shape:", sample["target"].shape)
    print("Condition frames:", sample["condition_frames"])
    print("Using frames:", sample["target_frames"])
    
    print("Condition:", sample["condition"])
    print("Target:", sample["target"])

