import os
import random
import torch
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml

# This code is from "https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking"
# "https://github.com/spoho-datascience/idsse-data"

# Setting seed with reproducibility
def set_seed(seed=42):
    random.seed(seed)                 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load team sheet information from matchinformation XML files
def load_team_sheets(path):
    info_files = [x for x in os.listdir(path) if "matchinformation" in x]
    team_sheets_all = pd.DataFrame()
    for file in info_files:
        team_sheets = read_teamsheets_from_mat_info_xml(os.path.join(path, file))
        team_sheets_combined = pd.concat([team_sheets["Home"].teamsheet, team_sheets["Away"].teamsheet])
        team_sheets_all = pd.concat([team_sheets_all, team_sheets_combined])
    return team_sheets_all

# Load all event data (passes, shots, etc.) from raw event XML files
def load_event_data(path):
    info_files = [x for x in os.listdir(path) if "matchinformation" in x]
    event_files = [x for x in os.listdir(path) if "events_raw" in x]
    all_events = pd.DataFrame()
    for events_file, info_file in zip(event_files, info_files):
        events, _, _ = read_event_data_xml(os.path.join(path, events_file), os.path.join(path, info_file))
        events_fullmatch = pd.DataFrame()
        for half in events:
            for team in events[half]:
                events_fullmatch = pd.concat([events_fullmatch, events[half][team].events])
        all_events = pd.concat([all_events, events_fullmatch])
    return all_events

# Count total number of position frames (Home team only) from raw position XML files
def load_position_data(path):
    info_files = [x for x in os.listdir(path) if "matchinformation" in x]
    position_files = [x for x in os.listdir(path) if "positions_raw" in x]
    n_frames = 0
    for position_file, info_file in zip(position_files, info_files):
        positions, _, _, _, _ = read_position_data_xml(os.path.join(path, position_file), os.path.join(path, info_file))
        n_frames += len(positions["firstHalf"]["Home"]) + len(positions["secondHalf"]["Home"])
    return n_frames

# Merge tracking data of Home and Away teams into a single DataFrame (excluding ball columns)
def merge_tracking_data(home,away):
    return home.drop(columns=['ball_x', 'ball_y']).merge( away, left_index=True, right_index=True )

# Flip second-half coordinates so both teams always attack in the same direction
def to_single_playing_direction(home,away,events):
    for team in [home,away,events]:
        # second_half_idx = team.Period.idxmax(2)
        second_half_idx = team[team['Period'] == 2].index.min()
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home,away,events

# Compute smoothed velocity and speed for each player and the ball
def calc_velocites(df, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, player_maxspeed=100, ball_maxspeed=1000):
    # remove any velocity data already in the dataframe
    columns = [c for c in df.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    
    df = df.drop(columns=columns)
    
    # Get the player ids
    player_ids = np.unique([c[:-2] for c in df.columns if c.startswith(('Home_', 'Away_')) and c[-2:] in ['_x', '_y']])

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = df['Time [s]'].diff()
    
    # index of first frame in second half
    # second_half_idx = df.Period.idxmax(2)
    second_half_idx = df[df['Period'] == 2].index.min()
    
    # estimate velocities for players in df
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = df[player+"_x"].diff() / dt
        vy = df[player+"_y"].diff() / dt

        if player_maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>player_maxspeed ] = np.nan
            vy[ raw_speed>player_maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        df[player + "_vx"] = vx
        df[player + "_vy"] = vy
        df[player + "_speed"] = np.sqrt( vx**2 + vy**2 )
        
         # 공 위치 속도 계산
    if 'ball_x' in df.columns and 'ball_y' in df.columns:
        bvx = df['ball_x'].diff() / dt
        bvy = df['ball_y'].diff() / dt
        bspeed = np.sqrt(bvx**2 + bvy**2)

        bvx[bspeed > ball_maxspeed] = np.nan
        bvy[bspeed > ball_maxspeed] = np.nan

        if smoothing:
            if filter_ == 'Savitzky-Golay':
                bvx.loc[:second_half_idx] = signal.savgol_filter(bvx.loc[:second_half_idx], window_length=window, polyorder=polyorder)
                bvy.loc[:second_half_idx] = signal.savgol_filter(bvy.loc[:second_half_idx], window_length=window, polyorder=polyorder)
                bvx.loc[second_half_idx:] = signal.savgol_filter(bvx.loc[second_half_idx:], window_length=window, polyorder=polyorder)
                bvy.loc[second_half_idx:] = signal.savgol_filter(bvy.loc[second_half_idx:], window_length=window, polyorder=polyorder)
            elif filter_ == 'moving average':
                ma = np.ones(window) / window
                bvx.loc[:second_half_idx] = np.convolve(bvx.loc[:second_half_idx], ma, mode='same')
                bvy.loc[:second_half_idx] = np.convolve(bvy.loc[:second_half_idx], ma, mode='same')
                bvx.loc[second_half_idx:] = np.convolve(bvx.loc[second_half_idx:], ma, mode='same')
                bvy.loc[second_half_idx:] = np.convolve(bvy.loc[second_half_idx:], ma, mode='same')

        df['ball_vx'] = bvx
        df['ball_vy'] = bvy
        df['ball_speed'] = np.sqrt(bvx**2 + bvy**2)

    return df

# Detect sudden jumps (large velocity spikes) in position sequence
def detect_jumps(xy_seq, maxspeed=12.0, fps=25.0):
    dt = 1.0 / fps
    # [T, 2] → [T-1]
    velocities = np.linalg.norm(np.diff(xy_seq, axis=0), axis=1) / dt
    jump_indices = np.where(velocities > maxspeed)[0] + 1
    return jump_indices

# Correct jump frames (and adjacent) using cubic spline interpolation
def correct_with_cubic_spline_adjacent(xy_seq, jump_indices):
    T = len(xy_seq)
    valid_mask = np.ones(T, dtype=bool)
    jump_and_adjacent = set()
    for t in jump_indices:
        for dt in [-2, -1, 0, 1, 2]:
            if 0 <= t + dt < T:
                jump_and_adjacent.add(t + dt)
    valid_mask[list(jump_and_adjacent)] = False

    if valid_mask.sum() < 4:
        return xy_seq 

    valid_t = np.where(valid_mask)[0]
    x_spline = CubicSpline(valid_t, xy_seq[valid_mask][:, 0])
    y_spline = CubicSpline(valid_t, xy_seq[valid_mask][:, 1])

    corrected = xy_seq.copy()
    for t in jump_and_adjacent:
        corrected[t, 0] = x_spline(t)
        corrected[t, 1] = y_spline(t)

    return corrected


# Apply jump correction to all players in the tracking DataFrame
def correct_all_player_jumps_adjacent(df: pd.DataFrame, framerate=25.0, maxspeed=12.0):
    corrected_df = df.copy()

    player_ids = sorted(set(
        col.rsplit("_", 1)[0] 
        for col in df.columns 
        if ("_x" in col or "_y" in col) and "ball" not in col
    ))

    for pid in player_ids:
        col_x = f"{pid}_x"
        col_y = f"{pid}_y"

        if col_x not in df.columns or col_y not in df.columns:
            continue

        xy_seq = df[[col_x, col_y]].values  # [T, 2]

        # Skip players with NaN in position
        if np.isnan(xy_seq).any():
            continue

        jump_indices = detect_jumps(xy_seq, maxspeed=maxspeed, fps=framerate)
        if len(jump_indices) == 0:
            continue

        corrected = correct_with_cubic_spline_adjacent(xy_seq, jump_indices)
        corrected_df[col_x] = corrected[:, 0]
        corrected_df[col_y] = corrected[:, 1]

    return corrected_df

