import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.interpolate import CubicSpline
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml

# This code is from "https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking"
# "https://github.com/spoho-datascience/idsse-data"

# Setting seed with reproducibility
def set_evertyhing(seed):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generator(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
def calc_velocites(df, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, player_maxspeed=12, ball_maxspeed=1000):
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


def per_player_frechet_loss(pred, target):
    # pred, target (B, T, N=11, 2)
    dists = torch.norm(pred - target, dim=-1) # (B, T, N)
    max_dists = dists.max(dim=1).values # (B, N)
    return max_dists.mean(dim=1).mean()


def plot_pitch( field_dimen = (106.0,68.0), field_color ='green', linewidth=2, markersize=20):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1] 
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=linewidth)
    ax.plot(-x,y,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax


## Vizualization
def plot_trajectories_on_pitch(others, target, pred, other_columns = None, target_columns = None, player_idx=None, annotate=False, save_path=None):
    if torch.is_tensor(others):
        others = others.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    
    fig, ax = plot_pitch(field_dimen=(105.0, 68.0), field_color='green')

    # 1) attackers
    for m in range(11):
        x, y = others[:, m, 0], others[:, m, 1]
        ax.plot(x, y, color='red', linestyle='-', linewidth=2.0, alpha = 0.7, label='Attackers' if m == 0 else None)
        ax.scatter(x[-1], y[-1], color='red', s=50, marker='o', alpha=0.7)
        if annotate and other_columns is not None:
            col_x = other_columns[2 * m]  # e.g. 'Home_2_x'
            jersey = col_x.split('_')[1]
            x0, y0 = others[0, m, 0], others[0, m, 1]
            ax.text(x0 + 0.5, y0 + 0.5, jersey, color='red', fontsize=10)
    # ball
    ball_x, ball_y = others[:, 11, 0], others[:, 11, 1]
    ax.plot(ball_x, ball_y, color='black', linestyle='-', linewidth=2.0, alpha = 1.0, label='Ball')
    ax.scatter(ball_x[-1], ball_y[-1], color='black', s=30, marker='o', alpha=1.0)

    # 2) defenders GT / Pred
    i = player_idx
    
    x, y = target[:, i, 0], target[:, i, 1]
    ax.plot(x, y, color='blue', linestyle='-', linewidth=2.0, alpha=0.7, label='Target' if i == 0 else None)
    ax.scatter(x[-1], y[-1], color='blue', s=50, marker='o', alpha=0.7)
    
    if annotate and target_columns is not None:
        col_x = target_columns[2 * i]  # e.g. 'Home_2_x'
        jersey = col_x.split('_')[1]
        x0, y0 = target[0, i, 0], target[0, i, 1]
        ax.text(x0 + 0.5, y0 + 0.5, jersey, color='blue', fontsize=10)
    
    x, y = pred[:, i, 0], pred[:, i, 1]
    ax.plot(x, y, color='blue', linestyle='--', linewidth=2.0, alpha=0.5, label='Predicted' if i == 0 else None)
    ax.scatter(x[-1], y[-1], color='blue', s=50, marker='x', alpha=0.5)
    
    if annotate and target_columns is not None:
        col_x = target_columns[2 * i]  # e.g. 'Home_2_x'
        jersey = col_x.split('_')[1]
        x0, y0 = pred[0, i, 0], pred[0, i, 1]
        ax.text(x0 + 0.5, y0 + 0.5, f"{jersey}(pred)", color='blue', fontsize=10)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=True)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


