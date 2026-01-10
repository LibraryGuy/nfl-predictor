import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp Pro Predictor", layout="wide")

@st.cache_data(show_spinner="Loading NFL Database...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        
        # Standardize Team Columns
        if 'recent_team' not in weekly.columns:
            team_col = 'team' if 'team' in weekly.columns else 'team_abbr'
            weekly = weekly.rename(columns={team_col: 'recent_team'})
        
        # Basic Cleaning
        weekly = weekly.dropna(subset=['player_name', 'position'])
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
        for m in metrics: weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
        
        # Calculate Rolling Averages
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())
        
        # Merge Environment Data
        sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
        df = weekly.merge(sched[sched_cols], left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        df['wind'] = df['wind'].fillna(0)
        df['temp'] = df['temp'].fillna(70)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception:
        return pd.DataFrame()

# --- INITIALIZE DATA ---
data = load_nfl_data_pro()

if data.empty:
    st.error("Could not load NFL data. Please refresh.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Game Conditions")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

# UI Selection
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list)

# Get Player Metadata
player_row = data[data['player_name'] == selected_player].iloc[-1]
player_pos = player_row['position']
vegas_line = st.sidebar.number_input("Vegas Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- FIXED PREDICTION ENGINE ---
def get_stable_prediction(df, player_name, pos, target_stat, temp, wind, is_grass):
    # 1. Train on ALL players of the same position to get "Global" patterns
    # This prevents Caleb Williams' turf data from being too sensitive
    pos_data = df[df['position'] == pos].copy()
    
    # 2. Features: Include the specific player's rolling average as a feature
    # This tells the model "This is a high-volume player" regardless of the name
    feature_cols = ['temp', 'wind', 'is_grass']
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    X = pos_data[feature_cols + [roll_col]].fillna(0)
    y = pos_data[target_stat]
    
    # 3. Robust Model (Regularized to prevent wild swings)
    model = XGBRegressor(n_estimators=50, max_depth=3, reg_lambda=10).fit(X, y)
    
    # 4. Predict for our specific player
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    input_data = pd.DataFrame([[temp, wind, is_grass, p_latest[roll_col]]], 
                               columns=feature_cols + [roll_col])
    
    pred = model.predict(input_data)[0]
    
    # 5. Safety Floor: Prediction can't be less than 60% of their 3-game average
    floor = p_latest[roll_col] * 0.6
    return max(pred, floor)

# --- EXECUTION & DISPLAY ---
st.title(f"🏈 {selected_player} Analysis ({player_pos})")
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_stable_prediction(data, selected_player, player_pos, target, curr_temp, curr_wind, is_grass_val)

col1, col2 = st.columns(2)
col1.metric("Projected Yards", f"{proj:.1f}")
col2.metric("Edge vs. Vegas", f"{proj - vegas_line:.1f}")

st.plotly_chart(px.line(data[data['player_name'] == selected_player], x='week', y=target, title="Season Trend"))
