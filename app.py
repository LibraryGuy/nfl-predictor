import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE CONFIG
st.set_page_config(page_title="NFL Sharp: Ultimate Prediction Engine", layout="wide")

# 2. CACHED DATA LOADER
@st.cache_data(show_spinner="Downloading NFL Stats (2024-2025)...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        
        # Fetching data using nflreadpy
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        
        # FIX: Align Player IDs between Stats and Play-by-Play
        pbp['player_id'] = pbp['receiver_player_id'].fillna(
            pbp.get('rusher_player_id', np.nan)).fillna(pbp.get('passer_player_id', np.nan))
        
        # FEATURE: Defensive Difficulty (EPA)
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # FEATURE: Red Zone High-Value Touches
        rz_touches = pbp[pbp['yardline_100'] <= 20].groupby(['season', 'week', 'player_id']).size().reset_index(name='rz_touches')
        
        # MERGE: Combine everything into one master dataframe
        df = weekly.merge(rz_touches, on=['season', 'week', 'player_id'], how='left').fillna(0)
        
        # CLEANUP: Standardize column names
        team_col = 'recent_team' if 'recent_team' in df.columns else 'team'
        df = df.rename(columns={team_col: 'recent_team'})
        
        # CALC: Scrimmage Yards & Rolling Averages
        df['total_scrimmage_yards'] = df['rushing_yards'] + df['receiving_yards']
        df = df.sort_values(['player_name', 'season', 'week'])
        df['rz_touches_roll3'] = df.groupby('player_name')['rz_touches'].transform(lambda x: x.rolling(3, 1).mean())
        
        # ENV: Merge Temperature/Wind/Surface
        df = df.merge(sched[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')

        # Final Fill
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception as e:
        st.sidebar.error(f"Load Error: {e}")
        return pd.DataFrame()

# 3. INITIALIZE DATA
data = load_nfl_data_pro()

# 4. MAIN APP LOGIC (The Safety Guard)
# This block ensures line 84 (player_list) only runs if data exists
if isinstance(data, pd.DataFrame) and not data.empty:
    
    st.title("🏈 NFL Sharp: Ultimate Prediction Engine")
    
    # --- SIDEBAR ---
    st.sidebar.header("Game Settings")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
    curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
    is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

    # --- PLAYER SELECTION (Line 84 Fix) ---
    player_list = sorted(data['player_name'].dropna().unique())
    selected_player = st.selectbox("Select Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    
    # --- PREDICTION LOGIC ---
    def get_prediction(target):
        features = ['temp', 'wind', 'is_grass', 'rz_touches_roll3', 'def_epa_allowed']
        X = player_subset[features].fillna(0)
        y = player_subset[target]
        
        if len(y) < 2: return 0.0
        
        model = XGBRegressor(n_estimators=30).fit(X, y)
        latest = X.iloc[[-1]].copy()
        latest['temp'], latest['wind'], latest['is_grass'] = curr_temp, curr_wind, is_grass_val
        return max(0, model.predict(latest)[0])

    # --- UI DISPLAY ---
    target_stat = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
    prediction = get_prediction(target_stat)
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Prediction", f"{prediction:.1f} Yds")
    col2.metric("Positional Rank", player_pos)
    col3.metric("R3 Red Zone Touches", f"{player_subset['rz_touches_roll3'].iloc[-1]:.1f}")
    
    st.plotly_chart(px.line(player_subset, x='week', y=target_stat, title=f"{selected_player} Yardage Trend"))

else:
    # If data failed to load, show this instead of crashing
    st.title("🏈 NFL Sharp")
    st.error("Data could not be loaded from nflverse.")
    st.info("Please check your internet connection or the nflreadpy documentation.")
    if st.button("Retry Load"):
        st.rerun()
