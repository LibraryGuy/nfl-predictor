import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp: Prediction Engine", layout="wide")

# 2. THE DATA LOADER (With explicit safety return)
@st.cache_data(show_spinner="Connecting to NFL Data...")
def load_nfl_data_pro():
    try:
        # Fetching 2024 and 2025 seasons
        weekly = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # ID Alignment: Mapping PBP players to standard IDs
        pbp['player_id'] = pbp['receiver_player_id'].fillna(
            pbp.get('rusher_player_id', np.nan)).fillna(pbp.get('passer_player_id', np.nan))
        
        # Calculate Metrics
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        rz_touches = pbp[pbp['yardline_100'] <= 20].groupby(['season', 'week', 'player_id']).size().reset_index(name='rz_touches')
        
        # Merge datasets
        df = weekly.merge(rz_touches, on=['season', 'week', 'player_id'], how='left').fillna(0)
        
        # Standardize and add scrimmage yards
        df['total_scrimmage_yards'] = df['rushing_yards'] + df['receiving_yards']
        df = df.sort_values(['player_name', 'season', 'week'])
        df['rz_touches_roll3'] = df.groupby('player_name')['rz_touches'].transform(lambda x: x.rolling(3, 1).mean())
        
        # Environmental Merge
        df = df.merge(sched[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')

        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df # Success return
        
    except Exception as e:
        # If anything fails, return an EMPTY dataframe, not None
        return pd.DataFrame()

# 3. GET DATA
data = load_nfl_data_pro()

# 4. SAFETY CHECK (Crucial fix for Line 82)
# If 'data' is empty or None, this block prevents the code from reaching the error line
if data is None or data.empty:
    st.title("🏈 NFL Sharp")
    st.error("Unable to load NFL data. This is often a temporary connection issue with the nflverse server.")
    if st.button("Retry Connection"):
        st.cache_data.clear()
        st.rerun()
    st.stop() # FORCES the script to stop here, so Line 82 is never reached

# 5. UI CONTROLS (Only runs if data exists)
st.title("🏈 NFL Sharp: Ultimate Prediction Engine")

player_list = sorted(data['player_name'].dropna().unique())
selected_player = st.selectbox("Select Player", player_list)

# The rest of your app logic...
player_subset = data[data['player_name'] == selected_player]
st.write(f"Analyzing {selected_player}...")
st.line_chart(player_subset[['week', 'total_scrimmage_yards']].set_index('week'))
