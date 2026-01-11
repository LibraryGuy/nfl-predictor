import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE (EXPANDED MAPPING) ---
@st.cache_data(ttl=3600, show_spinner="Syncing NFL Prop Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 

        # --- FIX A: FLATTEN MULTI-INDEX ---
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        w_raw.columns = [str(c).strip() for c in w_raw.columns]

        # --- FIX B: THE PROP MAPPER (Resolves 'rush_tds' Error) ---
        prop_map = {
            'player_name': ['player_display_name', 'player_name', 'player'],
            'recent_team': ['team_abbr', 'recent_team', 'team'],
            'passing_yards': ['pass_yards', 'passing_yards'],
            'rushing_yards': ['rush_yards', 'rushing_yards'],
            'receiving_yards': ['rec_yards', 'receiving_yards'],
            'pass_tds': ['passing_tds', 'pass_tds'],
            'rush_tds': ['rushing_tds', 'rush_tds'],
            'receiving_tds': ['rec_tds', 'receiving_tds']
        }

        for target, options in prop_map.items():
            found = next((c for c in options if c in w_raw.columns), None)
            if found: w_raw = w_raw.rename(columns={found: target})

        # --- FIX C: CLEAN & COMPUTE ---
        # Ensure name is a string to prevent .str crashes
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Fill missing metrics with 0
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'pass_tds', 'rush_tds', 'receiving_tds']
        for m in metrics:
            if m in w_raw.columns:
                w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)

        w_raw['total_scrimmage_yards'] = w_raw['rushing_yards'] + w_raw['receiving_yards']
        w_raw['total_tds'] = w_raw['rush_tds'] + w_raw['receiving_tds']

        # Weather & EPA Merge
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                         left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. UI & PREDICTION (RETAINED) ---
# [The rest of your Auth, Sidebar, and Prediction Engine code continues here...]
# Ensure your target_tds line uses the mapped names:
# target_tds = 'pass_tds' if player_pos == 'QB' else 'total_tds'

if not data.empty:
    # Sidebar logic
    with st.sidebar:
        st.header("🏟️ Game Context")
        curr_wind = st.slider("Wind", 0, 40, 5)
        curr_temp = st.slider("Temp", 0, 100, 45)
        is_grass = 1 if st.radio("Field", ["Grass", "Turf"]) == "Grass" else 0

    # Player Selection
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    player_subset = data[data['player_name'] == selected_player]
    
    if not player_subset.empty:
        player_pos = player_subset['position'].iloc[-1]
        target_yds = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
        target_tds = 'pass_tds' if player_pos == 'QB' else 'total_tds'
        
        # Prediction Logic
        # (Using the same get_safe_prediction function from your previous version)
        
        st.header(f"📊 {selected_player} Projections")
        col1, col2 = st.columns(2)
        # Display Projected Yards and TDs using these mapped columns
        # ...
