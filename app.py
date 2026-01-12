import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
import requests
from nfl_stadiums import NFLStadiums

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp: Genius Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. AUTH ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. THE "ATOMIC" DATA LOADER ---
@st.cache_data(ttl=3600, show_spinner="Deep Syncing NFL Data...")
def load_nfl_data_pro():
    try:
        # Loading 2024 and 2025 data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # A. COLUMN MAPPING (2025 Schema)
        # We prioritize 'player_display_name' which is the new standard
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player', 'display_name'],
            'recent_team': ['recent_team', 'team', 'team_abbr', 'pos_team'],
            'pass_yds': ['passing_yards', 'pass_yards'],
            'rush_yds': ['rushing_yards', 'rush_yards'],
            'rec_yds': ['receiving_yards', 'rec_yards']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in w_raw.columns), None)
            if found:
                w_raw = w_raw.rename(columns={found: target})
            elif target not in w_raw.columns:
                # B. ATOMIC COLUMN GUARD: Create missing columns to prevent AttributeError
                w_raw[target] = 0 if 'yds' in target else "Unknown"

        # C. CLEANING & MERGING
        # Ensure player_name is a string and drop any complete nulls
        w_raw['player_name'] = w_raw['player_name'].fillna("Unknown Player").astype(str)
        
        # Merge with Schedule for Weather/Venue context
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind', 'surface']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        # Calculate Total Scrimmage Yards safely
        df['total_scrimmage_yards'] = pd.to_numeric(df.get('rush_yds', 0), errors='coerce').fillna(0) + \
                                      pd.to_numeric(df.get('rec_yds', 0), errors='coerce').fillna(0)
        
        return df.fillna(0)
    except Exception as e: 
        st.error(f"Critical Data Sync Error: {e}")
        return pd.DataFrame(columns=['player_name', 'position', 'recent_team', 'total_scrimmage_yards'])

data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 4. SIDEBAR CONTEXT ---
with st.sidebar:
    st.header("🏟️ Stadium & Environment")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    # Lat/Lon for Open-Meteo API
    lat, lon = stad_info.get('Latitude', 44.5), stad_info.get('Longitude', -88.0)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 45.0, 5.0
    
    st.metric("Live Weather", f"{c_temp:.0f}°F", delta=f"{c_wind:.1f} MPH Wind")

# --- 5. THE GENIUS DASHBOARD ---
if not data.empty:
    # THE ERROR-FIXED SELECTBOX
    # We filter out "Unknown Player" and ensure only unique strings are sorted
    player_list = sorted([p for p in data['player_name'].unique() if p != "Unknown Player"])
    
    if player_list:
        p_name = st.selectbox("Search Player", player_list)
        p_sub = data[data['player_name'] == p_name]
        p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
        
        # Projection Target Selection
        t_yds = 'pass_yds' if p_pos == 'QB' else 'total_scrimmage_yards'
        v_line = st.number_input(f"Sportsbook Line ({t_yds})", value=225.5 if p_pos == 'QB' else 65.5)

        # Basic Model Logic
        def get_projection(df, pos, target, temp, wind):
            subset = df[df['position'] == pos]
            if subset.empty: return 0.0
            avg_perf = subset[target].mean()
            # Weather penalty logic
            weather_mod = 0.90 if wind > 15 or temp < 32 else 1.0
            return avg_perf * weather_mod

        p_yds = get_projection(data, p_pos, t_yds, c_temp, c_wind)
        edge = ((p_yds - v_line) / v_line * 100) if v_line > 0 else 0

        # --- UI DISPLAY ---
        st.header(f"📊 {p_name} ({p_pos}) Projection")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Proj", f"{p_yds:.1f} Yds")
        c2.metric("Market Line", f"{v_line}")
        c3.metric("Edge %", f"{edge:.1f}%", delta=f"{edge:.1f}%")

        if edge > 15: st.balloons(); st.success("🔥 High Value Betting Edge!")
        
        st.plotly_chart(px.bar(p_sub, x='week', y=t_yds, title="Weekly Performance History"), use_container_width=True)
    else:
        st.warning("No player data found for the selected seasons. Check your data source connection.")
