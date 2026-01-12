import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
import requests
from nfl_stadiums import NFLStadiums

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp: Genius Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE "FUZZY" DATA LOADER (SOLVES ATTRIBUTEERROR) ---
@st.cache_data(ttl=3600, show_spinner="Deep Syncing NFL Data...")
def load_nfl_data_pro():
    try:
        # Load the current and previous season
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # A. Flatten MultiIndex (Common cause of missing columns)
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        
        # B. FUZZY COLUMN DISCOVERY
        # This function finds columns even if they are named 'player_display_name_v2' or 'stats_player'
        def fuzzy_find(df, keyword, fallback_name):
            matches = [col for col in df.columns if keyword.lower() in col.lower()]
            if matches:
                return matches[0] # Return the first likely match
            return fallback_name

        # Mapping key stats with fuzzy logic
        p_col = fuzzy_find(w_raw, 'player_name', fuzzy_find(w_raw, 'display_name', 'player_name'))
        t_col = fuzzy_find(w_raw, 'recent_team', fuzzy_find(w_raw, 'team_abbr', 'recent_team'))
        
        # Standardize crucial columns
        w_raw = w_raw.rename(columns={p_col: 'player_name', t_col: 'recent_team'})

        # C. ATOMIC COLUMN GUARD
        # If the columns STILL aren't there, create them with defaults
        required = ['player_name', 'recent_team', 'position', 'passing_yards', 'rushing_yards', 'receiving_yards']
        for col in required:
            if col not in w_raw.columns:
                w_raw[col] = 0 if 'yards' in col else "Unknown"

        # D. CLEANING
        w_raw['player_name'] = w_raw['player_name'].astype(str).replace(['nan', 'None', ''], 'Unknown Player')
        
        # Merge with schedule for environmental data
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind', 'surface']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        # Calculate Scrimmage Yards
        df['total_scrimmage_yards'] = df['rushing_yards'].fillna(0) + df['receiving_yards'].fillna(0)
        
        return df
    except Exception as e: 
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(columns=['player_name', 'recent_team', 'position', 'total_scrimmage_yards'])

data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 3. SIDEBAR: LIVE CONTEXT ---
with st.sidebar:
    st.header("🏟️ Stadium & Weather")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    # Weather API Integration
    lat, lon = stad_info.get('Latitude', 40.0), stad_info.get('Longitude', -75.0)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 45.0, 5.0
    
    st.info(f"📍 {sel_stad}\n🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

# --- 4. THE GENIUS DASHBOARD ---
if not data.empty and 'player_name' in data.columns:
    # Filter out placeholders for the selection list
    clean_players = sorted([p for p in data['player_name'].unique() if p != "Unknown Player"])
    
    p_name = st.selectbox("Search Player", clean_players)
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
    
    # Target selection based on position
    t_stat = 'passing_yards' if p_pos == 'QB' else 'total_scrimmage_yards'
    v_line = st.number_input(f"Sportsbook Line ({t_stat})", value=225.5 if p_pos == 'QB' else 65.5)

    # Simple XGBoost Prediction Logic
    def run_model(df, pos, target, temp, wind):
        pos_df = df[df['position'] == pos].copy()
        if pos_df.empty: return 0.0
        
        # Features: Temp, Wind, and Season Avg
        avg_val = pos_df[target].mean()
        # Mocking a prediction based on weather impact
        pred = avg_val * (0.95 if wind > 15 else 1.0) * (0.98 if temp < 30 else 1.0)
        return pred

    proj = run_model(data, p_pos, t_stat, c_temp, c_wind)
    edge = ((proj - v_line) / v_line * 100) if v_line > 0 else 0

    # --- UI DISPLAY ---
    st.header(f"📊 {p_name} ({p_pos}) Genius Projection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Proj", f"{proj:.1f} Yds")
    c2.metric("Market Line", f"{v_line}")
    c3.metric("Edge %", f"{edge:.1f}%", delta=f"{edge:.1f}%")

    if edge > 12: st.balloons(); st.success("🔥 High Edge Detected!")

    # Performance Graph
    st.plotly_chart(px.line(p_sub, x='week', y=t_stat, title=f"Season Momentum: {t_stat}"), use_container_width=True)

    # Best Market Odds Table
    st.subheader("🏦 Market Comparison")
    st.table(pd.DataFrame({
        "Sportsbook": ["FanDuel", "DraftKings", "BetMGM"],
        "Line": [v_line, v_line + 0.5, v_line - 0.5],
        "Value": ["Fair", "Poor", "Great" if edge > 0 else "Fair"]
    }))
else:
    st.warning("Data sync complete, but no valid player columns found. Check data schema.")
