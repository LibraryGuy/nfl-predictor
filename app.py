import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from nfl_stadiums import NFLStadiums

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp: Genius Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADER (REPAIRED FOR 2026 SCHEMA) ---
@st.cache_data(ttl=3600, show_spinner="Syncing 2025/2026 Data...")
def load_nfl_data_pro():
    try:
        # Load Stats & Schedules
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # A. Flatten MultiIndex
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        
        # B. THE REPAIR: Broad-spectrum Column Mapping
        # We look for ANY column that likely represents the player's team or name
        rename_map = {}
        
        # Find Player Name Column
        for col in w_raw.columns:
            if col.lower() in ['player_name', 'player_display_name', 'player']:
                rename_map[col] = 'player_name'
                break
        
        # Find Team Column (The 'recent_team' Fix)
        for col in w_raw.columns:
            if col.lower() in ['recent_team', 'team', 'team_abbr', 'posteam']:
                rename_map[col] = 'recent_team'
                break

        # Find Opponent Column
        for col in w_raw.columns:
            if col.lower() in ['opponent_team', 'opponent', 'defteam']:
                rename_map[col] = 'opponent'
                break

        w_raw = w_raw.rename(columns=rename_map)

        # C. DvP Engine (Defense vs Position)
        dvp_df = w_raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'defending_team'})

        # D. Join with Weather/Stadium Logic
        # We merge schedules on the player's team to find their game environment
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind']], 
                         left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        df['total_scrimmage_yards'] = df['rushing_yards'].fillna(0) + df['receiving_yards'].fillna(0)
        
        return df.fillna(0), dvp_df
    except Exception as e: 
        st.error(f"Sync Failure: {e}")
        # Return a structure that won't crash the sidebar
        return pd.DataFrame(columns=['player_name', 'recent_team', 'position']), pd.DataFrame()

data, dvp_data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 3. SIDEBAR: LIVE CONTEXT ---
with st.sidebar:
    st.header("🏟️ Stadium & Weather")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Game Venue", all_stads)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    lat, lon = stad_info.get('Latitude', 40.0), stad_info.get('Longitude', -75.0)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 50.0, 5.0
    
    st.info(f"🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

# --- 4. THE GENIUS DASHBOARD ---
if not data.empty and 'player_name' in data.columns:
    players = sorted([p for p in data['player_name'].unique() if p != "Unknown"])
    p_name = st.selectbox("Search Player", players)
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
    
    # Defense Select
    opp_team = st.selectbox("Opposing Defense", sorted(data['recent_team'].unique()))

    # Calculate Modifiers
    def get_defense_mod(dvp, team, pos):
        match = dvp[(dvp['defending_team'] == team) & (dvp['position'] == pos)]
        if match.empty: return 1.0
        league_avg = dvp[dvp['position'] == pos]['receiving_yards'].mean()
        return 1.12 if match['receiving_yards'].iloc[0] > (league_avg * 1.15) else 0.92

    def_mod = get_defense_mod(dvp_data, opp_team, p_pos)
    
    t_stat = 'passing_yards' if p_pos == 'QB' else 'total_scrimmage_yards'
    v_line = st.number_input(f"Sportsbook Line ({t_stat})", value=65.5)
    
    proj = p_sub[t_stat].mean() * (0.95 if c_wind > 15 else 1.0) * def_mod
    edge = ((proj - v_line) / v_line * 100) if v_line > 0 else 0

    st.header(f"📊 {p_name} Projection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Proj", f"{proj:.1f} Yds")
    c2.metric("Matchup Edge", f"{def_mod:.2f}x")
    c3.metric("Edge %", f"{edge:.1f}%", delta=f"{edge:.1f}%")

    st.plotly_chart(px.line(p_sub, x='week', y=t_stat, title=f"{p_name} Trend"), use_container_width=True)
else:
    st.warning("Data schema mismatch. Try refreshing the app or clearing the cache in 'Manage App'.")
