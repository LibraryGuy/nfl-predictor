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

# --- 2. THE DATA LOADER (WITH DEFENSE ENGINE) ---
@st.cache_data(ttl=3600, show_spinner="Deep Syncing NFL Data...")
def load_nfl_data_pro():
    try:
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # A. Flatten MultiIndex
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        
        # B. Fuzzy Column Mapping
        def fuzzy_find(df, keyword, fallback):
            matches = [col for col in df.columns if keyword.lower() in col.lower()]
            return matches[0] if matches else fallback

        p_col = fuzzy_find(w_raw, 'player_name', 'player_name')
        t_col = fuzzy_find(w_raw, 'recent_team', 'recent_team')
        o_col = fuzzy_find(w_raw, 'opponent_team', 'opponent') # Crucial for defense

        w_raw = w_raw.rename(columns={p_col: 'player_name', t_col: 'recent_team', o_col: 'opponent'})

        # C. Defense vs Position (DvP) Engine
        # We calculate how many yards each defense gives up to each position
        dvp_df = w_raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'defending_team'})

        # D. Environmental Merge
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind']], 
                         left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        df['total_scrimmage_yards'] = df['rushing_yards'].fillna(0) + df['receiving_yards'].fillna(0)
        
        return df.fillna(0), dvp_df
    except Exception as e: 
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, dvp_data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 3. SIDEBAR: LIVE CONTEXT ---
with st.sidebar:
    st.header("🏟️ Stadium & Weather")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    # Weather
    lat, lon = stad_info.get('Latitude', 40.0), stad_info.get('Longitude', -75.0)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 45.0, 5.0
    
    st.info(f"📍 {sel_stad}\n🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

# --- 4. THE GENIUS DASHBOARD ---
if not data.empty:
    players = sorted([p for p in data['player_name'].unique() if p != "Unknown Player"])
    p_name = st.selectbox("Search Player", players)
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
    
    # NEW: Select the Opposing Defense
    all_teams = sorted(data['recent_team'].unique())
    opp_team = st.selectbox("Opposing Defense", all_teams)

    # NEW: Defense Modifier Logic
    def get_defense_mod(dvp, team, pos):
        match = dvp[(dvp['defending_team'] == team) & (dvp['position'] == pos)]
        if match.empty: return 1.0
        
        # Compare vs league average for that position
        league_avg = dvp[dvp['position'] == pos]['receiving_yards'].mean()
        team_allowed = match['receiving_yards'].iloc[0]
        
        # If defense allows 20% more than average, mod is 1.10
        return 1.10 if team_allowed > (league_avg * 1.2) else 0.95

    def_mod = get_defense_mod(dvp_data, opp_team, p_pos)

    # Model Calculation
    t_stat = 'passing_yards' if p_pos == 'QB' else 'total_scrimmage_yards'
    v_line = st.number_input(f"Sportsbook Line ({t_stat})", value=65.5)
    
    # Enhanced Projection (Base Avg * Weather Mod * Defense Mod)
    base_avg = p_sub[t_stat].mean()
    weather_mod = (0.95 if c_wind > 15 else 1.0)
    proj = base_avg * weather_mod * def_mod
    edge = ((proj - v_line) / v_line * 100) if v_line > 0 else 0

    # --- UI DISPLAY ---
    st.header(f"📊 {p_name} vs {opp_team}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Proj", f"{proj:.1f} Yds")
    c2.metric("Matchup Mod", f"{def_mod:.2f}x", help="Based on defense vs position history")
    c3.metric("Edge %", f"{edge:.1f}%", delta=f"{edge:.1f}%")

    if edge > 15: st.balloons(); st.success("🎯 Massive Value Matchup!")

    # Performance Graph
    st.plotly_chart(px.line(p_sub, x='week', y=t_stat, title=f"{p_name} Performance Trend"), use_container_width=True)
