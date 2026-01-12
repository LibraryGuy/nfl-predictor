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

# --- 3. ROBUST DATA LOADER (FIXES 'fillna' & 'recent_team') ---
@st.cache_data(ttl=3600, show_spinner="Syncing Genius Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # nflreadpy returns Polars; to_pandas() is essential
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 

        # A. Flatten Columns & Mapping
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player'],
            'recent_team': ['recent_team', 'team', 'team_abbr', 'pos_team'],
            'pass_yds': ['passing_yards', 'pass_yards', 'p_yds'],
            'rush_yds': ['rushing_yards', 'rush_yards', 'r_yds'],
            'rec_yds': ['receiving_yards', 'rec_yards'],
            'pass_tds': ['passing_tds', 'pass_tds', 'p_tds'],
            'rush_tds': ['rushing_tds', 'rush_tds', 'r_tds'],
            'rec_tds': ['receiving_tds', 'rec_tds']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in w_raw.columns), None)
            if found: w_raw = w_raw.rename(columns={found: target})

        # B. THE 'FILLNA' ERROR FIX (Safe Column Helper)
        def get_stat(df, col_name):
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
            return pd.Series(0, index=df.index)

        # Calculate scrimmage yards safely
        w_raw['total_scrimmage_yards'] = get_stat(w_raw, 'rush_yds') + get_stat(w_raw, 'rec_yds')
        w_raw['total_tds'] = get_stat(w_raw, 'rush_tds') + get_stat(w_raw, 'rec_tds')

        # C. Cleaning Player Names for Selectbox
        w_raw['player_name'] = w_raw['player_name'].astype(str).replace(['None', 'nan', ''], np.nan)
        w_raw = w_raw.dropna(subset=['player_name'])

        # D. Defense Context (EPA)
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # E. Merge Environment (Schedules + Defense)
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e: 
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame()

data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 4. SIDEBAR & WEATHER ---
with st.sidebar:
    st.header("🏟️ Venue & Weather")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    lat, lon = stad_info.get('Latitude', 40.0), stad_info.get('Longitude', -75.0)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 50.0, 5.0
    
    st.info(f"🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

# --- 5. THE GENIUS DASHBOARD ---
if not data.empty:
    # Use clean string list for the selectbox
    clean_players = sorted([str(p) for p in data['player_name'].unique()])
    p_name = st.selectbox("Search Player", clean_players)
    
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
    
    # Selection logic for targets
    t_yds = 'pass_yds' if p_pos == 'QB' else 'total_scrimmage_yards'
    v_line = st.number_input(f"Sportsbook Line ({t_yds})", value=225.5 if p_pos == 'QB' else 65.5)

    def predict_performance(df, pos, target, temp, wind):
        pos_data = df[df['position'] == pos].copy()
        if pos_data.empty: return 0.0
        
        features = ['temp', 'wind', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=40, max_depth=3).fit(pos_data[features], pos_data[target])
        
        pred = model.predict(pd.DataFrame([[temp, wind, 0.0]], columns=features))[0]
        return max(pred, 0) # Ensure no negative yards

    p_yds = predict_performance(data, p_pos, t_yds, c_temp, c_wind)
    edge = ((p_yds - v_line) / v_line * 100) if v_line > 0 else 0

    # --- DISPLAY ---
    st.header(f"📊 {p_name} ({p_pos}) Projection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Proj", f"{p_yds:.1f} Yds")
    c2.metric("Vegas Line", f"{v_line}")
    c3.metric("Edge", f"{edge:.1f}%", delta=f"{edge:.1f}%")

    if edge > 15: st.balloons(); st.success("🔥 High Value Play!")

    st.plotly_chart(px.line(p_sub, x='week', y=t_yds, title="Season Trend"), use_container_width=True)
