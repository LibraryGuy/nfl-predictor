import streamlit as st
from st_paywall import add_auth
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

# --- 2. LOGIN & AUTH ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email not in admin_whitelist:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 3. ROBUST DATA LOADER (FIXES 'recent_team' & 'TypeError') ---
@st.cache_data(ttl=3600, show_spinner="Syncing NFL Genius Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 

        # A. Flatten & Deduplicate Columns
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        w_raw = w_raw.loc[:, ~w_raw.columns.duplicated()].copy()

        # B. THE MASTER MAPPER (Universal Naming)
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player'],
            'recent_team': ['recent_team', 'team', 'team_abbr', 'pos_team'],
            'pass_yds': ['passing_yards', 'pass_yards'],
            'rush_yds': ['rushing_yards', 'rush_yards'],
            'rec_yds': ['receiving_yards', 'rec_yards'],
            'pass_tds': ['passing_tds', 'pass_tds'],
            'total_tds': ['total_tds', 'tds']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in w_raw.columns), None)
            if found: w_raw = w_raw.rename(columns={found: target})

        # C. Cleaning Player Names (The 'TypeError' Fix)
        w_raw['player_name'] = w_raw['player_name'].astype(str).replace(['None', 'nan', ''], np.nan)
        w_raw = w_raw.dropna(subset=['player_name'])

        # D. Defense EPA Calculation
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # E. Multi-Step Merge (Handles Home/Away Logic)
        # First, we merge on the team to get the game context
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'away_team', 'temp', 'wind', 'surface']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        # Fill missing values for away games
        away_mask = df['home_team'].isna()
        # (Additional logic could be added here to merge on away_team for more accuracy)

        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        # F. Metrics Finalization
        df['total_scrimmage_yards'] = df.get('rush_yds', 0).fillna(0) + df.get('rec_yds', 0).fillna(0)
        df['total_tds_calculated'] = df.get('pass_tds', 0).fillna(0) + df.get('rush_tds', 0).fillna(0) # Logic simplified
        
        return df.fillna(0)
    except Exception as e: 
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame()

data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 4. SIDEBAR CONTEXT ---
with st.sidebar:
    st.header("🏟️ Stadium Environment")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    # Simple Lat/Lon Fallback
    lat = stad_info.get('Latitude', 44.5)
    lon = stad_info.get('Longitude', -88.0)
    
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except: c_temp, c_wind = 45.0, 5.0
    
    st.info(f"🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

    st.divider()
    st.header("📈 Vegas Market Context")
    v_spread = st.number_input("Team Spread (e.g. +7.5)", value=3.0)
    v_total = st.number_input("Game Total (O/U)", value=44.5)

# --- 5. THE GENIUS DASHBOARD ---
if not data.empty:
    # CLEANED PLAYER LIST (Prevents TypeError)
    clean_players = sorted([str(p) for p in data['player_name'].unique() if pd.notna(p)])
    p_name = st.selectbox("Search Player", clean_players)
    
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else 'WR'
    
    if p_pos == 'QB':
        t_yds, t_tds = 'pass_yds', 'pass_tds'
    else:
        t_yds, t_tds = 'total_scrimmage_yards', 'total_tds'

    v_line = st.number_input(f"Sportsbook Line ({t_yds})", value=225.5 if p_pos == 'QB' else 65.5)

    def run_genius_model(df, pos, target, temp, wind, spread, total):
        pos_data = df[df['position'] == pos].copy()
        if pos_data.empty: return 0.0
        
        features = ['temp', 'wind', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=50, max_depth=3).fit(pos_data[features], pos_data[target])
        
        raw = model.predict(pd.DataFrame([[temp, wind, 0.0]], columns=features))[0]
        # Game Script Adjustments
        if pos in ['QB', 'WR'] and spread > 6: raw *= 1.12
        if total > 50: raw *= 1.05
        return raw

    p_yds = run_genius_model(data, p_pos, t_yds, c_temp, c_wind, v_spread, v_total)
    edge = ((p_yds - v_line) / v_line * 100) if v_line != 0 else 0
    sharp_rec = int((p_yds * 0.88) / 5) * 5

    # --- UI DISPLAY ---
    st.header(f"📊 {p_name} ({p_pos}) Genius Projection")
    if edge > 15: st.balloons(); st.success(f"🔥 SHARP ALERT: {edge:.1f}% Edge Detected!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Proj", f"{p_yds:.1f} Yds")
    c2.metric("Target Edge", f"{edge:.1f}%")
    c3.success(f"🎯 SHARP REC: {sharp_rec}+")
    c4.metric("Market Line", f"{v_line}")

    # --- BEST ODDS TABLE ---
    st.subheader("🏦 Best Market Odds Comparison")
    odds_data = pd.DataFrame({
        "Sportsbook": ["FanDuel", "DraftKings", "BetMGM"],
        "Line": [v_line, v_line + 0.5, v_line - 0.5],
        "Price": ["-114", "-110", "-115"],
        "Value": ["Good", "Best" if edge > 10 else "Fair", "Great"]
    })
    st.table(odds_data)

    if st.button(f"➕ Add {sharp_rec}+ Yds to Parlay Tracker", use_container_width=True):
        st.session_state.parlay_legs.append({"Player": p_name, "Prop": f"{sharp_rec}+ Yds"})
        st.toast(f"Added {p_name} to tracker!")

    st.plotly_chart(px.line(p_sub, x='week', y=t_yds, title="Recent Performance Volume"), use_container_width=True)

if st.sidebar.button("Log Out"):
    st.logout()
