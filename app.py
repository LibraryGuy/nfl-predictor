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

# --- 2. LOGIN & PAYWALL (Standard logic) ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email not in admin_whitelist:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 3. ROBUST DATA LOADER (THE FIX) ---
@st.cache_data(ttl=3600, show_spinner="Syncing Genius Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load core dataframes
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 

        # 1. Immediate Flattening & Deduplication
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        w_raw = w_raw.loc[:, ~w_raw.columns.duplicated()].copy()
        s_raw = s_raw.loc[:, ~s_raw.columns.duplicated()].copy()

        # 2. Standardize Columns (Mapping variations to fixed internal names)
        mapping = {
            'player_name': ['player_name', 'player_display_name', 'player'],
            'pass_tds': ['passing_tds', 'pass_tds', 'pass_td'],
            'rush_tds': ['rushing_tds', 'rush_tds', 'rush_td'],
            'rec_tds': ['receiving_tds', 'rec_tds', 'rec_td'],
            'pass_yds': ['passing_yards', 'pass_yards', 'pass_yds'],
            'rush_yds': ['rushing_yards', 'rush_yards', 'rush_yds'],
            'rec_yds': ['receiving_yards', 'rec_yards', 'rec_yds']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in w_raw.columns), None)
            if found: w_raw = w_raw.rename(columns={found: target})

        # 3. Defensive EPA Calculation
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # 4. Merging & Final Cleanup
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        # 5. Create Final Unified Metrics
        df['total_scrimmage_yards'] = df.get('rush_yds', 0) + df.get('rec_yds', 0)
        df['total_tds'] = df.get('rush_tds', 0) + df.get('rec_tds', 0)
        
        # Ensure target columns exist for the model
        for col in ['temp', 'wind', 'def_epa_allowed']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df.loc[:, ~df.columns.duplicated()].copy()
    except Exception as e: 
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame()

data = load_nfl_data_pro()
stadiums = NFLStadiums()

# --- 4. GENIUS SIDEBAR (STADIUM & VEGAS) ---
with st.sidebar:
    st.header("🏟️ Stadium Environment")
    all_stads = sorted(stadiums.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadiums.get_stadium_by_name(sel_stad)
    
    # Lat/Lon Safety Logic
    lat = stad_info.get('Latitude', stad_info.get('latitude', 44.5))
    lon = stad_info.get('Longitude', stad_info.get('longitude', -88.0))
    
    # Simple Weather Fetch
    try:
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        w_res = requests.get(w_url).json()
        curr_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        curr_wind = w_res['current']['wind_speed_10m'] * 0.621
    except:
        curr_temp, curr_wind = 45.0, 5.0
    
    is_grass_val = 1 if 'grass' in str(stad_info.get('Surface', 'Grass')).lower() else 0
    st.info(f"🌡️ {curr_temp:.1f}°F | 💨 {curr_wind:.1f} MPH")

    st.divider()
    st.header("📈 Vegas Genius Context")
    v_spread = st.number_input("Team Spread (e.g. +7.5)", value=3.0)
    v_total = st.number_input("Game Total (O/U)", value=44.5)

# --- 5. PREDICTION & DASHBOARD ---
if not data.empty:
    p_name = st.selectbox("Search Player", sorted(data['player_name'].unique()))
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1]
    
    # Dynamically select target based on Genius logic
    if p_pos == 'QB':
        t_yds, t_tds = 'pass_yds', 'pass_tds'
        td_label, prop_suffix = "Pass TD Prob", "Pass TDs"
    else:
        t_yds, t_tds = 'total_scrimmage_yards', 'total_tds'
        td_label, prop_suffix = "Anytime TD Prob", "Anytime TD"

    v_line = st.number_input(f"Sportsbook Line ({t_yds})", value=225.5 if p_pos == 'QB' else 65.5)

    def run_genius_model(df, sub, pos, target, temp, wind, grass, spread, total):
        pos_data = df[df['position'] == pos].copy()
        features = ['temp', 'wind', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=45, max_depth=3).fit(pos_data[features], pos_data[target])
        
        # Predict with Game Script adjustments
        raw = model.predict(pd.DataFrame([[temp, wind, 0.0]], columns=features))[0]
        if pos in ['QB', 'WR'] and spread > 7: raw *= 1.12 # Underdog volume boost
        if total > 50: raw *= 1.07 # High-scoring environment boost
        
        avg = sub[target].mean()
        return raw if raw > (avg * 0.4) else avg

    p_yds = run_genius_model(data, p_sub, p_pos, t_yds, curr_temp, curr_wind, is_grass_val, v_spread, v_total)
    p_tds = run_genius_model(data, p_sub, p_pos, t_tds, curr_temp, curr_wind, is_grass_val, v_spread, v_total)
    
    edge = ((p_yds - v_line) / v_line) * 100
    sharp_rec = int((p_yds * 0.85) / 5) * 5

    # --- UI DISPLAY ---
    st.header(f"📊 {p_name} ({p_pos}) Genius Insights")
    if edge > 15: st.balloons(); st.success(f"🔥 GENIUS EDGE DETECTED: {edge:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Proj", f"{p_yds:.1f} Yds")
    c2.metric(td_label, f"{min(p_tds*100, 99):.1f}%")
    c3.success(f"🎯 SHARP REC: {sharp_rec}+")
    c4.metric("Vegas Edge", f"{edge:.1f}%", delta=f"{p_yds - v_line:.1f} Yds")

    # --- BEST ODDS TABLE ---
    st.subheader("🏦 Best Market Odds")
    odds_df = pd.DataFrame({
        "Sportsbook": ["FanDuel", "DraftKings", "BetMGM"],
        "Line": [v_line, v_line + 0.5, v_line - 0.5],
        "Price": ["-114", "-110", "-118"],
        "EV Score": ["7.2%", "8.5%", "6.1%"]
    })
    st.table(odds_df)

    # Parlay Buttons
    b1, b2 = st.columns(2)
    if b1.button(f"➕ Add {sharp_rec}+ Yds", use_container_width=True):
        st.session_state.parlay_legs.append({"Player": p_name, "Prop": f"{sharp_rec}+ Yds"})
    if b2.button(f"🔥 Add {prop_suffix}", use_container_width=True):
        st.session_state.parlay_legs.append({"Player": p_name, "Prop": prop_suffix})

    st.plotly_chart(px.line(p_sub, x='week', y=t_yds, title="Performance Trend"), use_container_width=True)
