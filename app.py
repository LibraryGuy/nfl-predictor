import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
import requests
from nfl_stadiums import NFLStadiums

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="NFL Sharp: Genius Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. MOBILE-FRIENDLY LOGIN GATE ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.markdown("### Wild Card Weekend")
    st.info("Log in with Google to access pro-tier analytics and bypass the paywall.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. WHITELIST & PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email in admin_whitelist:
    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")
else:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. DATA LOADING & STADIUMS ---
@st.cache_resource
def get_stadium_loader():
    return NFLStadiums()

@st.cache_data(ttl=3600, show_spinner="Syncing Latest NFL Stats...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 
        
        # Flatten and Deduplicate
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        w_raw = w_raw.loc[:, ~w_raw.columns.duplicated()].copy()
        
        # Mapping Column Names
        name_map = {'passing_tds': 'pass_tds', 'rushing_tds': 'rush_tds', 'receiving_tds': 'receiving_tds'}
        w_raw = w_raw.rename(columns={v: k for k, v in name_map.items() if v in w_raw.columns})
        
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Metrics & Total Logic
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'pass_tds', 'rush_tds', 'receiving_tds']
        for m in metrics:
            if m in w_raw.columns:
                w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        w_raw['total_scrimmage_yards'] = w_raw['rushing_yards'] + w_raw['receiving_yards']
        w_raw['total_tds'] = w_raw['rush_tds'] + w_raw['receiving_tds']
        
        # Defense EPA
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e: 
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame()

data = load_nfl_data_pro()
stadium_loader = get_stadium_loader()

# --- 5. GENIUS ENGINE: WEATHER & GAME SCRIPT ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        res = requests.get(url, timeout=5).json()
        return (res['current']['temperature_2m'] * 1.8 + 32), (res['current']['wind_speed_10m'] * 0.62)
    except: return 45.0, 5.0

with st.sidebar:
    st.header("🏟️ Game Environment")
    all_stads = sorted(stadium_loader.get_list_of_stadium_names())
    sel_stad = st.selectbox("Venue", all_stads, index=all_stads.index("Lambeau Field") if "Lambeau Field" in all_stads else 0)
    stad_info = stadium_loader.get_stadium_by_name(sel_stad)
    lat = stad_info.get('Latitude', stad_info.get('latitude', 44.5))
    lon = stad_info.get('Longitude', stad_info.get('longitude', -88.0))
    curr_temp, curr_wind = get_live_weather(lat, lon)
    surf = stad_info.get('Surface', 'Grass').lower()
    is_grass = 1 if 'grass' in surf else 0
    st.info(f"🌡️ {curr_temp:.0f}°F | 💨 {curr_wind:.0f} MPH | 🌱 {surf.title()}")

    st.divider()
    st.header("📈 Vegas Context")
    v_spread = st.number_input("Team Spread (e.g. +7.5)", value=3.0, help="Underdogs (+ pts) often throw more to catch up.")
    v_total = st.number_input("Game Total (O/U)", value=44.5, help="High totals favor overs.")

# --- 6. PLAYER SELECTION & PREDICTION ---
if not data.empty:
    p_name = st.selectbox("Search Player", sorted(data['player_name'].unique()))
    o_team = st.selectbox("Opponent Defense", sorted(data['opponent_team'].unique()))
    
    p_sub = data[data['player_name'] == p_name]
    p_pos = p_sub['position'].iloc[-1]
    
    # Target Setup
    if p_pos == 'QB':
        t_yds, t_tds = 'passing_yards', 'pass_tds'
        td_lbl, suffix = "Pass TD Prob", "Pass TDs"
    else:
        t_yds, t_tds = 'total_scrimmage_yards', 'total_tds'
        td_lbl, suffix = "Anytime TD Prob", "Anytime TD"

    v_line = st.number_input(f"Sportsbook Line ({t_yds})", value=225.5 if p_pos == 'QB' else 65.5)

    def predict_genius(df, sub, pos, target, temp, wind, grass, opp, spread, total):
        pos_data = df[df['position'] == pos].copy()
        features = ['temp', 'wind', 'is_grass', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=50, max_depth=3).fit(pos_data[features], pos_data[target])
        opp_epa = df[df['opponent_team'] == opp]['def_epa_allowed'].mean()
        
        raw = model.predict(pd.DataFrame([[temp, wind, grass, opp_epa]], columns=features))[0]
        
        # Game Script Modifiers
        if pos in ['QB', 'WR'] and spread > 6: raw *= 1.08 # Trailing boost
        if total > 50: raw *= 1.05 # Shootout boost
        
        avg = sub[target].mean()
        return raw if raw > (avg * 0.4) else avg

    proj_yds = predict_genius(data, p_sub, p_pos, t_yds, curr_temp, curr_wind, is_grass, o_team, v_spread, v_total)
    proj_tds = predict_genius(data, p_sub, p_pos, t_tds, curr_temp, curr_wind, is_grass, o_team, v_spread, v_total)
    
    rec_yds = int((proj_yds * 0.85) / 5) * 5
    td_prob = min(max(proj_tds * 100, 5), 95) if p_pos != 'QB' else min(max(proj_tds * 45, 5), 95)
    edge = ((proj_yds - v_line) / v_line) * 100

    # --- 7. DASHBOARD DISPLAY ---
    st.header(f"📊 {p_name} ({p_pos}) Projection")
    if edge > 15: st.balloons(); st.success(f"🔥 GENIUS SIGNAL: {edge:.1f}% Edge Detected!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Proj", f"{proj_yds:.1f} Yds")
    c2.metric(td_lbl, f"{td_prob:.1f}%")
    c3.success(f"🎯 SHARP REC: {rec_yds}+")
    c4.metric("Vegas Edge", f"{edge:.1f}%", delta=f"{proj_yds - v_line:.1f} Yds")

    # --- 8. BEST ODDS TABLE ---
    st.subheader("🏦 Best Market Odds")
    odds_data = {
        "Sportsbook": ["FanDuel", "DraftKings", "BetMGM"],
        "Line": [v_line, v_line + 0.5, v_line - 0.5],
        "Price": ["-114", "-110", "-115"],
        "Value Score": ["B", "A" if edge > 10 else "B", "C"]
    }
    st.table(pd.DataFrame(odds_data))

    b1, b2 = st.columns(2)
    if b1.button(f"➕ Add {rec_yds}+ Yds to Parlay", use_container_width=True):
        st.session_state.parlay_legs.append({"Player": p_name, "Prop": f"{rec_yds}+ Yds"})
    if b2.button(f"🔥 Add {suffix} to Parlay", use_container_width=True):
        st.session_state.parlay_legs.append({"Player": p_name, "Prop": suffix})

    st.plotly_chart(px.line(p_sub, x='week', y=t_yds, title="Performance Trends"), use_container_width=True)

if st.sidebar.button("Log Out"):
    st.logout()
