import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, time
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

if "w_temp_val" not in st.session_state: st.session_state["w_temp_val"] = 70
if "w_wind_val" not in st.session_state: st.session_state["w_wind_val"] = 0
if "w_precip_val" not in st.session_state: st.session_state["w_precip_val"] = "None"
if "last_stadium_query" not in st.session_state: st.session_state["last_stadium_query"] = ""

FORCE_OUTDOOR = ["Gillette Stadium", "Lumen Field", "Hard Rock Stadium", "Acrisure Stadium", "Lambeau Field"]

# --- 2. LOGIC FUNCTIONS ---
def fetch_stadium_weather(lat, lon, game_time):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"], "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "auto"}
        res = requests.get(url, params=params, timeout=5).json()
        target_str = datetime.now().strftime(f"%Y-%m-%dT{game_time.strftime('%H')}:00")
        idx = res['hourly']['time'].index(target_str) if target_str in res['hourly']['time'] else 0
        t, w, p_v = res['hourly']['temperature_2m'][idx], res['hourly']['wind_speed_10m'][idx], res['hourly']['precipitation'][idx]
        p_t = "Rain" if p_v > 0.1 else "Snow" if (t < 32 and p_v > 0) else "None"
        return t, w, p_t
    except: return 70, 0, "None"

def get_weather_multiplier(roof, wind, temp, precip, pos):
    if any(x in roof.lower() for x in ['dome', 'closed', 'indoor']): return 1.0, "Dome"
    m, reasons = 1.0, []
    if wind >= 15:
        p = 0.05 if wind < 20 else 0.12
        if pos in ['QB', 'WR', 'TE']: m -= p; reasons.append(f"Wind (-{int(p*100)}%)")
        else: m += 0.03; reasons.append("Wind Vol. (+3%)")
    if precip != "None": m -= 0.05; reasons.append(f"{precip} (-5%)")
    return round(m, 2), (" + ".join(reasons) if reasons else "Fair")

def generate_smart_legs(p_name, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td):
    # 1. Base Market Leg
    risk_map = {"Conservative (-104)": -0.5, "Standard (+105)": 0.0, "Aggressive (+200)": 0.5}
    primary_val = max(0.5 if is_td else 0, round(p_mean + (risk_map[risk_level] * p_std)))
    legs = [{"label": f"{p_name}: {primary_val}+ {stat_label}", "type": "Primary", "color": "info"}]
    
    # 2. AVOID LEG (The "Under" Logic)
    # Use Poisson to find P(X=0). If mean is 0.4, P(0) = e^-0.4
    prob_zero = poisson.pmf(0, p_mean) 
    td_market_name = "Passing TD" if p_pos == "QB" else "Anytime TD"
    
    if prob_zero > 0.55: # If >55% chance of 0 TDs
        legs.append({"label": f"AVOID: {p_name} {td_market_name} (Prob 0: {int(prob_zero*100)}%)", "type": "Risk Alert", "color": "error"})

    # 3. Teammate Correlation
    tm = data[(data['team'] == p_team) & (data['player_name'] != p_name)].tail(10)
    if not tm.empty:
        if p_pos == 'QB':
            tgt = tm[tm['position'].isin(['WR', 'TE'])].groupby('player_name')['receiving_yards'].mean().idxmax()
            legs.append({"label": f"{tgt}: 40+ Rec Yds", "type": "Correlation", "color": "success"})
        else:
            qb = tm[tm['position'] == 'QB']['player_name'].iloc[-1] if 'QB' in tm['position'].values else "QB"
            legs.append({"label": f"{qb}: 200+ Pass Yds", "type": "QB Stack", "color": "success"})
    return legs

# --- 3. DATA & UI ---
@st.cache_data(ttl=3600)
def load_data():
    df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
    df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
    for c in ['passing_yards','rushing_yards','receiving_yards','passing_tds','rushing_tds','receiving_tds']: df[c] = df.get(c, 0).fillna(0)
    return df.dropna(subset=['player_name', 'opponent', 'position'])

data = load_data()
stadium_client = NFLStadiums()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Selection")
        sel_p = st.selectbox("Player", sorted(data['player_name'].unique()))
        p_df = data[data['player_name'] == sel_p].copy()
        p_pos, p_team = p_df['position'].iloc[-1], p_df['team'].iloc[-1]
        
        mkt = st.radio("Market", ["Yards", "Touchdowns"])
        if mkt == "Yards":
            stat_col = 'passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards'
        else:
            stat_col = 'passing_tds' if p_pos == 'QB' else 'receiving_tds' # Simplified for logic
        
        line = st.number_input("Line", value=0.5 if "tds" in stat_col else 50.0)
        risk = st.radio("Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
        
        st.subheader("🏟️ Weather")
        venue = st.selectbox("Venue", sorted(stadium_client.get_list_of_stadium_names()))
        time_in = st.time_input("Kickoff", time(13, 0))
        stad = stadium_client.get_stadium_by_name(venue)
        is_outdoor = not any(x in str(stad.get('roof_type','')).lower() for x in ['dome','closed']) or any(s in venue for s in FORCE_OUTDOOR)
        
        if is_outdoor:
            q_key = f"{venue}_{time_in.hour}"
            if st.session_state.last_stadium_query != q_key:
                st.session_state.w_temp_val, st.session_state.w_wind_val, st.session_state.w_precip_val = fetch_stadium_weather(stad['latitude'], stad['longitude'], time_in)
                st.session_state.last_stadium_query = q_key
            w_t = st.slider("Temp", -10, 100, key="w_temp_val")
            w_w = st.slider("Wind", 0, 40, key="w_wind_val")
            w_p = st.selectbox("Precip", ["None", "Rain", "Snow"], index=["None","Rain","Snow"].index(st.session_state.w_precip_val))
        else: w_t, w_w, w_p = 70, 0, "None"

    # --- FINAL CALCULATIONS ---
    p_mean, p_std = p_df[stat_col].mean(), (p_df[stat_col].std() if len(p_df)>1 else p_df[stat_col].mean()*0.4)
    w_m, w_r = get_weather_multiplier("Outdoor" if is_outdoor else "Dome", w_w, w_t, w_p, p_pos)
    model_proj = p_mean * w_m
    prob = (1 - poisson.cdf(max(0, line-0.5), model_proj))*100 if "tds" in stat_col else (1 - norm.cdf(line, model_proj, p_std))*100

    # --- DASHBOARD ---
    st.title(f"📊 {sel_p} Intelligence Hub")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("Model Projection", f"{round(model_proj, 1)} {mkt}", f"{round(prob, 1)}% Prob")
        if not p_df.empty:
            fig = go.Figure(go.Bar(x=[f"Wk {w}" for w in p_df.tail(5)['week']], y=p_df.tail(5)[stat_col], marker_color='#00ff96'))
            fig.add_hline(y=line, line_dash="dash", line_color="#ff4b4b")
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🚀 Smart Parlay Legs")
        for leg in generate_smart_legs(sel_p, p_pos, p_team, model_proj, p_std, mkt, data, risk, "tds" in stat_col):
            if leg['color'] == "error": st.error(f"⚠️ {leg['label']}")
            elif leg['color'] == "success": st.success(f"🔗 {leg['label']}")
            else: st.info(f"🔹 {leg['label']}")
        
        st.write(f"**Weather Impact**: {w_r}")
