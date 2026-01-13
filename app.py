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

# --- INITIALIZE SESSION STATE KEYS ---
if "w_temp_val" not in st.session_state:
    st.session_state["w_temp_val"] = 70
if "w_wind_val" not in st.session_state:
    st.session_state["w_wind_val"] = 0
if "w_precip_val" not in st.session_state:
    st.session_state["w_precip_val"] = "None"
if "last_stadium_query" not in st.session_state:
    st.session_state["last_stadium_query"] = ""

# --- STADIUM OVERRIDES ---
# These stadiums are often mislabeled as domes/indoor due to partial roofs or atriums
FORCE_OUTDOOR = [
    "Gillette Stadium", "Lumen Field", "Hard Rock Stadium", 
    "Acrisure Stadium", "GEHA Field at Arrowhead Stadium",
    "Highmark Stadium", "Lambeau Field", "Soldier Field"
]

# --- AUTOMATED WEATHER FETCH ---
def fetch_stadium_weather(lat, lon, game_time):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto", "forecast_days": 7
        }
        response = requests.get(url, params=params, timeout=5).json()
        hourly = response.get('hourly', {})
        times = hourly.get('time', [])
        target_str = datetime.now().strftime(f"%Y-%m-%dT{game_time.strftime('%H')}:00")
        
        idx = times.index(target_str) if target_str in times else 0
        temp = hourly.get('temperature_2m', [70])[idx]
        wind = hourly.get('wind_speed_10m', [0])[idx]
        precip_val = hourly.get('precipitation', [0])[idx]
        
        precip_type = "None"
        if precip_val > 0.1: precip_type = "Rain"
        if temp < 32 and precip_val > 0: precip_type = "Snow"
        
        return temp, wind, precip_type
    except Exception:
        return 70, 0, "None"

# --- WEATHER IMPACT LOGIC ---
def get_weather_multiplier(roof_type, wind, temp, precip, p_pos):
    # Check for actual domes or closed retractable roofs
    if any(x in roof_type.lower() for x in ['dome', 'closed', 'indoor']):
        return 1.0, "Dome (No Impact)"
    
    multiplier, impact_reasons = 1.0, []
    if wind >= 15:
        penalty = 0.05 if wind < 20 else 0.12
        if p_pos in ['QB', 'WR', 'TE']:
            multiplier -= penalty
            impact_reasons.append(f"High Wind (-{int(penalty*100)}%)")
        elif p_pos == 'RB':
            multiplier += 0.03
            impact_reasons.append("Wind Vol. Boost (+3%)")
    if precip in ['Rain', 'Snow']:
        multiplier -= 0.05
        impact_reasons.append(f"{precip} (-5%)")
    if temp <= 20:
        multiplier -= 0.03
        impact_reasons.append("Extreme Cold (-3%)")
    return round(multiplier, 2), (" + ".join(impact_reasons) if impact_reasons else "Fair Weather")

# --- CORE LOGIC & DATA LOADING (Keep original) ---
def get_dynamic_sos(data, stat_col):
    if data.empty: return {}
    league_avg = data[stat_col].mean()
    return (data.groupby('opponent')[stat_col].mean() / league_avg).to_dict()

def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td=False):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    offset = risk_map[risk_level]["offset"]
    primary_val = 0.5 if is_td and risk_level != "Aggressive (+200)" else 1.5 if is_td else max(0, round(p_mean + (offset * p_std)))
    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)].sort_values(['season', 'week'], ascending=False)
    if not teammates.empty:
        latest_season = teammates['season'].max()
        if p_pos == 'QB':
            valid_targets = teammates[(teammates['season'] == latest_season) & (teammates['position'].isin(['WR', 'TE']))]
            if not valid_targets.empty:
                top_target = valid_targets.groupby('player_name')['receiving_yards'].sum().idxmax()
                leg_val = 40 if risk_level == "Conservative (-104)" else 60
                parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
    return parlay_legs

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']:
            df[col] = df.get(col, 0).fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

raw_data, schedules = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

# --- UI RENDERING ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_team = p_df['team'].iloc[-1] if not p_df.empty else "N/A"
        p_pos = p_df['position'].iloc[-1] if not p_df.empty else "WR"

        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if selected_market == "Yards" else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        is_td_market = "tds" in stat_col

        market_line = st.number_input("Sportsbook Line", value=0.5 if is_td_market else 50.0, step=0.5)
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
        
        # --- ROBUST WEATHER LOGIC ---
        st.subheader("🏟️ Venue & Weather")
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        game_time = st.time_input("Kickoff Time (Local)", time(13, 0))
        
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type_raw = str(stad_obj.get('roof_type', 'Outdoor'))
        
        # Determine if we should force Outdoor behavior
        is_forced_outdoor = any(s in sel_stad_name for s in FORCE_OUTDOOR)
        is_actually_indoor = any(x in roof_type_raw.lower() for x in ['dome', 'closed', 'indoor']) and not is_forced_outdoor

        if is_actually_indoor:
            st.success(f"🏟️ Indoor: Conditions Controlled")
            w_wind, w_temp, w_precip = 0, 70, "None"
        else:
            lat, lon = (stad_obj.get('latitude'), stad_obj.get('longitude')) if stad_obj else (None, None)
            query_key = f"{sel_stad_name}_{game_time.hour}"
            
            if st.session_state["last_stadium_query"] != query_key and lat and lon:
                with st.spinner(f"Syncing Live Weather..."):
                    l_temp, l_wind, l_precip = fetch_stadium_weather(lat, lon, game_time)
                    st.session_state["w_temp_val"] = int(l_temp)
                    st.session_state["w_wind_val"] = int(l_wind)
                    st.session_state["w_precip_val"] = l_precip
                    st.session_state["last_stadium_query"] = query_key

            w_temp = st.slider("Temperature (F)", -10, 100, key="w_temp_val")
            w_wind = st.slider("Wind Speed (MPH)", 0, 40, key="w_wind_val")
            p_opts = ["None", "Rain", "Snow"]
            def_p_idx = p_opts.index(st.session_state.get("w_precip_val", "None"))
            w_precip = st.selectbox("Precipitation", p_opts, index=def_p_idx)

    # --- Calculations & Main View ---
    p_mean = p_df[stat_col].mean()
    p_std = p_df[stat_col].std() if len(p_df) > 1 else (p_mean * 0.4)
    weather_mult, weather_reason = get_weather_multiplier(roof_type_raw if not is_forced_outdoor else "Outdoor", w_wind, w_temp, w_precip, p_pos)
    model_proj = p_mean * get_dynamic_sos(data, stat_col).get(selected_opp, 1.0) * weather_mult
    win_prob = (1 - poisson.cdf(max(0, market_line - 0.5), model_proj)) * 100 if is_td_market else (1 - norm.cdf(market_line, model_proj, p_std)) * 100

    st.title(f"📊 {selected_p} Intelligence Hub")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("Model Projection", f"{round(model_proj, 1)} {selected_market}", f"{round(win_prob, 1)}% Prob > Line")
        last_5 = p_df.tail(5).copy()
        last_5['hit'] = last_5[stat_col] >= market_line
        fig = go.Figure(go.Bar(x=[f"Wk {w}" for w in last_5['week']], y=last_5[stat_col], marker_color=['#00ff96' if h else '#4a4a4a' for h in last_5['hit']]))
        fig.add_hline(y=market_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Line")
        fig.update_layout(title="Last 5 Games vs Current Line", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("🚀 Smart Parlay Legs")
        parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, p_std, selected_market, data, risk_pref, is_td_market)
        for leg in parlay_legs: st.info(f"🔹 **{leg['type']}**: {leg['label']}")
        st.write(f"**Weather Adjustment:** {weather_reason}")
