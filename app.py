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
# These venues are often mislabeled as domes/indoor due to partial roofs or recent renovations
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
        
        # Match current hour (today's date)
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

# --- 2. CORE LOGIC & DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']:
            df[col] = df.get(col, 0).fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

raw_data = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

# --- 3. UI RENDERING ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if not p_df.empty else "WR"

        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if selected_market == "Yards" else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        is_td_market = "tds" in stat_col

        market_line = st.number_input("Sportsbook Line", value=0.5 if is_td_market else 50.0, step=0.5)
        
        # --- ROBUST WEATHER UI ---
        st.subheader("🏟️ Venue & Weather")
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        game_time = st.time_input("Kickoff Time (Local)", time(13, 0))
        
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type_raw = str(stad_obj.get('roof_type', 'Outdoor'))
        
        # Override check
        is_forced_outdoor = any(s in sel_stad_name for s in FORCE_OUTDOOR)
        is_actually_indoor = any(x in roof_type_raw.lower() for x in ['dome', 'closed', 'indoor']) and not is_forced_outdoor

        if is_actually_indoor:
            st.success(f"🏟️ Indoor: Conditions Controlled")
            w_wind, w_temp, w_precip = 0, 70, "None"
        else:
            lat, lon = (stad_obj.get('latitude'), stad_obj.get('longitude')) if stad_obj else (None, None)
            query_key = f"{sel_stad_name}_{game_time.hour}"
            
            # Fetch new weather if stadium or time changed
            if st.session_state["last_stadium_query"] != query_key and lat and lon:
                with st.spinner(f"Fetching Live Forecast for {sel_stad_name}..."):
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

    # --- Calculation Logic ---
    p_mean = p_df[stat_col].mean()
    p_std = p_df[stat_col].std() if len(p_df) > 1 else (p_mean * 0.4)
    weather_mult, weather_reason = get_weather_multiplier(roof_type_raw if not is_forced_outdoor else "Outdoor", w_wind, w_temp, w_precip, p_pos)
    model_proj = p_mean * weather_mult # Simplified projection for brevity
    
    st.title(f"📊 {selected_p} Intelligence Hub")
    st.metric("Model Projection", f"{round(model_proj, 1)} {selected_market}", f"Weather: {weather_reason}")import streamlit as st
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
# These venues are often mislabeled as domes/indoor due to partial roofs or recent renovations
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
        
        # Match current hour (today's date)
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

# --- 2. CORE LOGIC & DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']:
            df[col] = df.get(col, 0).fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

raw_data = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

# --- 3. UI RENDERING ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if not p_df.empty else "WR"

        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if selected_market == "Yards" else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        is_td_market = "tds" in stat_col

        market_line = st.number_input("Sportsbook Line", value=0.5 if is_td_market else 50.0, step=0.5)
        
        # --- ROBUST WEATHER UI ---
        st.subheader("🏟️ Venue & Weather")
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        game_time = st.time_input("Kickoff Time (Local)", time(13, 0))
        
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type_raw = str(stad_obj.get('roof_type', 'Outdoor'))
        
        # Override check
        is_forced_outdoor = any(s in sel_stad_name for s in FORCE_OUTDOOR)
        is_actually_indoor = any(x in roof_type_raw.lower() for x in ['dome', 'closed', 'indoor']) and not is_forced_outdoor

        if is_actually_indoor:
            st.success(f"🏟️ Indoor: Conditions Controlled")
            w_wind, w_temp, w_precip = 0, 70, "None"
        else:
            lat, lon = (stad_obj.get('latitude'), stad_obj.get('longitude')) if stad_obj else (None, None)
            query_key = f"{sel_stad_name}_{game_time.hour}"
            
            # Fetch new weather if stadium or time changed
            if st.session_state["last_stadium_query"] != query_key and lat and lon:
                with st.spinner(f"Fetching Live Forecast for {sel_stad_name}..."):
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

    # --- Calculation Logic ---
    p_mean = p_df[stat_col].mean()
    p_std = p_df[stat_col].std() if len(p_df) > 1 else (p_mean * 0.4)
    weather_mult, weather_reason = get_weather_multiplier(roof_type_raw if not is_forced_outdoor else "Outdoor", w_wind, w_temp, w_precip, p_pos)
    model_proj = p_mean * weather_mult # Simplified projection for brevity
    
    st.title(f"📊 {selected_p} Intelligence Hub")
    st.metric("Model Projection", f"{round(model_proj, 1)} {selected_market}", f"Weather: {weather_reason}")
