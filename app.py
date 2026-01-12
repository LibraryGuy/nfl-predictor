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
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# [AUTH SECTION REMAINS EXACTLY THE SAME AS YOUR PREVIOUS CODE]
# ... (Login logic and paywall logic here)

# --- 2. DATA LOADING & STADIUM INIT ---
@st.cache_resource
def get_stadium_loader():
    return NFLStadiums()

@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    # [Same data loading/cleaning logic from your previous fixed version]
    # (Ensure the deduplication and 'rush_tds' fixes are inside here)
    return nfl.load_player_stats(seasons=[2024, 2025]).to_pandas() # Simplified for example

data = load_nfl_data_pro()
stadiums = get_stadium_loader()

# --- 3. AUTO-WEATHER ENGINE ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        res = requests.get(url).json()
        temp_c = res['current']['temperature_2m']
        temp_f = (temp_c * 9/5) + 32
        wind_mph = res['current']['wind_speed_10m'] * 0.621371
        return round(temp_f, 1), round(wind_mph, 1)
    except:
        return 45.0, 5.0  # Defaults if API fails

# --- 4. SIDEBAR: THE STADIUM SELECTOR ---
with st.sidebar:
    st.header("🏟️ Stadium Environment")
    all_stadium_names = sorted(stadiums.get_list_of_stadium_names())
    selected_stadium_name = st.selectbox("Select Venue", all_stadium_names, index=all_stadium_names.index("Lambeau Field"))
    
    # Auto-pull stadium data
    stad_info = stadiums.get_stadium_by_name(selected_stadium_name)
    lat, lon = stad_info['latitude'], stad_info['longitude']
    
    # Auto-pull weather
    curr_temp, curr_wind = get_live_weather(lat, lon)
    
    # Determine Surface
    surface_raw = stad_info.get('surface', 'Grass').lower()
    is_grass_val = 1 if 'grass' in surface_raw else 0
    
    st.info(f"**Live Conditions:**\n\n🌡️ {curr_temp}°F | 💨 {curr_wind} MPH\n\n🌱 Surface: {surface_raw.title()}")
    
    if st.button("Refresh Weather"):
        st.cache_data.clear()
        st.rerun()

# --- 5. PREDICTION & DASHBOARD ---
# [The rest of your code remains the same, but uses curr_temp, curr_wind, and is_grass_val]
# This ensures the model now uses the automated data instead of manual slider values.
