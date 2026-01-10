import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp: Advanced Analytics", layout="wide")
st.title("🏈 NFL Sharp: Analytics & Matchup Engine")

@st.cache_data(show_spinner="Deep-Syncing NFLverse Data (2023-2025)...")
def load_advanced_data():
    try:
        years = [2023, 2024, 2025]
        # Fetching core data
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas() 
        
        # Guard against empty downloads
        if weekly.empty or sched.empty:
            return pd.DataFrame()

        # Clean Column Names
        if 'recent_team' not in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team', 'team_abbr': 'recent_team'})
            
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds']
        for m in metrics: 
            if m in weekly.columns:
                weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly.get('rushing_tds', 0) + weekly.get('receiving_tds', 0)
        
        # Advanced Feature Engineering: Defense EPA
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Averages
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())
        
        # Merge Environment & Defense
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df # Always return a DataFrame
    except Exception as e:
        st.error(f"⚠️ Data Sync Failed: {e}")
        return pd.DataFrame() # Return empty DF instead of None

# --- INITIALIZE DATA ---
data = load_advanced_data()

# --- THE CRITICAL SAFETY GUARD ---
if data.empty:
    st.warning("🔄 The NFL data is currently unavailable. This is usually a temporary connection issue. Please refresh the page.")
    st.stop() # This prevents the crash on Line 70

# --- SIDEBAR & SELECTIONS ---
st.sidebar.header("Matchup Environment")
curr_wind = st.sidebar.slider("Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

# Now safe from TypeError
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Upcoming Opponent (Defense)", opp_list)

# ... [Rest of your prediction and dashboard code] ...
