import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp: Analytics & Matchup Engine", layout="wide")
st.title("🏈 NFL Sharp: Analytics & Prediction Dashboard")

@st.cache_data(show_spinner="Updating NFL Database...")
def load_nfl_data_pro():
    try:
        # Load multiple seasons for a stable predictive model
        years = [2023, 2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas() 
        
        # Guard 1: If nflreadpy returns None or an empty set, return an empty DataFrame
        if weekly is None or weekly.empty:
            return pd.DataFrame()

        # Standardize Team Columns
        if 'recent_team' not in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team', 'team_abbr': 'recent_team'})
        
        # Ensure Numeric Columns
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds']
        for m in metrics: 
            if m in weekly.columns:
                weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly.get('rushing_tds', 0) + weekly.get('receiving_tds', 0)
        
        # Defense Stats (EPA)
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Averages (Player Velocity)
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())

        # Master Data Merge
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df # Success
    except Exception as e:
        # Guard 2: If the network fails, return empty DF instead of None
        return pd.DataFrame()

# --- INITIALIZE DATA ---
data = load_nfl_data_pro()

# --- THE CRITICAL FIX: SAFETY GUARD ---
# Guard 3: This halts the app BEFORE line 83 if the data didn't load.
if data.empty or 'player_name' not in data.columns:
    st.warning("🔄 The NFL data server is taking too long to respond. This is common during peak hours.")
    if st.button("Force Re-Sync Data"):
        st.cache_data.clear()
        st.rerun()
    st.stop() # Prevents the "NoneType" crash on line 83

# --- DASHBOARD STARTS HERE ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Type", ["Grass", "Turf"]) == "Grass" else 0

# Line 83: This is now protected from crashing
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Upcoming Opponent", opp_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Vegas Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- PREDICTION ENGINE ---
def get_advanced_pred(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_df = df[df['position'] == pos].copy()
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed', roll_col]
    
    # Train robust model
    model = XGBRegressor(n_estimators=50, max_depth=3, reg_lambda=15).fit(pos_df[features].fillna(0), pos_df[target_stat])
    
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    
    input_data = pd.DataFrame([[temp, wind, is_grass, opp_epa, p_latest[roll_col]]], columns=features)
    return max(model.predict(input_data)[0], p_latest[roll_col] * 0.55)

# --- MAIN VIEW ---
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_advanced_pred(data, selected_player, target, curr_temp, curr_wind, is_grass_val, selected_opp)

st.header(f"📊 {selected_player} vs {selected_opp}")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model Projection", f"{proj:.1f} Yds")
    st.success(f"🎯 RECOMMEND: {int(proj*0.85/5)*5}+ Yards")
with c2:
    edge = proj - vegas_line
    st.metric("Vegas Edge", f"{edge:.1f} yds", delta=f"{((edge)/vegas_line)*100:.1f}%")
with c3:
    st.plotly_chart(px.line(player_subset, x='week', y=[target, 'yards_roll3' if player_pos != 'QB' else 'pass_roll3'], title="Trend Analysis"), use_container_width=True)

# --- TD GRAPH & HISTORY ---
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.subheader(f"🏟️ Career Performance vs. {selected_opp}")
    m_hist = player_subset[player_subset['opponent_team'] == selected_opp][['season', 'week', target, 'total_scrimmage_tds']]
    if not m_hist.empty:
        st.table(m_hist)
    else:
        st.info("No historical head-to-head found.")
with g2:
    st.plotly_chart(px.scatter(player_subset, x='total_scrimmage_yards', y='total_scrimmage_tds', trendline="ols", title="TD Efficiency Map"), use_container_width=True)
