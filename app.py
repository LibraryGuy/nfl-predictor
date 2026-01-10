import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp Pro Predictor", layout="wide")
st.title("🏈 NFL Sharp: Analytics & Parlay Builder")

@st.cache_data(show_spinner="Updating NFL Data...")
def load_nfl_data_pro():
    try:
        # Pulling multiple years for deep history and stable models
        years = [2023, 2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas() 
        
        # Check if data actually arrived
        if weekly is None or weekly.empty:
            return pd.DataFrame() # Return empty DF, NOT None

        # Standardize Team Columns
        if 'recent_team' not in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team', 'team_abbr': 'recent_team'})
        
        # Numeric Cleaning
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for m in metrics: 
            if m in weekly.columns:
                weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
        
        # Defense Stats (EPA Allowed)
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Averages (The "Velocity" of a player)
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())

        # Merge Environment & Defense
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame()

# --- INITIALIZE & PREVENT CRASH ---
data = load_nfl_data_pro()

if data.empty or 'player_name' not in data.columns:
    st.warning("⚠️ NFL Data is currently unavailable (Likely a connection timeout).")
    if st.button("Retry Connection"):
        st.cache_data.clear()
        st.rerun()
    st.stop() # THIS PREVENTS THE TYPEERROR ON LINE 72

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Type", ["Grass", "Turf"]) == "Grass" else 0

# --- PLAYER & OPPONENT SELECTION ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Opponent (Defense)", opp_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Enter Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- ADVANCED PREDICTION ENGINE ---
def get_advanced_prediction(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_df = df[df['position'] == pos].copy()
    
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed', roll_col]
    
    # Train Stable Position-Weighted Model
    model = XGBRegressor(n_estimators=50, max_depth=3, reg_lambda=15).fit(pos_df[features].fillna(0), pos_df[target_stat])
    
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    
    input_df = pd.DataFrame([[temp, wind, is_grass, opp_epa, p_latest[roll_col]]], columns=features)
    pred = model.predict(input_df)[0]
    return max(pred, p_latest[roll_col] * 0.55)

# --- DASHBOARD LAYOUT ---
target_stat = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_advanced_prediction(data, selected_player, target_stat, curr_temp, curr_wind, is_grass_val, selected_opp)

st.header(f"📊 {selected_player} vs {selected_opp}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Projection", f"{proj:.1f}")
    st.success(f"🎯 RECOMMENDED: {int(proj*0.85/5)*5}+ Yards")
with col2:
    edge = proj - vegas_line
    st.metric("Vegas Line Edge", f"{(edge/vegas_line)*100:.1f}%", delta=f"{edge:.1f} yds")
with col3:
    if proj > data[data['player_name']==selected_player][target_stat].median() * 1.4:
        st.error("⚠️ FADE ALERT: Historical Outlier")
    else:
        st.info("⚖️ NEUTRAL: Value aligns with trends.")

# --- NEW: MATCHUP HISTORY TABLE ---
st.subheader(f"🏟️ Career Performance vs. {selected_opp}")
matchup_history = player_subset[player_subset['opponent_team'] == selected_opp][['season', 'week', target_stat, 'total_scrimmage_tds', 'surface']]
if not matchup_history.empty:
    st.table(matchup_history.rename(columns={target_stat: 'Yards', 'total_scrimmage_tds': 'TDs'}))
else:
    st.info(f"No historical data for {selected_player} vs {selected_opp}.")

# --- THE GRAPHS (REINSTATED) ---
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(px.line(player_subset, x='week', y=[target_stat, 'yards_roll3' if player_pos != 'QB' else 'pass_roll3'], title="Trend Analysis"), use_container_width=True)
with g2:
    # THE TD GRAPH YOU WANTED BACK
    st.plotly_chart(px.scatter(player_subset, x='total_scrimmage_yards', y='total_scrimmage_tds', trendline="ols", title="TD Efficiency Map"), use_container_width=True)

# --- PARLAY BUILDER ---
st.divider()
st.header("🎟️ Parlay Builder")
parlay_players = st.multiselect("Add Scorers", player_list, default=[selected_player])
if parlay_players:
    probs = []
    for p in parlay_players:
        p_pos = data[data['player_name'] == p]['position'].iloc[-1]
        stat = 'passing_tds' if p_pos == 'QB' else 'total_scrimmage_tds'
        exp = get_advanced_prediction(data, p, stat, curr_temp, curr_wind, is_grass_val, selected_opp)
        probs.append(1 - poisson.pmf(0, exp))
    
    total_prob = np.prod(probs) * 100
    st.metric("Parlay Win Probability", f"{total_prob:.2f}%")
    st.progress(min(total_prob/100, 1.0))
