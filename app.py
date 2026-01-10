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
        # Pulling 3 years for deep Matchup History
        years = [2023, 2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        
        # EPA is the gold standard for defensive strength
        # We'll pull 2024-25 PBP for high-fidelity defensive metrics
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas() 
        
        # Cleanup & Column Normalization
        if 'recent_team' not in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team', 'team_abbr': 'recent_team'})
            
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'targets']
        for m in metrics: 
            if m in weekly.columns:
                weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly.get('rushing_tds', 0) + weekly.get('receiving_tds', 0)
        
        # Advanced Feature Engineering: Defense EPA Allowed
        # EPA measures how much a defense actually stops scoring progress
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Averages (The "Velocity" of a player's form)
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())
        
        # Merge Everything
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        # Final Polish
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df # <--- CRITICAL FIX: Ensure DF is always returned here
    except Exception as e:
        st.error(f"⚠️ API Connection Error: {e}")
        return pd.DataFrame() # Return empty DF instead of None

# --- INITIALIZE DATA WITH SAFETY GUARD ---
data = load_advanced_data()

if data is None or data.empty:
    st.warning("The NFL Data Pipeline is currently syncing. Please wait 10 seconds and refresh.")
    st.stop() # Prevents the 'NoneType' error on the lines below

# --- SIDEBAR & SELECTIONS ---
st.sidebar.header("Matchup Environment")
curr_wind = st.sidebar.slider("Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Upcoming Opponent (Defense)", opp_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- ADVANCED PREDICTION ENGINE ---
def get_advanced_pred(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    # Global Position Training (Prevents Caleb Williams/Small Sample outliers)
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_df = df[df['position'] == pos].copy()
    
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed', roll_col]
    
    # XGBoost with high L2 Regularization (lambda=15) for stability
    model = XGBRegressor(n_estimators=50, max_depth=3, reg_lambda=15).fit(pos_df[features].fillna(0), pos_df[target_stat])
    
    # Matchup-specific inputs
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    
    input_data = pd.DataFrame([[temp, wind, is_grass, opp_epa, p_latest[roll_col]]], columns=features)
    return max(model.predict(input_data)[0], p_latest[roll_col] * 0.55)

# --- DASHBOARD ---
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_advanced_pred(data, selected_player, target, curr_temp, curr_wind, is_grass_val, selected_opp)

st.header(f"📊 {selected_player} vs {selected_opp}")
c1, c2, c3 = st.columns(3)
c1.metric("Model Projection", f"{proj:.1f} Yds")
c2.metric("Vegas Edge", f"{proj - vegas_line:.1f} yds", delta=f"{((proj-vegas_line)/vegas_line)*100:.1f}%")
c3.success(f"🎯 RECOMMEND: {int(proj*0.85/5)*5}+ Yards")

# --- MATCHUP HISTORY ---
st.subheader(f"🏟️ Historical Performance vs. {selected_opp}")
m_hist = player_subset[player_subset['opponent_team'] == selected_opp][['season', 'week', target, 'total_scrimmage_tds', 'surface']]
if not m_hist.empty:
    st.table(m_hist.rename(columns={target: 'Yards', 'total_scrimmage_tds': 'TDs', 'surface': 'Field'}))
else:
    st.info(f"No historical head-to-head data found for {selected_player} vs {selected_opp}.")

# --- CHARTS ---
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(px.line(player_subset, x='week', y=[target, 'yards_roll3' if player_pos != 'QB' else 'pass_roll3'], title="Volume Velocity"), use_container_width=True)
with g2:
    st.plotly_chart(px.scatter(player_subset, x='total_scrimmage_yards', y='total_scrimmage_tds', trendline="ols", title="TD Efficiency Map"), use_container_width=True)

# --- PARLAY BUILDER ---
st.header("🎟️ Parlay Optimizer")
parlay_picks = st.multiselect("Add Scorers", player_list, default=[selected_player])
if parlay_picks:
    probs = [1 - poisson.pmf(0, get_advanced_pred(data, p, 'passing_tds' if data[data['player_name']==p]['position'].iloc[-1]=='QB' else 'total_scrimmage_tds', curr_temp, curr_wind, is_grass_val, selected_opp)) for p in parlay_picks]
    win_prob = np.prod(probs) * 100
    st.metric("Parlay Hit Probability", f"{win_prob:.2f}%")
    st.progress(min(win_prob/100, 1.0))
