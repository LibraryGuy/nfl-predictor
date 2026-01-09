import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# Set page layout
st.set_page_config(page_title="NFL Pro Predictor", layout="wide")
st.title("🏈 NFL Pro Analytics: Usage, Heat-Checks & Betting Legs")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    
    # 1. Load Data
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 2. Fix Team Column Name (KeyError Prevention)
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns:
            weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    # 3. Clean and Basic Feature Engineering
    weekly = weekly.dropna(subset=['player_name'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # Advanced Usage: Target Share & Proxy WOPR
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    weekly['wopr'] = weekly['target_share'] * 2.0  
    
    # 4. Rolling 3-game averages (Heat-Check logic)
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'passing_tds', 'total_scrimmage_tds', 'target_share']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 5. Merge with Schedule (Environmental Data)
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface']
    if 'wind' in sched.columns: sched_cols.append('wind')
    
    df = weekly.merge(sched[sched_cols], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], 
                      how='left')
    
    # 6. Final Clean-up
    df['wind'] = df['wind'].fillna(0) if 'wind' in df.columns else 0
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    
    return df

data = load_nfl_data_pro()

# --- SIDEBAR & SELECTION ---
st.sidebar.header("Pro Game Simulation")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Type", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list, index=0)
selected_opp = st.selectbox("Select Opponent", sorted(data['opponent_team'].unique()))

# --- PREDICTION ENGINE ---
def get_pro_prediction(df, player_name, target_stat, opponent, temp, wind, is_grass):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    X = p_data[['temp', 'wind', 'is_grass', 'target_share_roll3']].fillna(0)
    y = p_data[target_stat]
    
    model = XGBRegressor(n_estimators=45).fit(X, y)
    
    latest_roll = p_data[f'{target_stat}_roll3'].iloc[-1]
    curr_target_roll = p_data['target_share_roll3'].iloc[-1]
    
    pred = model.predict(pd.DataFrame([[temp, wind, is_grass, curr_target_roll]], 
                                       columns=['temp', 'wind', 'is_grass', 'target_share_roll3']))[0]
    
    return max(0, pred), p_data[target_stat].mean(), latest_roll

def calc_td_prob(expected_tds):
    prob = (1 - poisson.pmf(0, expected_tds)) * 100
    return min(99.9, max(0.1, prob))

# --- DASHBOARD LAYOUT ---
st.divider()
p_yds, p_avg, p_roll = get_pro_prediction(data, selected_player, 'passing_yards', selected_opp, curr_temp, curr_wind, is_grass_val)
s_yds, s_avg, s_roll = get_pro_prediction(data, selected_player, 'total_scrimmage_yards', selected_opp, curr_temp, curr_wind, is_grass_val)
s_td, _, _ = get_pro_prediction(data, selected_player, 'total_scrimmage_tds', selected_opp, curr_temp, curr_wind, is_grass_val)

col_p, col_s = st.columns(2)

with col_p:
    st.subheader("🎯 Passing Analytics")
    st.metric("Proj. Passing Yards", f"{p_yds:.1f}", delta=f"{p_yds - p_roll:.1f} vs L3 Avg")
    st.write(f"**Last 3 Games Avg:** {p_roll:.1f} yards")
    
    # BETTING LEG: Passing
    p_leg = int(p_yds * 0.85 / 5) * 5 # Conservative 85% floor rounded to nearest 5
    st.success(f"✅ Suggested Betting Leg: {p_leg}+ Passing Yards")

with col_s:
    st.subheader("🏃 Scrimmage Analytics")
    st.metric("Proj. Scrimmage Yards", f"{s_yds:.1f}", delta=f"{s_yds - s_roll:.1f} vs L3 Avg")
    st.write(f"**Last 3 Games Avg:** {s_roll:.1f} yards")
    
    # BETTING LEG: Scrimmage
    s_leg = int(s_yds * 0.80 / 5) * 5 # Conservative 80% floor rounded to nearest 5
    st.success(f"✅ Suggested Betting Leg: {s_leg}+ Scrimmage Yards")

# TD Probability Section
st.divider()
st.subheader(f"🏟️ {selected_player} Scoring Outlook vs {selected_opp}")
prob = calc_td_prob(s_td)
st.progress(prob / 100)
st.write(f"**Anytime TD Probability:** {prob:.1f}%")

# --- HISTORICAL CHARTS ---
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(data[data['player_name'] == selected_player], 
                            x='week', y=['total_scrimmage_yards', 'total_scrimmage_yards_roll3'], 
                            title="Yardage Trend: Raw vs Heat-Check"), use_container_width=True)
with c2:
    td_cols = ['passing_tds', 'rushing_tds', 'receiving_tds']
    st.plotly_chart(px.bar(data[data['player_name'] == selected_player], 
                           x='week', y=td_cols, 
                           title="Historical TD Log (Stacked)"), use_container_width=True)
