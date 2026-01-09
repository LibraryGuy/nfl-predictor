import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Pro Predictor", layout="wide")
st.title("🏈 NFL Pro Analytics: Usage, Efficiency & TD Probability")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    # Weekly stats for yards/TDs
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    # PBP/Seasonal for advanced Usage (Target Share, WOPR)
    seasonal = nfl.load_pfr_advstats(seasons=years, stat_type='rec').to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # Merge and Clean
    weekly = weekly.dropna(subset=['player_name'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # --- ADVANCED USAGE CALCS ---
    # target_share and wopr are often in seasonal/adv data
    # For simplicity in this dashboard, we map seasonal averages to our weekly data
    if 'target_share' in weekly.columns:
        weekly['wopr'] = 1.5 * weekly['target_share'] + 0.7 * weekly.get('air_yards_share', 0)
    else:
        # Fallback: calculate rough share if not present
        team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
        weekly['target_share'] = weekly['targets'] / team_tgts
        weekly['wopr'] = weekly['target_share'] * 2.0 # Proxy for WOPR
    
    # Rolling 3-game averages
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    for col in ['passing_yards', 'total_scrimmage_yards', 'target_share']:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # Environmental Data
    df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                      left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
    df['wind'] = df['wind'].fillna(0)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- SIDEBAR ---
st.sidebar.header("Pro Game Conditions")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list, index=player_list.index("B.Young") if "B.Young" in player_list else 0)
selected_opp = st.selectbox("Select Opponent", sorted(data['opponent_team'].unique()))

# --- ADVANCED PREDICTION ENGINE ---
def get_pro_prediction(df, player_name, target_stat, opponent, temp, wind, is_grass):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    # Adding Wind and Target Share to the model features
    X = p_data[['temp', 'wind', 'is_grass', 'target_share_roll3']].fillna(0)
    y = p_data[target_stat]
    
    model = XGBRegressor(n_estimators=50).fit(X, y)
    latest_feat = p_data[['temp', 'wind', 'is_grass', 'target_share_roll3']].iloc[-1:].copy()
    latest_feat['temp'], latest_feat['wind'], latest_feat['is_grass'] = temp, wind, is_grass
    
    pred = model.predict(latest_feat)[0]
    return max(0, pred), p_data[target_stat].mean(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- UI LAYOUT ---
st.divider()
p_yds, p_avg, p_roll = get_pro_prediction(data, selected_player, 'passing_yards', selected_opp, curr_temp, curr_wind, is_grass_val)
s_yds, s_avg, s_roll = get_pro_prediction(data, selected_player, 'total_scrimmage_yards', selected_opp, curr_temp, curr_wind, is_grass_val)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Proj. Passing", f"{p_yds:.1f}", delta=f"{p_yds - p_roll:.1f} vs L3")
    st.caption(f"Target Share (L3): {data[data['player_name']==selected_player]['target_share_roll3'].iloc[-1]:.1%}")
with col2:
    st.metric("Proj. Scrimmage", f"{s_yds:.1f}", delta=f"{s_yds - s_roll:.1f} vs L3")
    st.caption(f"WOPR: {data[data['player_name']==selected_player]['wopr'].iloc[-1]:.2f}")
with col3:
    s_td, _, _ = get_pro_prediction(data, selected_player, 'total_scrimmage_tds', selected_opp, curr_temp, curr_wind, is_grass_val)
    prob = (1 - poisson.pmf(0, s_td)) * 100
    st.metric("Anytime TD Prob", f"{prob:.1f}%")

# Charts
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(data[data['player_name']==selected_player], x='week', y='wopr', title="Usage Trend (WOPR)"), use_container_width=True)
with c2:
    st.plotly_chart(px.bar(data[data['player_name']==selected_player], x='week', y=['passing_tds', 'rushing_tds', 'receiving_tds'], title="Touchdown Log"), use_container_width=True)
