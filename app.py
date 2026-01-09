import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# Set page layout
st.set_page_config(page_title="NFL Predictor: TD Pro", layout="wide")
st.title("🏈 NFL Predictive Dashboard: Heat-Check & TD Tracker")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
    weekly = weekly.dropna(subset=['player_name'])
    
    # Initialize all Metrics
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # --- ROLLING AVERAGE LOGIC ---
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    rolling_targets = ['passing_yards', 'total_scrimmage_yards', 'passing_tds', 'total_scrimmage_tds']
    for col in rolling_targets:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
    
    # Merge Schedule
    df = weekly.merge(
        sched[['season', 'week', 'home_team', 'temp', 'surface']], 
        left_on=['season', 'week', 'recent_team'], 
        right_on=['season', 'week', 'home_team'], 
        how='left'
    )
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- SIDEBAR & SELECTION ---
st.sidebar.header("Game Simulation")
curr_temp = st.sidebar.slider("Temperature", 0, 100, 65)
curr_surface = st.sidebar.radio("Field Type", ["Grass", "Turf"])
is_grass_val = 1 if curr_surface == "Grass" else 0

player_list = sorted(data['player_name'].dropna().unique())
selected_player = st.selectbox("Select Player", player_list, index=player_list.index("B.Young") if "B.Young" in player_list else 0)
selected_opp = st.selectbox("Select Opponent", sorted(data['opponent_team'].unique()))

# --- PREDICTION ENGINE ---
def get_advanced_prediction(df, player_name, target_stat, opponent, temp, is_grass):
    def_avg = data.groupby('opponent_team')[target_stat].mean().reset_index()
    def_avg.columns = ['opponent_team', 'def_diff']
    opp_val = def_avg[def_avg['opponent_team'] == opponent]['def_diff'].iloc[0]
    
    p_data = df[df['player_name'] == player_name].copy()
    p_data = p_data.merge(def_avg, on='opponent_team', how='left')
    
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    roll_col = f'{target_stat}_roll3'
    X = p_data[['temp', 'is_grass', 'def_diff', roll_col]].fillna(0)
    y = p_data[target_stat]
    
    model = XGBRegressor(n_estimators=40).fit(X, y)
    latest_roll = p_data[roll_col].iloc[-1]
    pred = model.predict(pd.DataFrame([[temp, is_grass, opp_val, latest_roll]], 
                                       columns=['temp', 'is_grass', 'def_diff', roll_col]))[0]
    return max(0, pred), p_data[target_stat].mean(), latest_roll

def calc_td_prob(expected_tds):
    prob = (1 - poisson.pmf(0, expected_tds)) * 100
    return min(99.9, max(0.1, prob))

# --- DASHBOARD LAYOUT ---
st.divider()
col_p, col_s = st.columns(2)
p_pos = data[data['player_name'] == selected_player]['position'].iloc[0]

with col_p:
    st.subheader("🎯 Passing Analytics")
    p_yds, p_avg, p_roll = get_advanced_prediction(data, selected_player, 'passing_yards', selected_opp, curr_temp, is_grass_val)
    p_td, _, _ = get_advanced_prediction(data, selected_player, 'passing_tds', selected_opp, curr_temp, is_grass_val)
    st.metric("Proj. Passing Yards", f"{p_yds:.1f}", delta=f"{p_yds - p_roll:.1f} vs Heat-Check")
    st.metric("Passing TD Probability", f"{calc_td_prob(p_td):.1f}%")

with col_s:
    st.subheader("🏃 Scrimmage Analytics")
    s_yds, s_avg, s_roll = get_advanced_prediction(data, selected_player, 'total_scrimmage_yards', selected_opp, curr_temp, is_grass_val)
    s_td, _, _ = get_advanced_prediction(data, selected_player, 'total_scrimmage_tds', selected_opp, curr_temp, is_grass_val)
    st.metric("Proj. Scrimmage Yards", f"{s_yds:.1f}", delta=f"{s_yds - s_roll:.1f} vs Heat-Check")
    st.metric("Anytime TD Probability", f"{calc_td_prob(s_td):.1f}%")

# --- HISTORICAL CHARTS ---
st.divider()
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(px.line(data[data['player_name'] == selected_player], 
                            x='week', y=['total_scrimmage_yards', 'total_scrimmage_yards_roll3'], 
                            title="Yardage Trend: Raw vs Heat-Check"), use_container_width=True)

with chart_col2:
    # Touchdown Stacked Bar Chart
    td_cols = ['passing_tds', 'rushing_tds', 'receiving_tds']
    st.plotly_chart(px.bar(data[data['player_name'] == selected_player], 
                           x='week', y=td_cols, 
                           title="Historical TD Log (Stacked by Type)",
                           labels={"value": "Touchdowns", "variable": "Type"},
                           color_discrete_map={
                               "passing_tds": "#1f77b4", 
                               "rushing_tds": "#2ca02c", 
                               "receiving_tds": "#ff7f0e"
                           }), use_container_width=True)
