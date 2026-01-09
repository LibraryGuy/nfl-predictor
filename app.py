import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Sharp Pro Predictor", layout="wide")
st.title("🏈 NFL Sharp: Full Usage, Fades & Position Analytics")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    weekly = weekly.dropna(subset=['player_name', 'position'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    weekly['xtd'] = weekly['total_scrimmage_yards'] / 150
    
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'target_share', 'passing_tds', 'total_scrimmage_tds']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
    df = weekly.merge(sched[sched_cols], left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
    
    df['wind'] = df['wind'].fillna(0)
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Type", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list)
selected_opp = st.selectbox("Select Opponent", sorted(data['opponent_team'].unique()))

player_pos = data[data['player_name'] == selected_player]['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Enter Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- ENGINE ---
def get_prediction(df, player_name, target_stat, temp, wind, is_grass):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    X = p_data[['temp', 'wind', 'is_grass', 'target_share_roll3']].fillna(0)
    model = XGBRegressor(n_estimators=45).fit(X, p_data[target_stat])
    
    latest_usage = p_data['target_share_roll3'].iloc[-1]
    input_df = pd.DataFrame([[temp, wind, is_grass, latest_usage]], 
                             columns=['temp', 'wind', 'is_grass', 'target_share_roll3'])
    
    return max(0, model.predict(input_df)[0]), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- DASHBOARD ---
st.divider()
p_yds, p_med, p_roll = get_prediction(data, selected_player, 'passing_yards', curr_temp, curr_wind, is_grass_val)
s_yds, s_med, s_roll = get_prediction(data, selected_player, 'total_scrimmage_yards', curr_temp, curr_wind, is_grass_val)

# ROW 1: PREDICTIONS & BETTING
col1, col2, col3 = st.columns(3)
with col1:
    if player_pos == 'QB':
        rec_val = int(p_yds * 0.85 / 5) * 5
        st.success(f"🎯 RECOMMENDED LEG: {rec_val}+ Passing Yds")
        st.metric("Proj. Passing", f"{p_yds:.1f}", delta=f"{p_yds - p_roll:.1f} vs L3")
    else:
        rec_val = int(s_yds * 0.8 / 5) * 5
        st.success(f"🎯 RECOMMENDED LEG: {rec_val}+ Scrimmage Yds")
        st.metric("Proj. Scrimmage", f"{s_yds:.1f}", delta=f"{s_yds - s_roll:.1f} vs L3")

with col2:
    main_roll = p_roll if player_pos == 'QB' else s_roll
    main_med = p_med if player_pos == 'QB' else s_med
    if main_roll > main_med * 1.5 or (curr_wind > 18 and player_pos != 'RB'):
        st.error(f"⚠️ FADE ALERT: Avoid 'Over'")
        st.caption("Reason: High Variance/Wind Detected")
    else:
        st.warning("⚖️ NEUTRAL: No strong Fade signals.")

with col3:
    compare_val = p_yds if player_pos == 'QB' else s_yds
    edge = compare_val - vegas_line
    edge_pct = (edge / vegas_line) * 100
    st.metric("Vegas Line Edge", f"{edge_pct:.1f}%", delta=f"{edge:.1f} yds")
    st.caption(f"Vegas Line: {vegas_line}")

# ROW 2: TD PROBABILITY
st.divider()
td_stat = 'passing_tds' if player_pos == 'QB' else 'total_scrimmage_tds'
td_exp, _, _ = get_prediction(data, selected_player, td_stat, curr_temp, curr_wind, is_grass_val)
prob = (1 - poisson.pmf(0, td_exp)) * 100
st.subheader(f"{'Passing' if player_pos == 'QB' else 'Anytime'} TD Probability")
st.progress(min(prob/100, 1.0))
st.write(f"Model Probability: **{prob:.1f}%**")

# ROW 3: GRAPHS
st.divider()
g1, g2 = st.columns(2)
with g1:
    chart_stat = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
    st.plotly_chart(px.line(data[data['player_name'] == selected_player], 
                            x='week', y=[chart_stat, f'{chart_stat}_roll3'], 
                            title=f"{selected_player} Yardage Velocity"), use_container_width=True)
with g2:
    st.plotly_chart(px.scatter(data[data['player_name'] == selected_player], 
                               x='total_scrimmage_yards', y='total_scrimmage_tds', 
                               trendline="ols", title="Efficiency: TDs per Yard"), use_container_width=True)
