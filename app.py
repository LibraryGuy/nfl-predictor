import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Sharp: Ultimate Prediction Engine", layout="wide")
st.title("🏈 NFL Sharp: Ultimate Prediction Engine")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    
    # 1. Load Data
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    pbp = nfl.load_pbp(seasons=years).to_pandas() # Play-by-Play for EPA & RZ
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 2. Extract Red Zone & EPA from PBP
    # Filtering for Red Zone (yardline_100 <= 20) and high-value touches
    rz_data = pbp[pbp['yardline_100'] <= 20].copy()
    rz_touches = rz_data.groupby(['season', 'week', 'fantasy_player_id']).size().reset_index(name='rz_touches')
    
    # Team EPA Averages (Opponent Difficulty)
    team_epa = pbp.groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index(name='off_epa')
    def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
    
    # 3. Join Advanced Metrics to Weekly Stats
    weekly = weekly.merge(rz_touches, on=['season', 'week', 'fantasy_player_id'], how='left').fillna(0)
    
    # 4. Standard Column Maintenance
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    weekly = weekly.dropna(subset=['player_name', 'position'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # Rolling Features (Heat-Check)
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'rz_touches', 'passing_tds', 'total_scrimmage_tds']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 5. Merge Schedules & Def EPA (Opponent Adjustment)
    df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
    
    df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                  right_on=['season', 'week', 'defteam'], how='left')
    
    df[['wind', 'def_epa_allowed']] = df[['wind', 'def_epa_allowed']].fillna(0)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    
    return df

data = load_nfl_data_pro()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Player Select", player_list)
player_pos = data[data['player_name'] == selected_player]['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Vegas Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- PREDICTION ENGINE (Advanced) ---
def get_prediction(df, player_name, target_stat, temp, wind, is_grass):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    # Features: Usage + Environment + Opponent Strength (Def EPA)
    features = ['temp', 'wind', 'is_grass', 'rz_touches_roll3', 'def_epa_allowed']
    X = p_data[features].fillna(0)
    model = XGBRegressor(n_estimators=50).fit(X, p_data[target_stat])
    
    # Scenario Input
    avg_def_epa = df[df['player_name']==player_name]['def_epa_allowed'].mean()
    input_df = pd.DataFrame([[temp, wind, is_grass, p_data['rz_touches_roll3'].iloc[-1], avg_def_epa]], 
                             columns=features)
    
    return max(0, model.predict(input_df)[0]), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- DASHBOARD ---
p_yds, p_med, p_roll = get_prediction(data, selected_player, 'passing_yards', curr_temp, curr_wind, is_grass_val)
s_yds, s_med, s_roll = get_prediction(data, selected_player, 'total_scrimmage_yards', curr_temp, curr_wind, is_grass_val)

col1, col2, col3 = st.columns(3)
with col1:
    rec = int((p_yds if player_pos == 'QB' else s_yds) * 0.8 / 5) * 5
    st.success(f"🎯 RECOMMENDED LEG: {rec}+ {'Pass' if player_pos=='QB' else 'Scrim'} Yds")
    st.metric("Model Proj.", f"{p_yds if player_pos=='QB' else s_yds:.1f}")

with col2:
    if p_roll > p_med * 1.5 or s_roll > s_med * 1.5:
        st.error("⚠️ FADE ALERT: Regression Candidate")
    else: st.warning("⚖️ NEUTRAL: No strong Fade signals.")

with col3:
    edge = (p_yds if player_pos=='QB' else s_yds) - vegas_line
    st.metric("Vegas Edge", f"{(edge/vegas_line)*100:.1f}%", delta=f"{edge:.1f} yds")

# Efficiency & Velocity Graphs
st.divider()
g1, g2 = st.columns(2)
player_data = data[data['player_name'] == selected_player]
with g1:
    st.plotly_chart(px.line(player_data, x='week', y=['rz_touches', 'rz_touches_roll3'], title="Red Zone Usage Velocity"), use_container_width=True)
with g2:
    st.plotly_chart(px.scatter(player_data, x='def_epa_allowed', y='total_scrimmage_yards' if player_pos!='QB' else 'passing_yards', 
                               trendline="ols", title="Performance vs. Defense Quality"), use_container_width=True)

# PARLAY BUILDER (Preserved)
st.divider()
st.header("🎟️ Parlay Builder")
parlay_players = st.multiselect("Add Scorers to Ticket", player_list, default=[selected_player])
if parlay_players:
    probs = []
    for p in parlay_players:
        pos = data[data['player_name'] == p]['position'].iloc[-1]
        exp, _, _ = get_prediction(data, p, 'passing_tds' if pos=='QB' else 'total_scrimmage_tds', curr_temp, curr_wind, is_grass_val)
        probs.append(1 - poisson.pmf(0, exp))
    st.metric("Combined Hit Probability", f"{np.prod(probs)*100:.2f}%")
