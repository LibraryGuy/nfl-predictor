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
    
    # 1. Load Datasets
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    pbp = nfl.load_pbp(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 2. Robust ID Mapping (Fixing the KeyError)
    # PBP uses specific IDs; we coalesce them into a single 'player_id' to match weekly stats
    pbp['player_id'] = pbp['receiver_player_id'].fillna(
        pbp['rusher_player_id']).fillna(pbp['passer_player_id'])
    
    # 3. Efficiency & High-Value Usage
    # Team Defense EPA (Opponent Difficulty)
    def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
    
    # Red Zone Touches (Plays inside the 20-yard line)
    rz_data = pbp[pbp['yardline_100'] <= 20].copy()
    rz_touches = rz_data.groupby(['season', 'week', 'player_id']).size().reset_index(name='rz_touches')
    
    # 4. Merge Advanced Metrics into Weekly Stats
    # 'player_id' is the standard key in load_player_stats()
    weekly = weekly.merge(rz_touches, on=['season', 'week', 'player_id'], how='left').fillna(0)
    
    # 5. Core Column Maintenance
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    weekly = weekly.dropna(subset=['player_name', 'position'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # 6. Heat-Check (Rolling Averages)
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'rz_touches', 'passing_tds', 'total_scrimmage_tds']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 7. Merge Environmental & Opponent Factors
    df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
    
    df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                  right_on=['season', 'week', 'defteam'], how='left')
    
    df[['wind', 'def_epa_allowed']] = df[['wind', 'def_epa_allowed']].fillna(0)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- SIDEBAR: GAME CONTROLS ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list)
player_pos = data[data['player_name'] == selected_player]['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Enter Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- PREDICTION ENGINE ---
def get_prediction(df, player_name, target_stat, temp, wind, is_grass):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    features = ['temp', 'wind', 'is_grass', 'rz_touches_roll3', 'def_epa_allowed']
    X = p_data[features].fillna(0)
    model = XGBRegressor(n_estimators=50).fit(X, p_data[target_stat])
    
    avg_def_epa = df[df['player_name']==player_name]['def_epa_allowed'].mean()
    input_df = pd.DataFrame([[temp, wind, is_grass, p_data['rz_touches_roll3'].iloc[-1], avg_def_epa]], 
                             columns=features)
    
    return max(0, model.predict(input_df)[0]), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- DASHBOARD: MAIN VIEW ---
st.divider()
p_yds, p_med, p_roll = get_prediction(data, selected_player, 'passing_yards', curr_temp, curr_wind, is_grass_val)
s_yds, s_med, s_roll = get_prediction(data, selected_player, 'total_scrimmage_yards', curr_temp, curr_wind, is_grass_val)

c1, c2, c3 = st.columns(3)
with c1:
    main_val = p_yds if player_pos == 'QB' else s_yds
    rec_val = int(main_val * 0.85 / 5) * 5
    st.success(f"🎯 RECOMMENDED LEG: {rec_val}+ {'Pass' if player_pos=='QB' else 'Scrim'} Yds")
    st.metric("Model Proj.", f"{main_val:.1f}")

with c2:
    if p_roll > p_med * 1.5 or s_roll > s_med * 1.5:
        st.error("⚠️ FADE ALERT: Major Regression Likely")
    else:
        st.warning("⚖️ NEUTRAL: Value is stable.")

with c3:
    edge = (p_yds if player_pos=='QB' else s_yds) - vegas_line
    st.metric("Vegas Line Edge", f"{(edge/vegas_line)*100:.1f}%", delta=f"{edge:.1f} yds")

# Visual Analytics
st.divider()
g1, g2 = st.columns(2)
player_data = data[data['player_name'] == selected_player]
with g1:
    st.plotly_chart(px.line(player_data, x='week', y=['rz_touches', 'rz_touches_roll3'], 
                            title="Red Zone Usage Trends"), use_container_width=True)
with g2:
    chart_stat = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
    st.plotly_chart(px.scatter(player_data, x='def_epa_allowed', y=chart_stat, 
                               trendline="ols", title="Efficiency vs Defense Strength"), use_container_width=True)

# --- PARLAY BUILDER ---
st.divider()
st.header("🎟️ Parlay Builder")
parlay_players = st.multiselect("Add Scorers to Parlay", player_list, default=[selected_player])

if parlay_players:
    probs = []
    ticket_rows = []
    for p in parlay_players:
        pos = data[data['player_name'] == p]['position'].iloc[-1]
        stat = 'passing_tds' if pos == 'QB' else 'total_scrimmage_tds'
        exp, _, _ = get_prediction(data, p, stat, curr_temp, curr_wind, is_grass_val)
        hit_prob = (1 - poisson.pmf(0, exp))
        probs.append(hit_prob)
        ticket_rows.append({"Player": p, "Market": "Passing TD" if pos == 'QB' else "Anytime TD", "Prob": f"{hit_prob*100:.1f}%"})
    
    total_prob = np.prod(probs) * 100
    st.table(pd.DataFrame(ticket_rows))
    st.metric("Ticket Hit Probability", f"{total_prob:.2f}%")
    st.progress(min(total_prob/100, 1.0))
