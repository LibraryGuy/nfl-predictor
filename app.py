import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# Set page layout
st.set_page_config(page_title="NFL Predictor: TD Projections", layout="wide")
st.title("🏈 NFL Predictive Dashboard: Yards & TD Probability")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
    weekly = weekly.dropna(subset=['player_name'])
    
    # Initialize Metrics
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
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
    
    if len(p_data) < 3: return 0.0, 0.0
    
    X = p_data[['temp', 'is_grass', 'def_diff']].fillna(0)
    y = p_data[target_stat]
    model = XGBRegressor(n_estimators=40).fit(X, y)
    pred = model.predict(pd.DataFrame([[temp, is_grass, opp_val]], columns=['temp', 'is_grass', 'def_diff']))[0]
    return max(0, pred), p_data[target_stat].mean()

# --- CALCULATING TD PROBABILITY ---
def calc_td_prob(expected_tds):
    # Poisson probability of 1 or more TDs: 1 - P(0)
    prob = (1 - poisson.pmf(0, expected_tds)) * 100
    return min(99.9, max(0.1, prob))

# --- DISPLAY ---
p_pos = data[data['player_name'] == selected_player]['position'].iloc[0]
st.divider()

col_p, col_s = st.columns(2)

with col_p:
    st.subheader("🎯 Passing & Scoring")
    p_yds, p_avg = get_advanced_prediction(data, selected_player, 'passing_yards', selected_opp, curr_temp, is_grass_val)
    p_td, _ = get_advanced_prediction(data, selected_player, 'passing_tds', selected_opp, curr_temp, is_grass_val)
    
    st.metric("Proj. Passing Yards", f"{p_yds:.1f}")
    st.metric("Passing TD Probability", f"{calc_td_prob(p_td):.1f}%", help="Chance of throwing 1+ TDs")

with col_s:
    st.subheader("🏃 Scrimmage & Scoring")
    s_yds, s_avg = get_advanced_prediction(data, selected_player, 'total_scrimmage_yards', selected_opp, curr_temp, is_grass_val)
    s_td, _ = get_advanced_prediction(data, selected_player, 'total_scrimmage_tds', selected_opp, curr_temp, is_grass_val)
    
    st.metric("Proj. Scrimmage Yards", f"{s_yds:.1f}")
    st.metric("Anytime TD Probability", f"{calc_td_prob(s_td):.1f}%", help="Chance of scoring 1+ Rush/Rec TDs")

# --- BETTING ADVICE ---
st.divider()
st.subheader("🤑 Recommended Betting Slip")
rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    main_yds = p_yds if p_pos == 'QB' else s_yds
    label = "Passing" if p_pos == 'QB' else "Scrimmage"
    st.info(f"**Yards Leg:** {selected_player} OVER {int(main_yds * 0.88)}.5 {label} Yards")

with rec_col2:
    atd_prob = calc_td_prob(s_td)
    if atd_prob > 45:
        st.success(f"**TD Leg:** {selected_player} Anytime TD (High Confidence: {atd_prob:.1f}%)")
    else:
        st.warning(f"**TD Leg:** Avoid ATD (Low Confidence: {atd_prob:.1f}%)")

st.plotly_chart(px.bar(data[data['player_name'] == selected_player], x='week', y=['rushing_tds', 'receiving_tds', 'passing_tds'], title="Historical Scoring Log"))
