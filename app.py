import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Position-Smart Predictor", layout="wide")
st.title("🏈 NFL Sharp: Position-Specific Analytics")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 1. Column Maintenance
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    weekly = weekly.dropna(subset=['player_name', 'position'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    # 2. Advanced Metrics
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    
    # 3. Rolling Averages
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'target_share', 'passing_tds', 'total_scrimmage_tds']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 4. Merge Schedule
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
    df = weekly.merge(sched[sched_cols], left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
    
    df['wind'] = df['wind'].fillna(0)
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- INPUTS ---
selected_player = st.selectbox("Select Player", sorted(data['player_name'].unique()))
selected_opp = st.selectbox("Select Opponent", sorted(data['opponent_team'].unique()))
vegas_line = st.sidebar.number_input(f"Enter Sportsbook Line", value=200.0 if "QB" in data[data['player_name']==selected_player]['position'].values else 50.0)

# --- PREDICTION ENGINE ---
def get_prediction(df, player_name, target_stat):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    X = p_data[['temp', 'wind', 'target_share_roll3']].fillna(0)
    model = XGBRegressor(n_estimators=45).fit(X, p_data[target_stat])
    input_data = pd.DataFrame([[70, 5, p_data['target_share_roll3'].iloc[-1]]], columns=['temp', 'wind', 'target_share_roll3'])
    return max(0, model.predict(input_data)[0]), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- DYNAMIC DASHBOARD ---
player_pos = data[data['player_name'] == selected_player]['position'].iloc[-1]
st.write(f"**Position Detected:** {player_pos}")

p_yds, p_med, p_roll = get_prediction(data, selected_player, 'passing_yards')
s_yds, s_med, s_roll = get_prediction(data, selected_player, 'total_scrimmage_yards')

c1, c2, c3 = st.columns(3)

with c1:
    # RECOMENDATION LOGIC BY POSITION
    if player_pos == 'QB':
        rec_val = int(p_yds * 0.85 / 5) * 5
        st.success(f"✅ RECOMMENDED LEG\n{rec_val}+ Passing Yards")
        st.caption(f"Safety: 85% of Proj ({p_yds:.1f})")
    else:
        rec_val = int(s_yds * 0.8 / 5) * 5
        st.success(f"✅ RECOMMENDED LEG\n{rec_val}+ Scrimmage Yards")
        st.caption(f"Safety: 80% of Proj ({s_yds:.1f})")

with c2:
    # FADE LOGIC BY POSITION
    main_roll = p_roll if player_pos == 'QB' else s_roll
    main_med = p_med if player_pos == 'QB' else s_med
    if main_roll > main_med * 1.5:
        st.error(f"⚠️ FADE ALERT\nAvoid 'Over' on {selected_player}")
        st.caption("Reason: Extreme 3-game yardage spike (Regression imminent)")
    else:
        st.warning("⚖️ NEUTRAL\nNo strong Fade signals.")

with c3:
    # VALUE CALCULATOR
    compare_val = p_yds if player_pos == 'QB' else s_yds
    edge = compare_val - vegas_line
    edge_pct = (edge / vegas_line) * 100
    st.metric("Vegas Line Edge", f"{edge_pct:.1f}%", delta=f"{edge:.1f} yds", delta_color="normal" if edge > 0 else "inverse")
    st.caption(f"Model: {compare_val:.1f} vs Line: {vegas_line}")

# TD Probability
st.divider()
td_stat = 'passing_tds' if player_pos == 'QB' else 'total_scrimmage_tds'
td_exp, _, _ = get_prediction(data, selected_player, td_stat)
prob = (1 - poisson.pmf(0, td_exp)) * 100
st.subheader(f"{'Passing' if player_pos == 'QB' else 'Anytime'} TD Probability")
st.progress(min(prob/100, 1.0))
st.write(f"Estimated Probability: **{prob:.1f}%**")
