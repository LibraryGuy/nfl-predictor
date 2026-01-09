import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# Set page layout
st.set_page_config(page_title="NFL Sharp Predictor", layout="wide")
st.title("🏈 NFL Sharp: Pro Usage, Fades & Line Value")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    
    # 1. Load Datasets
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 2. Key Maintenance (Ensures we have a team column to merge on)
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    # 3. Data Cleaning
    weekly = weekly.dropna(subset=['player_name'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # 4. Feature Engineering: Target Share & xTD (Expected TDs)
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    weekly['xtd'] = weekly['total_scrimmage_yards'] / 150 # Pro proxy for scoring expectations
    
    # 5. Rolling Averages (Heat-Check Logic)
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    roll_cols = ['passing_yards', 'total_scrimmage_yards', 'target_share', 'total_scrimmage_tds']
    for col in roll_cols:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 6. FIXED MERGE: Correcting the KeyError
    # Matches player's team to the schedule's home_team to get weather data
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
    df = weekly.merge(
        sched[sched_cols], 
        left_on=['season', 'week', 'recent_team'], 
        right_on=['season', 'week', 'home_team'], 
        how='left'
    )
    
    # 7. Final Formatting
    df['wind'] = df['wind'].fillna(0)
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    
    return df

data = load_nfl_data_pro()

# --- SIDEBAR: Game Environment ---
st.sidebar.header("Step 1: Game Environment")
curr_wind = st.sidebar.slider("Simulated Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Simulated Temp (F)", 0, 100, 65)

# --- MAIN SELECTION ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Step 2: Select Player", player_list)
selected_opp = st.selectbox("Step 3: Select Opponent", sorted(data['opponent_team'].unique()))

# Vegas Line Input
st.sidebar.divider()
st.sidebar.header("Step 4: Line Comparison")
vegas_line = st.sidebar.number_input(f"Enter Sportsbook Line for {selected_player}", value=50.0)

# --- ENGINE ---
def get_sharp_prediction(df, player_name, target_stat, temp, wind):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    # XGBoost Prediction using Temp, Wind, and Recent Usage
    X = p_data[['temp', 'wind', 'target_share_roll3']].fillna(0)
    model = XGBRegressor(n_estimators=45).fit(X, p_data[target_stat])
    
    latest_target_roll = p_data['target_share_roll3'].iloc[-1]
    input_data = pd.DataFrame([[temp, wind, latest_target_roll]], columns=['temp', 'wind', 'target_share_roll3'])
    
    pred = model.predict(input_data)[0]
    return max(0, pred), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- DASHBOARD ---
st.divider()
p_yds, p_med, p_roll = get_sharp_prediction(data, selected_player, 'passing_yards', curr_temp, curr_wind)
s_yds, s_med, s_roll = get_sharp_prediction(data, selected_player, 'total_scrimmage_yards', curr_temp, curr_wind)

# UI COLUMNS
c1, c2, c3 = st.columns(3)

with c1:
    st.success(f"✅ RECOMMENDED LEG\n{int(s_yds * 0.8 / 5) * 5}+ Scrimmage Yds")
    st.caption(f"Safety Floor: 80% of model {s_yds:.1f}")

with c2:
    # FADE LOGIC
    is_fade = False
    fade_reason = ""
    if s_roll > s_med * 1.5:
        is_fade, fade_reason = True, "Unsustainable Heat-Check (50%+ over median)"
    elif curr_wind > 18:
        is_fade, fade_reason = True, "High Wind Volatility"
    
    if is_fade:
        st.error(f"⚠️ FADE ALERT\nAvoid 'Over' on {selected_player}")
        st.caption(f"Reason: {fade_reason}")
    else:
        st.warning("⚖️ NEUTRAL\nNo strong Fade signals.")

with c3:
    # VALUE CALCULATOR
    edge = s_yds - vegas_line
    edge_pct = (edge / vegas_line) * 100
    color = "inverse" if edge < 0 else "normal"
    st.metric("Vegas Line Edge", f"{edge_pct:.1f}%", delta=f"{edge:.1f} yds", delta_color=color)
    st.caption(f"Model: {s_yds:.1f} | Vegas: {vegas_line}")

# TD Probability
st.divider()
s_td_exp, _, _ = get_sharp_prediction(data, selected_player, 'total_scrimmage_tds', curr_temp, curr_wind)
td_prob = (1 - poisson.pmf(0, s_td_exp)) * 100
st.subheader("Anytime TD Probability")
st.progress(min(td_prob/100, 1.0))
st.write(f"The model estimates a **{td_prob:.1f}%** chance of a score based on usage.")

# CHART
st.plotly_chart(px.line(data[data['player_name']==selected_player], x='week', y=['total_scrimmage_yards', 'total_scrimmage_yards_roll3'], 
                        title="Yardage Velocity (Actual vs 3-Game Heat Check)"), use_container_width=True)
