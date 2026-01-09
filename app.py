import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Sharp Predictor", layout="wide")
st.title("🏈 NFL Sharp: Pro Usage & Fade Alerts")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns: weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    weekly = weekly.dropna(subset=['player_name'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # Usage & xTD (Expected TD Proxy)
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    # xTD: Generally 1 TD per 150 yards for WR/RBs
    weekly['xtd'] = weekly['total_scrimmage_yards'] / 150
    
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    for col in ['passing_yards', 'total_scrimmage_yards', 'target_share', 'total_scrimmage_tds']:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
    df = weekly.merge(sched[sched_cols], on=['season', 'week', 'home_team'], how='left')
    df['wind'] = df['wind'].fillna(0)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    return df

data = load_nfl_data_pro()

# --- SIDEBAR ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp (F)", 0, 100, 65)

selected_player = st.selectbox("Player", sorted(data['player_name'].unique()))
selected_opp = st.selectbox("Opponent", sorted(data['opponent_team'].unique()))

def get_sharp_prediction(df, player_name, target_stat, opponent, temp, wind):
    p_data = df[df['player_name'] == player_name].copy()
    if len(p_data) < 3: return 0.0, 0.0, 0.0
    
    X = p_data[['temp', 'wind', 'target_share_roll3']].fillna(0)
    model = XGBRegressor(n_estimators=45).fit(X, p_data[target_stat])
    
    feat = pd.DataFrame([[temp, wind, p_data['target_share_roll3'].iloc[-1]]], 
                        columns=['temp', 'wind', 'target_share_roll3'])
    return max(0, model.predict(feat)[0]), p_data[target_stat].median(), p_data[f'{target_stat}_roll3'].iloc[-1]

# --- UI ---
st.divider()
p_yds, p_med, p_roll = get_sharp_prediction(data, selected_player, 'passing_yards', selected_opp, curr_temp, curr_wind)
s_yds, s_med, s_roll = get_sharp_prediction(data, selected_player, 'total_scrimmage_yards', selected_opp, curr_temp, curr_wind)

c1, c2 = st.columns(2)
with c1:
    st.success(f"🎯 RECOMMENDED LEG: {int(s_yds * 0.8 / 5) * 5}+ Scrimmage Yards")
    st.info(f"Basis: 80% of model projection ({s_yds:.1f})")

with c2:
    # FADE LOGIC
    is_fade = False
    fade_reason = ""
    if s_roll > s_med * 1.4:
        is_fade, fade_reason = True, "Unsustainable Heat-Check (40% above median)"
    elif curr_wind > 18:
        is_fade, fade_reason = True, "High Wind (Pass/Catch volatility)"
    
    if is_fade:
        st.error(f"⚠️ FADE ALERT: Avoid 'Over' on {selected_player}")
        st.caption(f"Reason: {fade_reason}")
    else:
        st.warning("⚖️ NEUTRAL: No significant fade signals detected.")

st.divider()
st.subheader("Historical vs Expected Scoring")
chart_df = data[data['player_name'] == selected_player]
st.plotly_chart(px.scatter(chart_df, x='total_scrimmage_yards', y='total_scrimmage_tds', 
                           trendline="ols", title="TD Regression: Actual vs Yardage-Based Expectations"))
