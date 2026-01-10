import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# --- 1. NATIVE AUTHENTICATION GATE ---
# This uses Streamlit's built-in OIDC system. 
# It checks if the user is logged in via Google before doing anything else.
if not st.user.get("is_logged_in"):
    st.set_page_config(page_title="NFL Sharp - Login", page_icon="🏈")
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.markdown("### Secure Member Login")
    st.info("Please log in with your Google account to access the Sharp Analytics Suite.")
    
    # Using the native login system prevents the 'AuthError' crash
    if st.button("Log in with Google", type="primary"):
        st.login()
    st.stop()

# --- 2. SUBSCRIPTION PAYWALL ---
# Once authenticated, this checks Stripe for an active subscription.
add_auth(
    required=True,
    subscription_button_text="Unlock Pro Insights",
    button_color="#FF4B4B"
)

# --- 3. DASHBOARD CONFIGURATION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
st.title(f"🏈 NFL Sharp Pro: Welcome {st.user.name}")

@st.cache_data(show_spinner="Syncing NFL Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas() 
        
        # Data Cleaning
        if 'recent_team' not in weekly.columns:
            team_col = 'team' if 'team' in weekly.columns else 'team_abbr'
            weekly = weekly.rename(columns={team_col: 'recent_team'})
        
        weekly = weekly.dropna(subset=['player_name', 'position'])
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for m in metrics: 
            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
        
        # Defense Stats (EPA)
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Metrics
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())

        # Final Dataset Merge
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception:
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.error("⚠️ Data connection lost. Please refresh.")
    st.stop()

# --- SIDEBAR & SELECTION ---
st.sidebar.header("Game Settings")
curr_wind = st.sidebar.slider("Wind (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temp (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Surface", ["Grass", "Turf"]) == "Grass" else 0

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Upcoming Matchup", opp_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- PREDICTION ENGINE ---
def get_stable_prediction(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_data = df[df['position'] == pos].copy()
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed', roll_col]
    
    model = XGBRegressor(n_estimators=45, max_depth=3, reg_lambda=10).fit(pos_data[features].fillna(0), pos_data[target_stat])
    
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    
    input_df = pd.DataFrame([[temp, wind, is_grass, opp_epa, p_latest[roll_col]]], columns=features)
    return max(model.predict(input_df)[0], p_latest[roll_col] * 0.6)

# --- MAIN DASHBOARD VIEW ---
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_stable_prediction(data, selected_player, target, curr_temp, curr_wind, is_grass_val, selected_opp)

st.header(f"📊 {selected_player} Projections")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Model Projection", f"{proj:.1f} Yds")
with c2: st.success(f"🎯 RECOMMENDED: {int(proj*0.85/5)*5}+ Yards")
with c3: 
    edge = proj - vegas_line
    st.metric("Vegas Edge", f"{edge:.1f} yds", delta=f"{((edge)/vegas_line)*100:.1f}%")

# Trends & History
g1, g2 = st.columns(2)
with g1: st.plotly_chart(px.line(player_subset, x='week', y=[target, 'yards_roll3' if player_pos != 'QB' else 'pass_roll3'], title="Yardage Velocity"), use_container_width=True)
with g2: 
    st.subheader(f"🏟️ History vs {selected_opp}")
    match_hist = player_subset[player_subset['opponent_team'] == selected_opp].copy()
    if not match_hist.empty:
        st.dataframe(match_hist[['season', 'week', target, 'total_scrimmage_tds']], hide_index=True)
    else:
        st.info("No prior career games found against this opponent.")

st.divider()
if st.button("Logout"):
    st.logout()
