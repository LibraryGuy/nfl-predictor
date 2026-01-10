import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
import sklearn

# --- 1. SESSION STATE INITIALIZATION ---
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. NATIVE AUTHENTICATION GATE ---
if not st.user.is_logged_in:
    st.set_page_config(page_title="NFL Sharp - Login", page_icon="🏈")
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.info("Please log in with your Google account to access pro-tier analytics.")
    st.button("Log in with Google", on_click=st.login, type="primary")
    st.stop()

# --- 3. WHITELIST & PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email in admin_whitelist:
    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")
else:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. DATA LOADING ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")

@st.cache_data(show_spinner="Syncing NFL Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas() 
        if 'recent_team' not in weekly.columns:
            team_col = 'team' if 'team' in weekly.columns else 'team_abbr'
            weekly = weekly.rename(columns={team_col: 'recent_team'})
        weekly = weekly.dropna(subset=['player_name', 'position'])
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for m in metrics: 
            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        return df
    except Exception: return pd.DataFrame()

data = load_nfl_data_pro()

# --- 5. PARLAY LOGIC ---
def add_to_parlay(name, yards, pos):
    leg = {"Player": name, "Prop": f"{yards}+ Yds", "Position": pos}
    if leg not in st.session_state.parlay_legs:
        st.session_state.parlay_legs.append(leg)
        st.toast(f"✅ Added {name} to Parlay!")

def clear_parlay():
    st.session_state.parlay_legs = []

# --- 6. SIDEBAR & TOOLS ---
with st.sidebar:
    st.header("⚙️ Game Conditions")
    curr_wind = st.slider("Wind (MPH)", 0, 40, 5)
    curr_temp = st.slider("Temp (F)", 0, 100, 65)
    is_grass_val = 1 if st.radio("Field", ["Grass", "Turf"]) == "Grass" else 0
    
    st.divider()
    st.header("🎟️ Parlay Builder")
    if st.session_state.parlay_legs:
        for i, leg in enumerate(st.session_state.parlay_legs):
            st.info(f"**{leg['Player']}**: {leg['Prop']}")
        
        # Simple Odds Multiplier (Estimated -110 per leg)
        odds = 1.91 ** len(st.session_state.parlay_legs)
        st.write(f"**Estimated Payout:** {odds:.2f}x")
        
        if st.button("Clear All Legs"):
            clear_parlay()
            st.rerun()
    else:
        st.write("Add players to see parlay options.")

# --- 7. MAIN ENGINE ---
player_list = sorted(data['player_name'].dropna().unique())
selected_player = st.selectbox("Search Player", player_list)
selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent_team'].dropna().unique()))

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.number_input("Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

def get_prediction(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_data = df[df['position'] == pos].copy()
    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed']
    model = XGBRegressor(n_estimators=45, max_depth=3).fit(pos_data[features].fillna(0), pos_data[target_stat])
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    input_df = pd.DataFrame([[temp, wind, is_grass, opp_epa]], columns=features)
    return model.predict(input_df)[0]

target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_prediction(data, selected_player, target, curr_temp, curr_wind, is_grass_val, selected_opp)
rec_yards = int((proj * (0.88 if player_pos == 'QB' else 0.82)) / 5) * 5

# Display Metrics
st.header(f"📊 {selected_player} Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Model Projection", f"{proj:.1f} Yds")
c2.success(f"🎯 SHARP REC: {rec_yards}+ Yds")
edge = proj - vegas_line
c3.metric("Vegas Edge", f"{edge:.1f} yds", delta=f"{((edge)/vegas_line)*100:.1f}%")

# Parlay Button
if st.button(f"➕ Add {rec_yards}+ Yards to Parlay", type="primary"):
    add_to_parlay(selected_player, rec_yards, player_pos)

st.plotly_chart(px.line(player_subset, x='week', y=target, title="Performance Trend"), use_container_width=True)

if st.sidebar.button("Log Out"):
    st.logout()
