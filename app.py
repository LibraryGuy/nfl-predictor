 import streamlit as st

from st_paywall import add_auth

import nflreadpy as nfl

import pandas as pd

from xgboost import XGBRegressor

import plotly.express as px

import numpy as np

import sklearn


# --- 1. CONFIG & SESSION STATE ---

st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:

    st.session_state.parlay_legs = []


# --- 2. MOBILE-FRIENDLY LOGIN GATE ---

# We move this out of the sidebar so mobile users see it immediately.

if not st.user.is_logged_in:

    st.title("🏈 NFL Sharp: Pro Predictor")

    st.markdown("### Wild Card Weekend")

    st.info("Log in with Google to access pro-tier analytics and bypass the paywall if whitelisted.")

    

    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)

    st.stop()


# --- 3. WHITELIST & PAYWALL ---

admin_whitelist = st.secrets.get("whitelist", [])

if st.user.email in admin_whitelist:

    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")

else:

    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")


# --- 4. DATA LOADING ---

@st.cache_data(ttl=3600, show_spinner="Fetching Latest NFL Stats...")

def load_nfl_data_pro():

    try:

        years = [2024, 2025]

        # Load data using nflverse/nflreadpy

        weekly = nfl.load_player_stats(seasons=years).to_pandas()

        sched = nfl.load_schedules(seasons=years).to_pandas()

        pbp = nfl.load_pbp(seasons=years).to_pandas() 

        

        # Standardize Names & Teams

        weekly['player_name'] = weekly['player_name'].str.strip()

        if 'recent_team' not in weekly.columns:

            team_col = 'team' if 'team' in weekly.columns else 'team_abbr'

            weekly = weekly.rename(columns={team_col: 'recent_team'})

        

        # Clean Nulls to fix Sorting Error

        weekly = weekly.dropna(subset=['player_name', 'position'])

        

        # Metric cleanup

        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']

        for m in metrics: 

            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)

        

        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']

        

        # Defense EPA & Weather

        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')

        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 

                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')

        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')

        

        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)

        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)

        

        return df

    except Exception: return pd.DataFrame()


data = load_nfl_data_pro()


# --- 5. PARLAY BUILDER ---

with st.sidebar:

    st.header("🎟️ Parlay Builder")

    if st.session_state.parlay_legs:

        for leg in st.session_state.parlay_legs:

            st.info(f"**{leg['Player']}**: {leg['Prop']}")

        if st.button("Clear All"):

            st.session_state.parlay_legs = []

            st.rerun()

    else:

        st.write("Add legs to build a parlay.")

    

    st.divider()

    st.header("🏟️ Game Environment")

    curr_wind = st.slider("Wind (MPH)", 0, 40, 5)

    curr_temp = st.slider("Temp (F)", 0, 100, 45)

    is_grass_val = 1 if st.radio("Field", ["Grass", "Turf"]) == "Grass" else 0


# --- 6. PLAYER SELECTION ---

player_list = sorted(data['player_name'].unique())

selected_player = st.selectbox("Search Player", player_list)

selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent_team'].dropna().unique()))


player_subset = data[data['player_name'] == selected_player]

player_pos = player_subset['position'].iloc[-1]

target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'


vegas_line = st.number_input(f"Sportsbook Line ({target})", value=225.5 if player_pos == 'QB' else 65.5)


# --- 7. PREDICTION ENGINE WITH FALLBACK ---

def get_safe_prediction(df, player_name, target_stat, temp, wind, is_grass, opp_team):

    # Train position-based model

    pos_data = df[df['position'] == player_pos].copy()

    features = ['temp', 'wind', 'is_grass', 'def_epa_allowed']

    model = XGBRegressor(n_estimators=45, max_depth=3).fit(pos_data[features].fillna(0), pos_data[target_stat])

    

    # Predict

    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()

    input_data = pd.DataFrame([[temp, wind, is_grass, opp_epa]], columns=features)

    raw_proj = model.predict(input_data)[0]

    

    # FIX FOR 7.2 / 0.0 ERROR:

    # If prediction is suspiciously low for a starter, use their season average

    season_avg = player_subset[target_stat].mean()

    if player_pos == 'QB' and raw_proj < 50: # QBs rarely throw for < 50 yds unless injured

        return season_avg

    if player_pos != 'QB' and raw_proj < 10: # WRs/RBs fallback

        return season_avg

    

    return raw_proj


proj = get_safe_prediction(data, selected_player, target, curr_temp, curr_wind, is_grass_val, selected_opp)

rec_yards = int((proj * (0.88 if player_pos == 'QB' else 0.82)) / 5) * 5


# Display

st.header(f"📊 {selected_player} Projections")

c1, c2, c3 = st.columns(3)

c1.metric("Model Projection", f"{proj:.1f} Yds")

c2.success(f"🎯 SHARP REC: {rec_yards}+ Yds")

edge = proj - vegas_line

c3.metric("Vegas Edge", f"{edge:.1f} yds", delta=f"{((edge)/vegas_line)*100:.1f}%")


if st.button(f"➕ Add {rec_yards}+ Yards to Parlay", use_container_width=True):

    leg = {"Player": selected_player, "Prop": f"{rec_yards}+ Yds"}

    if leg not in st.session_state.parlay_legs:

        st.session_state.parlay_legs.append(leg)

        st.rerun()


st.plotly_chart(px.line(player_subset, x='week', y=target, title="Yardage History"), use_container_width=True)


if st.sidebar.button("Log Out"):

    st.logout()

