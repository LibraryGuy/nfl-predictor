import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. MOBILE-FRIENDLY LOGIN GATE ---
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

# --- 4. DATA LOADING (DEDUPLICATED & SANITIZED) ---
@st.cache_data(ttl=3600, show_spinner="Syncing Latest NFL Stats...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 
        
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]
        
        w_raw = w_raw.loc[:, ~w_raw.columns.duplicated()].copy()
        s_raw = s_raw.loc[:, ~s_raw.columns.duplicated()].copy()

        def find_and_rename(df, target, options):
            found = next((opt for opt in options if opt in df.columns), None)
            if found: return df.rename(columns={found: target})
            return df

        w_raw = find_and_rename(w_raw, 'player_name', ['player_name', 'player_display_name', 'player'])
        w_raw = find_and_rename(w_raw, 'recent_team', ['recent_team', 'team_abbr', 'team'])
        w_raw = find_and_rename(w_raw, 'pass_tds', ['passing_tds', 'pass_tds'])
        w_raw = find_and_rename(w_raw, 'rush_tds', ['rushing_tds', 'rush_tds'])
        w_raw = find_and_rename(w_raw, 'receiving_tds', ['receiving_tds', 'rec_tds'])

        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'pass_tds', 'rush_tds', 'receiving_tds']
        for m in metrics:
            if m in w_raw.columns:
                w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        w_raw['total_scrimmage_yards'] = w_raw['rushing_yards'] + w_raw['receiving_yards']
        w_raw['total_tds'] = w_raw['rush_tds'] + w_raw['receiving_tds']
        
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df.loc[:, ~df.columns.duplicated()].copy()
    except Exception as e: 
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame()

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
if not data.empty:
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent_team'].dropna().unique()))

    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    
    # Define targets and LABELS based on position
    if player_pos == 'QB':
        target_yds = 'passing_yards'
        target_tds = 'pass_tds'
        td_label = "Pass TD Prob"
        prop_suffix = "Pass TDs"
    else:
        target_yds = 'total_scrimmage_yards'
        target_tds = 'total_tds'
        td_label = "Anytime TD Prob"
        prop_suffix = "Anytime TD"

    vegas_line = st.number_input(f"Sportsbook Line ({target_yds})", value=225.5 if player_pos == 'QB' else 65.5)

    # --- 7. PREDICTION ENGINE ---
    def get_safe_prediction(df, p_subset, pos, target_stat, temp, wind, is_grass, opp_team):
        pos_data = df[df['position'] == pos].copy()
        features = ['temp', 'wind', 'is_grass', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=45, max_depth=3).fit(pos_data[features].fillna(0), pos_data[target_stat])
        
        opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
        input_data = pd.DataFrame([[temp, wind, is_grass, opp_epa]], columns=features)
        raw_proj = model.predict(input_data)[0]
        
        season_avg = p_subset[target_stat].mean()
        if raw_proj < (season_avg * 0.3): return season_avg
        return raw_proj

    proj_yds = get_safe_prediction(data, player_subset, player_pos, target_yds, curr_temp, curr_wind, is_grass_val, selected_opp)
    proj_tds = get_safe_prediction(data, player_subset, player_pos, target_tds, curr_temp, curr_wind, is_grass_val, selected_opp)
    
    rec_yards = int((proj_yds * (0.88 if player_pos == 'QB' else 0.82)) / 5) * 5
    # Probability scaling: TDs per game to % chance
    td_prob = min(max(proj_tds * 100, 5), 95) if player_pos != 'QB' else min(max(proj_tds * 40, 5), 95)

    # --- 8. DASHBOARD DISPLAY ---
    st.header(f"📊 {selected_player} ({player_pos}) Projections")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yardage Proj", f"{proj_yds:.1f} Yds")
    c2.metric(td_label, f"{td_prob:.1f}%")
    c3.success(f"🎯 SHARP REC: {rec_yards}+ Yds")
    edge = proj_yds - vegas_line
    c4.metric("Vegas Edge", f"{edge:.1f} yds", delta=f"{((edge)/vegas_line)*100:.1f}%")

    b1, b2 = st.columns(2)
    if b1.button(f"➕ Add {rec_yards}+ Yards to Parlay", use_container_width=True):
        leg = {"Player": selected_player, "Prop": f"{rec_yards}+ Yds"}
        if leg not in st.session_state.parlay_legs:
            st.session_state.parlay_legs.append(leg)
            st.rerun()
            
    # DYNAMIC TD BUTTON
    button_text = f"🔥 Add {prop_suffix} to Parlay"
    if b2.button(button_text, use_container_width=True):
        leg = {"Player": selected_player, "Prop": prop_suffix}
        if leg not in st.session_state.parlay_legs:
            st.session_state.parlay_legs.append(leg)
            st.rerun()

    st.plotly_chart(px.line(player_subset, x='week', y=target_yds, title=f"{target_yds.replace('_', ' ').title()} History"), use_container_width=True)

if st.sidebar.button("Log Out"):
    st.logout()
