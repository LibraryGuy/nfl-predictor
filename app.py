import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. MOBILE-FRIENDLY LOGIN ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.info("Log in with Google to access Wild Card Weekend projections.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. WHITELIST & PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email in admin_whitelist:
    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")
else:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. DATA LOADING (FLATTENED) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        weekly = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        pbp = nfl.load_pbp(seasons=[2024, 2025]).to_pandas()
        
        # FIX: Flatten MultiIndex columns to prevent .str errors
        for df_obj in [weekly, sched, pbp]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                df_obj.columns = df_obj.columns.get_level_values(0)
            # Remove any potential duplicate columns from merges
            df_obj.columns = pd.Index([str(c) for c in df_obj.columns])
        
        # Standardize Names (Handles 2026 update)
        name_col = 'player_display_name' if 'player_display_name' in weekly.columns else 'player_name'
        weekly = weekly.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Clean Player Names (Guaranteed to be Series now)
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        weekly = weekly.dropna(subset=['player_name', 'position'])
        
        # Convert Yards
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            weekly[col] = pd.to_numeric(weekly[col], errors='coerce').fillna(0)
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']

        # Defensive EPA Calculation
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Final Merge
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data sync in progress... refresh in 10 seconds.")
    st.stop()

# --- 5. SIDEBAR TOOLS ---
with st.sidebar:
    st.header("🎟️ Ticket")
    if st.session_state.parlay_legs:
        for leg in st.session_state.parlay_legs:
            st.info(f"**{leg['Player']}**: {leg['Prop']}")
        if st.button("Clear All"):
            st.session_state.parlay_legs = []
            st.rerun()
    
    st.divider()
    curr_temp = st.slider("Game Temp", 0, 100, 35)
    curr_wind = st.slider("Wind MPH", 0, 40, 10)

# --- 6. PLAYER DASHBOARD ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent_team'].unique()))

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'

vegas_line = st.number_input(f"Sportsbook Line ({target})", value=225.5 if player_pos == 'QB' else 65.5)

# --- 7. XGBOOST PREDICTION ---
def get_prediction(df, p_name, t_stat, temp, wind, opp):
    pos_data = df[df['position'] == player_pos].copy()
    features = ['temp', 'wind', 'def_epa_allowed']
    
    # Train model
    model = XGBRegressor(n_estimators=50).fit(pos_data[features], pos_data[t_stat])
    
    # Predict
    opp_epa = df[df['opponent_team'] == opp]['def_epa_allowed'].mean()
    raw = model.predict(pd.DataFrame([[temp, wind, opp_epa]], columns=features))[0]
    
    # Fail-safe for 7.2 / 5.3 errors
    avg = player_subset[t_stat].mean()
    if raw < (avg * 0.4):
        return avg
    return raw

proj = get_prediction(data, selected_player, target, curr_temp, curr_wind, selected_opp)
rec = int((proj * 0.85) / 5) * 5
edge = proj - vegas_line

# Display Metrics
st.header(f"📈 {selected_player} Projections")
c1, c2, c3 = st.columns(3)
c1.metric("Model Proj", f"{proj:.1f} Yds")
c2.success(f"🎯 RECOMMENDED: {rec}+ Yds")
c3.metric("Vegas Edge", f"{edge:.1f} Yds", delta=f"{(edge/vegas_line)*100:.1f}%")

if st.button(f"➕ Add {rec}+ Yards to Ticket", use_container_width=True):
    st.session_state.parlay_legs.append({"Player": selected_player, "Prop": f"{rec}+ Yds"})
    st.rerun()

# Plotly History
st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, title="Performance Trend"), use_container_width=True)
