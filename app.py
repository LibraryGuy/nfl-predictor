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

# --- 4. DATA LOADING (REPAIRED) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- FIX: COLUMN NAME MAPPING ---
        # Map new nflverse names back to our code's expected names
        rename_dict = {
            'player_display_name': 'player_name',
            'team_abbr': 'recent_team'
        }
        weekly = weekly.rename(columns=rename_dict)
        
        # Ensure column exists before proceeding
        if 'player_name' not in weekly.columns:
            st.error("Critical Error: Player Name column not found in data feed.")
            return pd.DataFrame()

        # Clean strings
        weekly['player_name'] = weekly['player_name'].str.strip()
        weekly = weekly.dropna(subset=['player_name', 'position'])
        
        # Force numeric conversion
        target_cols = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for col in target_cols:
            if col in weekly.columns:
                weekly[col] = pd.to_numeric(weekly[col], errors='coerce').fillna(0)
            
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        
        # Merge Weather
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='left')
        return df.fillna(0)
    except Exception as e: 
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# Check if data is empty before rendering UI
if data.empty:
    st.warning("Data sync in progress... Please refresh in a moment.")
    st.stop()

# --- 5. PLAYER DASHBOARD ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player", player_list)
selected_opp = st.selectbox("Opponent", sorted(data['opponent_team'].unique()))

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'

# --- 6. SMART PREDICTION ENGINE ---
def get_sharp_prediction(df, player_name, target_stat):
    pos_data = df[df['position'] == player_pos].copy()
    features = ['temp', 'wind']
    model = XGBRegressor(n_estimators=50).fit(pos_data[features], pos_data[target_stat])
    
    # Predict with baseline conditions
    raw_proj = model.predict(pd.DataFrame([[45, 10]], columns=features))[0]
    
    # Season Average Fallback
    season_avg = player_subset[target_stat].mean()
    if raw_proj < (season_avg * 0.3): # Catch 5.3/7.2 errors
        return season_avg
    return raw_proj

proj = get_sharp_prediction(data, selected_player, target)
rec_yards = int((proj * 0.85) / 5) * 5

# --- 7. UI DISPLAY ---
st.header(f"📈 {selected_player} ({player_pos})")
c1, c2 = st.columns(2)
c1.metric("Sharp Projection", f"{proj:.1f} Yds")
c2.success(f"🎯 RECOMMENDED: {rec_yards}+ Yds")

if st.button(f"➕ Add {rec_yards}+ Yds to Parlay", use_container_width=True):
    st.session_state.parlay_legs.append({"Player": selected_player, "Prop": f"{rec_yards}+ Yds"})
    st.toast("Leg Added!")

# Parlay Sidebar
with st.sidebar:
    st.header("🎟️ Your Parlay")
    for leg in st.session_state.parlay_legs:
        st.write(f"✅ {leg['Player']}: {leg['Prop']}")
    if st.button("Clear"): 
        st.session_state.parlay_legs = []
        st.rerun()

st.plotly_chart(px.line(player_subset, x='week', y=target, title="Season Trend"))
