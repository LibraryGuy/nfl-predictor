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

# --- 2. MOBILE LOGIN ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.info("Log in with Google for Wild Card Weekend Projections.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. WHITELIST & PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email in admin_whitelist:
    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")
else:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. REPAIRED DATA LOADING ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # 1. Convert to Pandas immediately
        weekly = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # 2. CRITICAL FIX: Flatten MultiIndex columns
        # This prevents the 'DataFrame object has no attribute str' error
        if isinstance(weekly.columns, pd.MultiIndex):
            weekly.columns = weekly.columns.get_level_values(0)
        
        # 3. Handle 2026 Name Column Update
        if 'player_display_name' in weekly.columns and 'player_name' not in weekly.columns:
            weekly = weekly.rename(columns={'player_display_name': 'player_name'})
            
        # 4. Force player_name to be a clean String Series
        # We use .iloc[:, 0] if a duplicate column exists by accident
        if isinstance(weekly['player_name'], pd.DataFrame):
            weekly['player_name'] = weekly['player_name'].iloc[:, 0]
            
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # 5. Yardage Cleanup
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in weekly.columns:
                weekly[col] = pd.to_numeric(weekly[col], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        
        # 6. Weather Merge
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team' if 'recent_team' in weekly.columns else 'team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data sync in progress... refresh in 10 seconds.")
    st.stop()

# --- 5. UI DASHBOARD ---
# Now that data['player_name'] is a clean Series, this won't crash
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent_team'].unique()))

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'

# --- 6. SMART PREDICTION ---
def get_final_prediction(df, p_name, t_stat):
    pos_data = df[df['position'] == player_pos].copy()
    # Simple model for playoff stability
    model = XGBRegressor(n_estimators=40).fit(pos_data[['temp', 'wind']], pos_data[t_stat])
    raw = model.predict(pd.DataFrame([[45, 10]], columns=['temp', 'wind']))[0]
    
    # Jordan Love/Caleb Williams Fail-safe
    avg = player_subset[t_stat].mean()
    return avg if raw < (avg * 0.4) else raw

proj = get_final_prediction(data, selected_player, target)
rec = int((proj * 0.85) / 5) * 5

st.header(f"🏈 {selected_player}")
st.metric("Sharp Projection", f"{proj:.1f} Yds")
st.success(f"🎯 RECOMMENDED: {rec}+ Yds")

if st.button("➕ Add to Ticket", use_container_width=True):
    st.session_state.parlay_legs.append(f"{selected_player}: {rec}+ Yds")
    st.rerun()

with st.sidebar:
    st.header("🎟️ Ticket")
    for leg in st.session_state.parlay_legs: st.write(leg)
    if st.button("Clear"): st.session_state.parlay_legs = []; st.rerun()
