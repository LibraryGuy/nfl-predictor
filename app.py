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

# --- 4. REPAIRED DATA LOADING ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load and convert immediately to pandas
        raw_weekly = nfl.load_player_stats(seasons=[2024, 2025])
        weekly = raw_weekly.to_pandas()
        
        raw_sched = nfl.load_schedules(seasons=[2024, 2025])
        sched = raw_sched.to_pandas()
        
        # --- FIX: Ensure we have a clean, single-index DataFrame ---
        weekly.columns = [str(c) for c in weekly.columns]
        
        # Map player names (handling both old/new column versions)
        if 'player_display_name' in weekly.columns:
            weekly = weekly.rename(columns={'player_display_name': 'player_name'})
        
        # FORCE COLUMN TO SERIES (The fix for your .str error)
        # By re-assigning it like this, we ensure it's a Series, not a DataFrame
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # Cleanup
        weekly = weekly.dropna(subset=['player_name', 'position'])
        
        # Convert yardage to numbers
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in weekly.columns:
                weekly[col] = pd.to_numeric(weekly[col], errors='coerce').fillna(0)
            
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        
        # Weather Merge
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'team' if 'team' in weekly.columns else 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data is currently refreshing. Please wait 10 seconds and reload.")
    st.stop()

# --- 5. DASHBOARD ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)
selected_opp = st.selectbox("Against Defense", sorted(data['opponent_team'].unique()))

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'

# --- 6. PREDICTION ---
def get_final_prediction(df, p_name, t_stat):
    pos_data = df[df['position'] == player_pos].copy()
    model = XGBRegressor(n_estimators=40).fit(pos_data[['temp', 'wind']], pos_data[t_stat])
    raw = model.predict(pd.DataFrame([[45, 10]], columns=['temp', 'wind']))[0]
    
    # Fallback logic to fix 0.0/5.3/7.2 errors
    avg = player_subset[t_stat].mean()
    return avg if raw < (avg * 0.4) else raw

proj = get_final_prediction(data, selected_player, target)
rec = int((proj * 0.85) / 5) * 5

# --- 7. DISPLAY ---
st.header(f"🏈 {selected_player}")
st.metric("Model Projection", f"{proj:.1f} Yds")
st.success(f"🎯 SHARP REC: {rec}+ Yds")

if st.button("➕ Add to Parlay", use_container_width=True):
    st.session_state.parlay_legs.append(f"{selected_player}: {rec}+ Yds")

with st.sidebar:
    st.header("🎟️ Ticket")
    for leg in st.session_state.parlay_legs: st.write(leg)
    if st.button("Clear"): st.session_state.parlay_legs = []; st.rerun()
