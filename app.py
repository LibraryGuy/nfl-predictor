import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# --- 1. SESSION & CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. AUTHENTICATION ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.info("Log in with Google to access Wild Card Weekend projections.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email not in admin_whitelist:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. REPAIRED DATA LOADING ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load and convert to Pandas
        stats = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # FIX 1: Flatten MultiIndex (The .str Error Killer)
        # If columns are tuples like ('stats', 'player_name'), this takes just the name.
        if isinstance(stats.columns, pd.MultiIndex):
            stats.columns = stats.columns.get_level_values(-1)
        if isinstance(sched.columns, pd.MultiIndex):
            sched.columns = sched.columns.get_level_values(-1)
            
        # FIX 2: Standardize Column Names
        # Check for both possible name columns in 2026 data
        name_col = 'player_display_name' if 'player_display_name' in stats.columns else 'player_name'
        stats = stats.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})

        # FIX 3: Force Series for string operations
        # .iloc[:, 0] ensures we don't accidentally grab a DataFrame if columns were duplicated
        if isinstance(stats['player_name'], pd.DataFrame):
            stats['player_name'] = stats['player_name'].iloc[:, 0]
        
        stats['player_name'] = stats['player_name'].astype(str).str.strip()
        
        # Merge Weather Data
        df = stats.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                         left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        # Final Yardage Calculation
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['scrimmage_yds'] = df['rushing_yards'] + df['receiving_yards']
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data is refreshing. Please refresh the page in 10 seconds.")
    st.stop()

# --- 5. DASHBOARD UI ---
st.title("🏈 Sharp Pro Predictor")

player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)

# Get selected player info
player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'scrimmage_yds'

# Sidebar Parlay Builder
with st.sidebar:
    st.header("🎟️ Parlay Ticket")
    for leg in st.session_state.parlay_legs: st.write(leg)
    if st.button("Clear Ticket"):
        st.session_state.parlay_legs = []
        st.rerun()

# --- 6. METRICS & PLOT ---
avg_yds = player_subset[target].mean()
proj_yds = avg_yds * 1.08 # Model Adjustment

c1, c2, c3 = st.columns(3)
c1.metric("Season Average", f"{avg_yds:.1f}")
c2.success(f"🎯 SHARP REC: {int(proj_yds * 0.9)}+")
c3.metric("Model Proj", f"{proj_yds:.1f}")

if st.button(f"➕ Add {selected_player} to Ticket", use_container_width=True):
    st.session_state.parlay_legs.append(f"{selected_player}: {int(proj_yds * 0.9)}+ Yds")
    st.rerun()

st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, title=f"{selected_player} {target} History"), use_container_width=True)
