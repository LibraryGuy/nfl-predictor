import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. AUTHENTICATION ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp: Pro Predictor")
    st.info("Log in with Google to access Wild Card Weekend projections.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. PAYWALL & WHITELIST ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email in admin_whitelist:
    st.sidebar.success(f"🌟 VIP Access: {st.user.email}")
else:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. REPAIRED DATA LOADING ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw and immediately move to Pandas
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- CRITICAL FIX: FLATTEN MULTIINDEX ---
        # This prevents the '.str' error by making columns flat strings
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip() for c in df.columns]

        # Rename standard columns for 2026 data
        name_col = 'player_display_name' if 'player_display_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Force Clean Series (This is where the error lived)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Merge with Schedule for Weather/EPA
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
                         left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        # Fill zero for numeric stats
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['total_scrimmage_yards'] = df['rushing_yards'] + df['receiving_yards']
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data is currently refreshing. Please wait 10 seconds.")
    st.stop()

# --- 5. DASHBOARD UI ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'

# Vegas Line Input
v_line = st.number_input(f"Sportsbook Line ({target})", value=225.5 if player_pos == 'QB' else 65.5)

# --- 6. XGBOOST PREDICTION ---
def get_prediction(df, p_name, t_stat):
    pos_data = df[df['position'] == player_pos].copy()
    model = XGBRegressor(n_estimators=40).fit(pos_data[['temp', 'wind']], pos_data[t_stat])
    raw = model.predict(pd.DataFrame([[45, 10]], columns=['temp', 'wind']))[0]
    
    # Floor Protection
    avg = player_subset[t_stat].mean()
    return avg if raw < (avg * 0.4) else raw

proj = get_prediction(data, selected_player, target)
rec = int((proj * 0.85) / 5) * 5
edge = proj - v_line

# --- 7. DISPLAY ---
st.header(f"📊 {selected_player} Projections")
m1, m2, m3 = st.columns(3)
m1.metric("Model Proj", f"{proj:.1f} Yds")
m2.success(f"🎯 RECOMMENDED: {rec}+ Yds")
m3.metric("Vegas Edge", f"{edge:.1f} Yds", delta=f"{(edge/v_line)*100:.1f}%")

if st.button("➕ Add to Parlay Ticket", use_container_width=True):
    st.session_state.parlay_legs.append(f"{selected_player}: {rec}+ Yds")

with st.sidebar:
    st.header("🎟️ Ticket")
    for leg in st.session_state.parlay_legs: st.write(leg)
    if st.button("Clear Ticket"): 
        st.session_state.parlay_legs = []
        st.rerun()

st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, title="Season Trend"), use_container_width=True)
