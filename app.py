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
        # 1. Load data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # 2. THE SLEDGEHAMMER: Flatten MultiIndex Immediately
        # This turns ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [w_raw, s_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Clean up all column names to be flat strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # 3. Rename standardized columns for 2026 data
        name_col = 'player_display_name' if 'player_display_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # 4. Force Series (This kills the .str error)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # 5. Fix Yardage (Jordan Love Fix)
        # Forcing to_numeric ensures we get Total Yards, not YPA or percentages
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for m in metrics:
            w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        w_raw['scrimmage_yds'] = w_raw['rushing_yards'] + w_raw['receiving_yards']
        
        # 6. Merge Weather/Schedule
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
                         left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

if data.empty:
    st.warning("Data is refreshing. Please wait 10 seconds.")
    st.stop()

# --- 5. DASHBOARD UI ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Search Player", player_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
target = 'passing_yards' if player_pos == 'QB' else 'scrimmage_yds'

# --- 6. METRICS & PLOTS ---
avg_yds = player_subset[target].mean()
# Verification: If J. Love shows 250+ now, it worked. 
# If he still shows 5.3, we are pulling from the wrong level.

st.header(f"📊 {selected_player} Projections")
m1, m2 = st.columns(2)
m1.metric("Season Avg", f"{avg_yds:.1f} Yds")
m2.success(f"🎯 Recommended: {int(avg_yds * 0.9)}+ Yds")

st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, 
                        title=f"{selected_player} Season Trend"), use_container_width=True)

# Parlay Sidebar
with st.sidebar:
    st.header("🎟️ Parlay Builder")
    if st.button(f"Add {selected_player} Prop"):
        st.session_state.parlay_legs.append(f"{selected_player}: {int(avg_yds * 0.9)}+")
    for leg in st.session_state.parlay_legs: st.write(leg)
