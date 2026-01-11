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
    st.info("Log in with Google to access pro-tier analytics.")
    st.button("Log in with Google", on_click=st.login, type="primary", use_container_width=True)
    st.stop()

# --- 3. PAYWALL ---
admin_whitelist = st.secrets.get("whitelist", [])
if st.user.email not in admin_whitelist:
    add_auth(required=True, subscription_button_text="Unlock Pro Insights", button_color="#FF4B4B")

# --- 4. REPAIRED DATA LOADING ---
@st.cache_data(ttl=3600, show_spinner="Syncing NFL Pro Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and convert to Pandas immediately
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        p_raw = nfl.load_pbp(seasons=years).to_pandas() 
        
        # --- THE FIX: FLATTEN MULTIINDEX ---
        # This collapses ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [w_raw, s_raw, p_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # We take the deepest level name which contains the actual stat
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean, single-level strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names (Handling 2026 data standard)
        name_col = 'player_display_name' if 'player_display_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Force Series for string operations (Stops the AttributeError)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        w_raw = w_raw.dropna(subset=['player_name', 'position'])
        
        # Repair Yardage: Force TOTAL yards
        for m in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        w_raw['total_scrimmage_yards'] = w_raw['rushing_yards'] + w_raw['receiving_yards']
        
        # Defense EPA & Merge
        def_epa = p_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 5. DASHBOARD UI ---
if not data.empty:
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    target = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
    
    # Prediction logic
    avg_yds = player_subset[target].mean()
    proj = avg_yds * 1.08
    
    st.header(f"📊 {selected_player} Projections")
    c1, c2 = st.columns(2)
    c1.metric("Season Avg", f"{avg_yds:.1f} Yds")
    c2.success(f"Sharp Prediction: {proj:.1f} Yds")
    
    st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True), use_container_width=True)
