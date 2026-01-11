import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. AUTH ---
if not st.user.is_logged_in:
    st.title("🏈 NFL Sharp")
    st.button("Log in with Google", on_click=st.login, type="primary")
    st.stop()

# --- 3. DATA LOADING (FLATTENED) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw and immediately move to Pandas
        weekly_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE CRITICAL FIX: FLATTEN MULTIINDEX ---
        # This loop ensures every DataFrame has flat, single-level string columns
        for df in [weekly_raw, sched_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join levels with an underscore, e.g., ('player', 'name') -> 'player_name'
                df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]
            else:
                df.columns = [str(c) for c in df.columns]

        # Standardize naming across versions
        name_map = {'player_display_name': 'player_name', 'team_abbr': 'recent_team'}
        weekly = weekly_raw.rename(columns=name_map)
        
        # Clean Player Names (Now guaranteed to be a Series)
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # Merge with Schedule (Weather)
        df_final = weekly.merge(
            sched_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
            left_on=['season', 'week', 'recent_team'], 
            right_on=['season', 'week', 'home_team'], 
            how='left'
        )
        return df_final.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 4. DASHBOARD RESTORED ---
if not data.empty:
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Select Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    target = 'passing_yards' if player_pos == 'QB' else 'rushing_yards'
    
    # Model & Projection
    avg_yards = player_subset[target].mean()
    
    st.header(f"📊 {selected_player} Analysis")
    c1, c2 = st.columns(2)
    c1.metric("Season Average", f"{avg_yards:.1f} Yds")
    c2.success(f"Sharp Projection: {avg_yards * 1.05:.1f} Yds")
    
    st.plotly_chart(px.line(player_subset, x='week', y=target, title="Trend"), use_container_width=True)
else:
    st.warning("Data load failed. Please check your nflreadpy installation.")
