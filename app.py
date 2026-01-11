import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (THE HARD RESET) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE FIX: MANUALLY REBUILD COLUMN LIST ---
        # We loop through columns and force them into single strings.
        # This is the only way to guarantee a 'Series' instead of a 'DataFrame'
        for df in [w_raw, s_raw]:
            new_column_names = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # If it's ('player', 'player_name'), we take 'player_name'
                    new_column_names.append(str(col[-1]))
                else:
                    new_column_names.append(str(col))
            df.columns = new_column_names  # Assign the flat list of strings back

        # --- MAPPING & CLEANING ---
        # Map 2026 data names back to your dashboard's logic
        col_map = {
            'player_display_name': 'player_name',
            'team_abbr': 'recent_team'
        }
        w_raw = w_raw.rename(columns=col_map)

        # IMPORTANT: Select ONLY the column to ensure it's a Series
        # This will now work because 'player_name' is a single string key
        if 'player_name' in w_raw.columns:
            # We use .copy() to ensure we aren't working on a slice
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Merge logic
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (UNTOUCHED FEATURES) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty and 'player_name' in data.columns:
        player_list = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", player_list)
        
        st.divider()
        if st.button("Add to Parlay"):
            st.session_state.parlay_legs.append(selected_player)
            st.success(f"Added {selected_player}")
            
        if st.session_state.parlay_legs:
            st.subheader("Current Slip")
            for leg in st.session_state.parlay_legs:
                st.write(f"✅ {leg}")
            if st.button("Clear Slip"):
                st.session_state.parlay_legs = []
                st.rerun()

# --- 4. MAIN DASHBOARD ---
if not data.empty:
    # Double check 'selected_player' exists to prevent UI crash
    p_data = data[data['player_name'] == selected_player]
    
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", latest.get('spread_line', 'N/A'))

        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Performance Trend"), use_container_width=True)
    else:
        st.warning("No data found for this player.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
