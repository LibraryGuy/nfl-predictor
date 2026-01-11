import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE FIX: FORCED FLATTENING ---
        # This collapses [('offense', 'passing_yards')] into 'passing_yards'
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # We drop the top level (e.g., 'offense', 'player') to keep original names
                df.columns = df.columns.get_level_values(-1)
            # Remove any unintended whitespace or objects
            df.columns = [str(c).strip() for c in df.columns]

        # --- RE-MAPPING TO ORIGINAL KEYS ---
        # Mapping common 2026 header changes back to your variables
        mapping = {'player_display_name': 'player_name', 'team_abbr': 'recent_team'}
        w_raw = w_raw.rename(columns=mapping)

        # Now 'player_name' is a single column. String methods WILL work.
        if 'player_name' in w_raw.columns:
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Ensure numeric types (Fixes the 5.3 yard average bug)
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Merge with Schedule
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        # If this fails, it will show you exactly which column is missing
        st.error(f"Critical Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. UI (ORIGINAL VIEW) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty and 'player_name' in data.columns:
        player_list = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", player_list)
        
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
    p_data = data[data['player_name'] == selected_player]
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics")
        
        # Row 1
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", latest.get('spread_line', 'N/A'))

        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Yardage Trend"), use_container_width=True)
    else:
        st.warning("Player data found but is empty.")
else:
    st.warning("Syncing... please refresh in 30 seconds.")
