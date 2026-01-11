import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG & SESSION STATE ---
# We keep these exactly as you had them
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (FLATTEN & TARGET FIX) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data for current Wild Card context
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- FIX 1: FLATTEN MULTIINDEX ---
        # This stops the 'DataFrame' object has no attribute 'str' error
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # We take the last part of the header, e.g., ('offense', 'passing_yards') -> 'passing_yards'
                df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns.values]
            df.columns = [str(c).strip() for c in df.columns]

        # --- FIX 2: TARGET REAL YARDS (NOT PER-ATTEMPT) ---
        # Re-mapping to your original variables and avoiding the '5.3 yard' glitch
        col_map = {
            'player_display_name': 'player_name', 
            'team_abbr': 'recent_team'
        }
        w_raw = w_raw.rename(columns=col_map)
        
        # Ensure 'passing_yards' is numeric total and not an average
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Now 'player_name' is a single column, so .str will work perfectly!
        if 'player_name' in w_raw.columns:
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()

        # Merge with Schedule (Weather, Surface, Lines)
        # Using a left merge on the player's team and week
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (UNTOUCHED FEATURES) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty:
        # Player selection works because the index is flattened
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

# --- 4. MAIN DASHBOARD (RETAINED LAYOUT) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    if not p_data.empty:
        latest = p_data.iloc[-1]
        
        st.header(f"📊 {selected_player} Analytics")
        
        # Original 4-metric row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", latest.get('spread_line', 'N/A'))

        # Weekly Trend Line Chart
        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Performance Trend"), use_container_width=True)
        
        # Footer Info Box
        st.info(f"🏟️ Surface: {str(latest.get('surface', 'Turf')).title()} | 📉 O/U Total: {latest.get('total_line', 'N/A')}")
    else:
        st.warning("No performance data found for this selection.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
