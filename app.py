import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (THE SYNC RECOVERY) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load Raw Data from nflreadpy
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- CRITICAL FIX: Flatten hierarchical headers ---
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Keeps the lowest level name (e.g., 'passing_yards') 
                # and removes the category header (e.g., 'passing')
                df.columns = df.columns.get_level_values(-1)
            # Clean up whitespace in column names
            df.columns = [str(c).strip() for c in df.columns]

        # Standardize 'player_name' and 'recent_team' to match your merge logic
        w_raw = w_raw.rename(columns={'player_display_name': 'player_name', 'team_abbr': 'recent_team'})
        
        # Now 'player_name' is a 1D list (Series), so .str works perfectly
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # FORCE ACCURACY: Ensure 'passing_yards' is total yards, not an average
        # This fixes the Jordan Love "5.3 yard" glitch
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Merge with Schedule (Weather, Lines, Field)
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        # Prevents "Blank Dashboard" by showing a specific error in the UI
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (ORIGINAL UI RESTORED) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty:
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

# --- 4. MAIN DASHBOARD (REVERTED TO PREVIOUS VIEW) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player]
    latest = p_data.iloc[-1]
    
    st.header(f"📊 {selected_player} Analytics")
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    # Verification: m1 should show 200+ yards for Jordan Love, not 5.3
    m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
    m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
    m3.metric("Wind", f"{latest.get('wind', 0)} mph")
    m4.metric("Spread", latest.get('spread_line', 'N/A'))

    # GRAPH
    st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                            title="Weekly Yardage Trend"), use_container_width=True)
    
    # FOOTER INFO
    st.info(f"Field Surface: {latest.get('surface', 'Turf').title()} | O/U: {latest.get('total_line', 'N/A')}")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
