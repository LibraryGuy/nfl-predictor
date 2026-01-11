import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (THE "IRONCLAD" CLEANER) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE RESET: FLATTEN AND DE-DUPLICATE ---
        for df in [w_raw, s_raw]:
            # Step A: Flatten MultiIndex (e.g., ('player', 'name') -> 'player_name')
            df.columns = [
                "_".join(filter(None, map(str, col))).strip() 
                if isinstance(col, tuple) else str(col).strip() 
                for col in df.columns.values
            ]
            
            # Step B: THE VITAL FIX - Remove duplicate column names
            # If 'player_name' exists twice, this keeps only the first one.
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- MAPPING TO YOUR DASHBOARD KEYS ---
        # Map the complex 2026 names to the simple ones your UI expects
        mapping = {
            'player_display_name': 'player_name',
            'player_player_name': 'player_name',
            'team_recent_team': 'recent_team',
            'passing_yards': 'passing_yards' # Targets the total volume stat
        }
        w_raw = w_raw.rename(columns=mapping)

        # Force 'player_name' to be a clean Series (this is where the crash happened)
        if 'player_name' in w_raw.columns:
            # We use .squeeze() as a final safety to force it into a Series
            names = w_raw['player_name']
            if isinstance(names, pd.DataFrame):
                names = names.iloc[:, 0] # Take first column if still a DF
            w_raw['player_name'] = names.astype(str).str.strip()
        
        # Clean yardage data (fixing the Jordan Love 5.3 yard average glitch)
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Merge with Schedule
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (YOUR ORIGINAL LAYOUT) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty and 'player_name' in data.columns:
        # Create a clean list for the dropdown
        player_list = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", player_list)
        
        st.divider()
        if st.button("Add to Parlay"):
            if selected_player not in st.session_state.parlay_legs:
                st.session_state.parlay_legs.append(selected_player)
                st.success(f"Added {selected_player}")
            
        if st.session_state.parlay_legs:
            st.subheader("Current Slip")
            for leg in st.session_state.parlay_legs:
                st.write(f"✅ {leg}")
            if st.button("Clear Slip"):
                st.session_state.parlay_legs = []
                st.rerun()

# --- 4. MAIN DASHBOARD (DASHBOARD AS IT WAS) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics")
        
        # The 4-Metric Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", latest.get('spread_line', 'N/A'))

        # Chart
        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Performance Trend"), use_container_width=True)
        
        st.info(f"🏟️ Surface: {str(latest.get('surface', 'Turf')).title()} | 📉 O/U: {latest.get('total_line', 'N/A')}")
    else:
        st.warning("No data records found for this player.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
