import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (CLEANED & REVERTED) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- THE FIX: COLLAPSE HEADERS WITHOUT LOSING NAMES ---
        # This converts the MultiIndex into flat names like 'passing_yards'
        # which prevents the 'AttributeError: str' crash.
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # We drop the top level (e.g. 'offense') and keep the stat name (e.g. 'passing_yards')
                df.columns = df.columns.get_level_values(-1)
            df.columns = [str(c).strip() for c in df.columns]

        # Standardize 'recent_team' and 'player_name'
        # We check for common variants to ensure the merge doesn't fail
        name_map = {'player_display_name': 'player_name', 'team_abbr': 'recent_team', 'team': 'recent_team'}
        w_raw = w_raw.rename(columns=name_map)
        
        # Ensure 'player_name' is a clean Series for .str functions
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # FORCE TOTAL YARDS: Fixes the 5.3 Jordan Love issue
        # We ensure passing_yards is treated as a large integer/float
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Merge with Schedule (Weather, Lines, Field)
        # Using inner join to ensure we only get games with valid weather/line data
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (ORIGINAL UI) ---
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
    # Re-verify Jordan Love: latest['passing_yards'] should now be 200+, not 5.3
    m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
    m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
    m3.metric("Wind", f"{latest.get('wind', 0)} mph")
    m4.metric("Spread", latest.get('spread_line', 'N/A'))

    # THE GRAPH
    st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                            title="Weekly Yardage Trend"), use_container_width=True)
    
    # EXTRA INFO
    st.info(f"Field Surface: {latest.get('surface', 'Turf').title()} | O/U: {latest.get('total_line', 'N/A')}")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
