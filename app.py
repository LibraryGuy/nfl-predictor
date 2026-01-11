import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. SETUP ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE (MANDATORY FLATTENING) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE FIX: FORCED FLATTENING ---
        # This converts [('player', 'player_name')] into just 'player_name'
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join levels with underscore if they exist, or just take the last level
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # --- RE-MAPPING TO YOUR ORIGINAL VARIABLES ---
        # The library renamed columns. We force them back so your old code doesn't break.
        col_map = {
            'player_player_name': 'player_name',
            'player_display_name': 'player_name',
            'team_recent_team': 'recent_team',
            'team_team_abbr': 'recent_team',
            'passing_passing_yards': 'passing_yards'
        }
        w_raw = w_raw.rename(columns=col_map)

        # Now 'player_name' is a 1D Series. .str WILL work now.
        if 'player_name' in w_raw.columns:
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Merge with Schedule
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (YOUR ORIGINAL UI) ---
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
    # This filter now works because 'player_name' is a single column
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
                                title="Weekly Yardage Trend"), use_container_width=True)
    else:
        st.warning("No data found for this player.")
else:
    st.warning("Syncing... please refresh in 30 seconds.")
