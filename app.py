import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CORE CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE (SMART COLUMN MAPPING) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- THE FIX: FLATTEN AND AUTO-FIND COLUMNS ---
        for df in [w_raw, s_raw]:
            # Flatten MultiIndex if it exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]
            # Remove duplicates
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- SMART MAPPING: SEARCH FOR MISSING KEYS ---
        # We look for ANY column that looks like Player Name or Team
        def find_col(df, options):
            for opt in options:
                if opt in df.columns: return opt
            # If not found, look for partial matches (e.g., 'player_name' inside 'offense_player_name')
            for col in df.columns:
                if any(opt in col.lower() for opt in options): return col
            return None

        # Re-assign keys to your dashboard's expected names
        name_col = find_col(w_raw, ['player_name', 'player_display_name', 'player'])
        team_col = find_col(w_raw, ['recent_team', 'team_abbr', 'team', 'posteam'])
        yard_col = find_col(w_raw, ['passing_yards', 'pass_yards'])

        if name_col: w_raw = w_raw.rename(columns={name_col: 'player_name'})
        if team_col: w_raw = w_raw.rename(columns={team_col: 'recent_team'})
        if yard_col: w_raw = w_raw.rename(columns={yard_col: 'passing_yards'})

        # Final Cleaning for .str and Numeric
        if 'player_name' in w_raw.columns:
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        if 'passing_yards' in w_raw.columns:
            w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # Merge with Schedule (Syncing player stats with game conditions)
        # We check for existence to prevent the KeyError: 'recent_team'
        if 'recent_team' in w_raw.columns and 'home_team' in s_raw.columns:
            df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                             right_on=['season', 'week', 'home_team'], how='left')
        else:
            # Fallback merge if team names are still weird
            df = w_raw 
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (UNCHANGED DASHBOARD FEATURES) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty and 'player_name' in data.columns:
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

        # Trend Chart
        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Performance Trend"), use_container_width=True)
        
        # Footer
        st.info(f"🏟️ Surface: {str(latest.get('surface', 'Turf')).title()} | 📉 O/U: {latest.get('total_line', 'N/A')}")
    else:
        st.warning("Player stats identified, but no specific game records found.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
