import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. SETTINGS & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE (SMART LOADING) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Raw Data
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- A. FLATTEN HEADERS ---
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- B. SMART MAPPING ---
        def find_col(df, options):
            for opt in options:
                if opt in df.columns: return opt
            for col in df.columns:
                if any(opt in col.lower() for opt in options): return col
            return None

        # Standardize Names for the Dashboard
        name_key = find_col(w_raw, ['player_name', 'player_display_name'])
        team_key = find_col(w_raw, ['recent_team', 'team_abbr'])
        yard_key = find_col(w_raw, ['passing_yards', 'pass_yards'])

        if name_key: w_raw = w_raw.rename(columns={name_key: 'player_name'})
        if team_key: w_raw = w_raw.rename(columns={team_key: 'recent_team'})
        if yard_key: w_raw = w_raw.rename(columns={yard_key: 'passing_yards'})

        # Clean Strings and Numbers (Stops the .str crash)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        w_raw['passing_yards'] = pd.to_numeric(w_raw['passing_yards'], errors='coerce').fillna(0)

        # --- C. THE HOME/AWAY SYNC FIX ---
        # We need weather and lines for BOTH Home and Away teams
        s_home = s_raw.copy().rename(columns={'home_team': 'team', 'away_team': 'opponent'})
        s_away = s_raw.copy().rename(columns={'away_team': 'team', 'home_team': 'opponent'})
        if 'spread_line' in s_away.columns:
            s_away['spread_line'] = s_away['spread_line'] * -1 # Flip spread for away team
            
        full_sched = pd.concat([s_home, s_away], ignore_index=True)

        # Merge Everything
        df = w_raw.merge(full_sched, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (YOUR ORIGINAL PARLAY TOOLS) ---
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

# --- 4. MAIN DASHBOARD (RECONSTRUCTED LAYOUT) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics")
        
        # THE 4-METRIC ROW
        m1, m2, m3, m4 = st.columns(4)
        
        # Metric 1: Season Avg (Using fixed passing_yards)
        avg_yds = p_data['passing_yards'].mean()
        m1.metric("Season Avg", f"{avg_yds:.1f} Yds")
        
        # Metric 2: Temperature (Checks for Dome context)
        temp = latest.get('temp', 'N/A')
        if latest.get('roof') in ['dome', 'closed']:
            temp = "70 (Dome)"
        m2.metric("Temp", f"{temp}°F" if isinstance(temp, (int, float)) else temp)
        
        # Metric 3: Wind
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        
        # Metric 4: Spread
        m4.metric("Spread", f"{latest.get('spread_line', 'N/A')}")

        # TREND CHART
        fig = px.line(p_data, x='week', y='passing_yards', markers=True, 
                      title=f"Performance Trend: {selected_player}",
                      labels={'passing_yards': 'Passing Yards', 'week': 'Week'})
        st.plotly_chart(fig, use_container_width=True)
        
        # FOOTER INFO
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"🏟️ **Stadium:** {latest.get('stadium', 'Unknown')} | **Surface:** {str(latest.get('surface', 'Turf')).title()}")
        with c2:
            st.info(f"📈 **Matchup:** vs {latest.get('opponent', 'N/A')} | **O/U Total:** {latest.get('total_line', 'N/A')}")
            
    else:
        st.warning("No performance records found for the selected player.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
