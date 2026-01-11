import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CORE SETUP ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA CURE (WEATHER & LINES FIX) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Player Stats & Schedules
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- FIX A: FLATTEN HEADERS ---
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- FIX B: THE "HOME/AWAY" WEATHER SYNC ---
        # We create a lookup where every game exists for BOTH teams
        s_home = s_raw.copy()
        s_home['team'] = s_home['home_team']
        s_home['opponent'] = s_home['away_team']
        
        s_away = s_raw.copy()
        s_away['team'] = s_away['away_team']
        s_away['opponent'] = s_away['home_team']
        # For away teams, the spread is reversed
        if 'spread_line' in s_away.columns:
            s_away['spread_line'] = s_away['spread_line'] * -1
            
        # Combine them so every team/week has a row with weather/lines
        s_lookup = pd.concat([s_home, s_away], ignore_index=True)

        # --- FIX C: DEFENSE RANKINGS ---
        # Calculate Team Defense (Yards Allowed) from player stats
        def_stats = w_raw.groupby(['season', 'week', 'recent_team'])['passing_yards'].sum().reset_index()
        def_stats.columns = ['season', 'week', 'opponent', 'def_yards_allowed']

        # --- FINAL MERGE ---
        # 1. Map Player Stats to Team (recent_team)
        w_raw = w_raw.rename(columns={'player_display_name': 'player_name', 'team_abbr': 'recent_team'})
        
        # 2. Join Player Stats + Weather/Lines (s_lookup)
        df = w_raw.merge(s_lookup, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'team'], how='left')
        
        # 3. Join with Defense Data (Against the Opponent)
        df = df.merge(def_stats, on=['season', 'week', 'opponent'], how='left')

        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (UNCHANGED FEATURES) ---
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

# --- 4. MAIN DASHBOARD (RETAINED & EXPANDED) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics")
        
        # Row 1: Metrics (Weather & Betting Lines now fixed)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        m2.metric("Temp", f"{latest.get('temp', 70)}°F") # Domes default to 70
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", f"{latest.get('spread_line', 0)}")

        # Row 2: Defense Option
        st.subheader(f"🛡️ Matchup vs {latest.get('opponent', 'Opponent')}")
        d_avg = latest.get('def_yards_allowed', 0)
        st.progress(min(d_avg / 400, 1.0), text=f"Opponent Allowing {d_avg:.0f} Pass Yds this week")

        # Trend Chart
        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title="Weekly Performance Trend"), use_container_width=True)
        
        st.info(f"🏟️ Roof: {str(latest.get('roof', 'Outdoors')).title()} | 📉 O/U: {latest.get('total_line', 'N/A')}")
    else:
        st.warning("Player found, but no game history is available.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
