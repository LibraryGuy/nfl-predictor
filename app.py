import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA ENGINE (THE 2026 REPAIR) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load Stats and Schedules
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- FIX 1: FLATTEN & DE-DUPLICATE ---
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- FIX 2: SMART COLUMN MAPPING ---
        # Find the team column (usually 'team_abbr' in 2026)
        team_options = ['team_abbr', 'recent_team', 'team', 'posteam']
        found_team_col = next((c for c in team_options if c in w_raw.columns), None)
        
        if found_team_col:
            w_raw = w_raw.rename(columns={found_team_col: 'recent_team'})
        
        # Standardize Player Names
        name_col = next((c for c in ['player_display_name', 'player_name', 'player'] if c in w_raw.columns), None)
        if name_col:
            w_raw = w_raw.rename(columns={name_col: 'player_name'})
            w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()

        # --- FIX 3: THE "HOME/AWAY" WEATHER SYNC ---
        # We duplicate schedules so every game is tied to both the Home and Away team
        home_sched = s_raw.copy().rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
        away_sched = s_raw.copy().rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
        # Flip spread for away teams
        if 'spread_line' in away_sched.columns:
            away_sched['spread_line'] = away_sched['spread_line'] * -1
            
        full_sched = pd.concat([home_sched, away_sched], ignore_index=True)

        # Final Merge: Stats + Schedule (Weather/Lines/Surface)
        df = w_raw.merge(full_sched, on=['season', 'week', 'recent_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (YOUR CUSTOM FEATURES) ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty and 'player_name' in data.columns:
        player_list = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", player_list)
        
        st.divider()
        if st.button("Add to Parlay"):
            if selected_player not in st.session_state.parlay_legs:
                st.session_state.parlay_legs.append(selected_player)
                st.success(f"Added {selected_player} to slip")
            
        if st.session_state.parlay_legs:
            st.subheader("Current Slip")
            for leg in st.session_state.parlay_legs:
                st.write(f"✅ {leg}")
            if st.button("Clear Slip"):
                st.session_state.parlay_legs = []
                st.rerun()

# --- 4. MAIN DASHBOARD ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    if not p_data.empty:
        latest = p_data.iloc[-1]
        st.header(f"📊 {selected_player} Analytics ({latest['recent_team']})")
        
        # The 4-Metric Row (Weather & Betting now restored)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Avg", f"{p_data['passing_yards'].mean():.1f} Yds")
        # Handle dome games which often have 'None' for temp
        temp = latest.get('temp', 70) if latest.get('roof') in ['dome', 'closed'] else latest.get('temp', 'N/A')
        m2.metric("Temp", f"{temp}°F")
        m3.metric("Wind", f"{latest.get('wind', 0)} mph")
        m4.metric("Spread", latest.get('spread_line', 'N/A'))

        # Trend Chart
        st.plotly_chart(px.line(p_data, x='week', y='passing_yards', markers=True, 
                                title=f"2024-25 Performance Trend"), use_container_width=True)
        
        # Matchup Footer
        st.info(f"🏟️ Venue: {latest.get('stadium', 'Unknown')} ({latest.get('roof', 'Outdoors')}) | 📈 Over/Under: {latest.get('total_line', 'N/A')}")
    else:
        st.warning("No performance records found for the selected player.")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
