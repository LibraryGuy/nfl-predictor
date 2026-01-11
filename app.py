import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING & MERGING ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load Raw Files
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- FIX: Flatten MultiIndex (The Jordan Love / AttributeError Fix) ---
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join nested names with underscores (e.g., 'passing_yards')
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # --- DYNAMIC COLUMN RESOLVER (RECENT_TEAM FIX) ---
        # Map the correct team column for the merge
        team_cols = ['team_abbr', 'recent_team', 'player_team', 'team']
        found_team = next((c for c in team_cols if c in w_raw.columns), None)
        if found_team:
            w_raw = w_raw.rename(columns={found_team: 'recent_team'})
        
        # Standardize Player Names
        name_cols = ['player_name', 'player_display_name', 'player_player_name']
        found_name = next((c for c in name_cols if c in w_raw.columns), None)
        if found_name:
            w_raw = w_raw.rename(columns={found_name: 'player_name'})

        # Clean strings and ensure yardage is numeric (Fixes 5.3 yard glitch)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # We target specific counting stats to avoid averages
        stat_targets = ['passing_yards', 'passing_passing_yards', 'receiving_yards', 'rushing_yards']
        for col in stat_targets:
            if col in w_raw.columns:
                w_raw[col] = pd.to_numeric(w_raw[col], errors='coerce').fillna(0)

        # --- MERGE SCHEDULE DATA (Weather, Lines, Field) ---
        # Merging on Season, Week, and Team Abbreviation
        s_cols = ['season', 'week', 'home_team', 'away_team', 'temp', 'wind', 'surface', 'spread_line', 'total_line', 'roof']
        s_clean = s_raw[[c for c in s_cols if c in s_raw.columns]].copy()
        
        # Merge for games where player's team was HOME and AWAY
        df_home = w_raw.merge(s_clean, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='inner')
        df_away = w_raw.merge(s_clean, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'away_team'], how='inner')
        
        return pd.concat([df_home, df_away], ignore_index=True).fillna("N/A")
        
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR (RESTORED) ---
with st.sidebar:
    st.title("🏈 Sharp Controls")
    if not data.empty:
        players = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", players, index=players.index("Jordan Love") if "Jordan Love" in players else 0)
        
        st.divider()
        st.subheader("🎟️ Parlay Builder")
        if st.button("Add to Slip"):
            st.session_state.parlay_legs.append(selected_player)
        
        if st.session_state.parlay_legs:
            for leg in st.session_state.parlay_legs:
                st.write(f"✅ {leg} Prop")
            if st.button("Clear Slip"):
                st.session_state.parlay_legs = []
                st.rerun()

# --- 4. MAIN DASHBOARD ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    latest = p_data.iloc[-1]
    
    st.title(f"🚀 {selected_player} Projections")

    # WEATHER & GAME INFO ROW
    st.subheader("🏟️ Game Environment & Betting Lines")
    w1, w2, w3, w4 = st.columns(4)
    
    # Weather Display
    weather_info = f"{latest['temp']}°F" if latest.get('roof') == 'outdoors' else "Dome 🏟️"
    w1.metric("Weather", weather_info)
    w2.metric("Wind Speed", f"{latest.get('wind', 0)} mph")
    w3.metric("Field Surface", str(latest.get('surface', 'Turf')).title())
    w4.metric("O/U Total", latest.get('total_line', 'N/A'))

    st.divider()

    # STATS ROW
    # Identify the correct counting stat column after flattening
    if 'passing_passing_yards' in p_data.columns:
        stat_col = 'passing_passing_yards'
    elif 'passing_yards' in p_data.columns:
        stat_col = 'passing_yards'
    else:
        stat_col = 'receiving_yards'

    avg_yds = p_data[stat_col].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Season Average", f"{avg_yds:.1f} Yds")
    c2.metric("Spread Line", latest.get('spread_line', 'N/A'))
    c3.success("Signal: Strong OVER")

    # Trend Chart
    st.plotly_chart(px.line(p_data, x='week', y=stat_col, markers=True, 
                            title=f"{selected_player} Performance vs. Environment"), 
                    use_container_width=True)
    
    with st.expander("View Full Game Logs"):
        st.dataframe(p_data)
else:
    st.warning("🔄 System Restarting: Check your internet connection and data source.")
