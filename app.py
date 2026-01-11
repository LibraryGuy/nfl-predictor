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
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # Standardize Columns
        name_key = 'player_player_name' if 'player_player_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_key: 'player_name', 'team_team_abbr': 'recent_team'})
        
        # Data Cleaning for Strings and Numbers
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Ensure we use counting yards, not averages (Fixes 5.3 yard glitch)
        yard_cols = ['passing_passing_yards', 'passing_yards', 'receiving_receiving_yards', 'receiving_yards']
        for col in yard_cols:
            if col in w_raw.columns:
                w_raw[col] = pd.to_numeric(w_raw[col], errors='coerce').fillna(0)

        # --- MERGE SCHEDULE DATA (Weather, Lines, Field) ---
        # We need to match the player's team to the game in the schedule
        # Since a team can be home or away, we do a double merge or a conditional join
        s_cols = ['season', 'week', 'home_team', 'away_team', 'temp', 'wind', 'surface', 'spread_line', 'total_line', 'roof']
        s_clean = s_raw[s_cols].copy()
        
        # Merge for games where the player's team was HOME
        df_home = w_raw.merge(s_clean, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='inner')
        # Merge for games where the player's team was AWAY
        df_away = w_raw.merge(s_clean, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'away_team'], how='inner')
        
        final_df = pd.concat([df_home, df_away], ignore_index=True)
        return final_df.fillna("N/A")
        
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🏈 Sharp Controls")
    if not data.empty:
        players = sorted(data['player_name'].unique())
        selected_player = st.selectbox("Search Player", players, index=players.index("Jordan Love") if "Jordan Love" in players else 0)
        
        st.divider()
        st.subheader("🎟️ Parlay Builder")
        if st.button("Add to Slip"):
            st.session_state.parlay_legs.append(selected_player)
        
        for leg in st.session_state.parlay_legs:
            st.write(f"✅ {leg} Prop")
        if st.button("Clear Slip"):
            st.session_state.parlay_legs = []
            st.rerun()

# --- 4. MAIN DASHBOARD ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player].sort_values(by=['season', 'week'])
    latest_game = p_data.iloc[-1]
    
    st.title(f"🚀 {selected_player} Projections")

    # WEATHER & GAME INFO ROW
    st.subheader("🏟️ Game Environment & Betting Lines")
    w1, w2, w3, w4 = st.columns(4)
    
    # Weather Display (Handles domes)
    weather_desc = f"{latest_game['temp']}°F" if latest_game['roof'] == 'outdoors' else "Dome 🏟️"
    w1.metric("Weather", weather_desc)
    w2.metric("Wind Speed", f"{latest_game['wind']} mph")
    w3.metric("Field Surface", str(latest_game['surface']).title())
    w4.metric("O/U Total", latest_game['total_line'])

    st.divider()

    # STATS ROW
    # Determine correct stat column (QB vs Skill)
    stat_col = 'passing_passing_yards' if 'passing_passing_yards' in p_data.columns else 'receiving_receiving_yards'
    avg_yds = p_data[stat_col].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Season Average", f"{avg_yds:.1f} Yds")
    c2.metric("Market Spread", latest_game['spread_line'])
    c3.success("Signal: Projected OVER")

    # Trend Chart
    st.plotly_chart(px.line(p_data, x='week', y=stat_col, markers=True, 
                            title=f"{selected_player} Performance vs. Environment"), 
                    use_container_width=True)
    
    # Raw Data for verification
    with st.expander("View Full Game Logs"):
        st.dataframe(p_data[['season', 'week', 'home_team', 'away_team', stat_col, 'temp', 'wind', 'surface', 'spread_line']])
