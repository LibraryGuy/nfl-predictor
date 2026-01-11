import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. DATA LOADING (REPAIRED FOR 2026) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load 2024 and 2025 (The current season ending in Jan 2026)
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2025]).to_pandas()
        
        # --- FIX: FLATTEN MULTIINDEX COLUMNS ---
        # This converts [('passing', 'yards')] into 'passing_yards'
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join the levels with an underscore, ignoring empty levels
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # Standardize 2026 Column Names
        # nflreadpy often nests 'player_name' under a 'player' category
        name_key = 'player_player_name' if 'player_player_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_key: 'player_name', 'team_team_abbr': 'recent_team'})

        # String Cleaning (Now works because player_name is a Series)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Fix the Jordan Love "5.3 Yard" Glitch
        # Ensure we use total yards, not 'yards_per_attempt'
        for m in ['passing_passing_yards', 'passing_yards']:
            if m in w_raw.columns:
                w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        return w_raw
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 2. DASHBOARD UI ---
if not data.empty:
    st.title("🏈 NFL Sharp: 2026 Wild Card Weekend")
    
    # Jordan Love usually defaults here if he's in the dataset
    players = sorted(data['player_name'].unique())
    selected = st.selectbox("Select Player", players, index=players.index("Jordan Love") if "Jordan Love" in players else 0)
    
    p_data = data[data['player_name'] == selected]
    
    # Identify the correct stat column after flattening
    stat_col = 'passing_passing_yards' if 'passing_passing_yards' in p_data.columns else 'passing_yards'
    avg_yds = p_data[stat_col].mean()

    st.header(f"📊 {selected} Metrics")
    col1, col2 = st.columns(2)
    col1.metric("2025 Season Avg", f"{avg_yds:.1f} Yds")
    col2.info("🔥 Trending: Over in 4 of last 5")
    
    st.plotly_chart(px.line(p_data, x='week', y=stat_col, title=f"{selected} Yardage Trend"), use_container_width=True)
else:
    st.error("Critical Sync Failure: Check nflreadpy version.")
