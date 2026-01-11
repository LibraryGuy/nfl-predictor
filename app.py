import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (THE MULTI-INDEX FIX) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Loading 2024 and 2025 seasons (Wild Card 2026 data)
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()
        
        # --- CRITICAL: FLATTEN HEADERS ---
        # This collapses [('offense', 'passing_yards')] into 'passing_yards'
        # This fixes the "'DataFrame' object has no attribute 'str'" error
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join the levels with an underscore, e.g., 'passing_yards'
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # --- DYNAMIC COLUMN MAPPING ---
        # Checks for common naming variations in the 2026 dataset
        name_key = next((c for c in ['player_name', 'player_player_name', 'player_display_name'] if c in w_raw.columns), None)
        team_key = next((c for c in ['recent_team', 'team_team_abbr', 'team_abbr'] if c in w_raw.columns), None)
        
        if name_key: w_raw = w_raw.rename(columns={name_key: 'player_name'})
        if team_key: w_raw = w_raw.rename(columns={team_key: 'recent_team'})

        # Clean strings (Now works because 'player_name' is a Series)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Ensure 'passing_yards' is the game total (Fixes Jordan Love 5.3 glitch)
        yard_col = 'passing_passing_yards' if 'passing_passing_yards' in w_raw.columns else 'passing_yards'
        if yard_col in w_raw.columns:
            w_raw[yard_col] = pd.to_numeric(w_raw[yard_col], errors='coerce').fillna(0)
            w_raw['ui_yards'] = w_raw[yard_col]

        # Merge with Schedule (Weather, Lines, Field)
        # Using home/away logic to ensure road games aren't missed
        df_home = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'home_team'], how='inner')
        df_away = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], right_on=['season', 'week', 'away_team'], how='inner')
        
        return pd.concat([df_home, df_away], ignore_index=True).fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🏈 NFL Sharp Pro")
    if not data.empty:
        player_list = sorted(data['player_name'].unique())
        # Default to Jordan Love for testing the yardage fix
        default_idx = player_list.index("Jordan Love") if "Jordan Love" in player_list else 0
        selected_player = st.selectbox("Search Player", player_list, index=default_idx)
        
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

# --- 4. MAIN DASHBOARD ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player]
    latest = p_data.iloc[-1]
    
    st.header(f"📊 {selected_player} Analytics")
    
    # METRICS ROW (ORIGINAL STYLE)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season Avg", f"{p_data['ui_yards'].mean():.1f} Yds")
    m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
    m3.metric("Wind", f"{latest.get('wind', 0)} mph")
    m4.metric("Spread", latest.get('spread_line', 'N/A'))

    # THE GRAPH
    st.plotly_chart(px.line(p_data, x='week', y='ui_yards', markers=True, 
                            title="Weekly Yardage Trend"), use_container_width=True)
    
    # FOOTER INFO
    st.info(f"Field Surface: {str(latest.get('surface', 'Turf')).title()} | O/U Total: {latest.get('total_line', 'N/A')}")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
