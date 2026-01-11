import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA LOADING (FIXED MULTI-INDEX) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- THE FIX: Collapse MultiIndex into flat strings ---
        # This prevents the 'DataFrame object has no attribute str' error
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # We join levels with an underscore, e.g., ('offense', 'passing_yards') -> 'offense_passing_yards'
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # Standardize names after flattening
        # Note: Depending on the flatten result, these might be 'player_player_name' or just 'player_name'
        name_key = 'player_player_name' if 'player_player_name' in w_raw.columns else 'player_name'
        team_key = 'recent_team' if 'recent_team' in w_raw.columns else 'team_team_abbr'
        
        w_raw = w_raw.rename(columns={name_key: 'player_name', team_key: 'recent_team'})
        
        # Now 'player_name' is a 1D Series, so .str.strip() works!
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Force Passing Yards to be a number (Fixes Jordan Love 5.3 glitch)
        # We look for the flattened version of the yardage column
        yard_col = 'passing_passing_yards' if 'passing_passing_yards' in w_raw.columns else 'passing_yards'
        if yard_col in w_raw.columns:
            w_raw[yard_col] = pd.to_numeric(w_raw[yard_col], errors='coerce').fillna(0)

        # Merge with Schedule
        df = w_raw.merge(s_raw, left_on=['season', 'week', 'recent_team'], 
                         right_on=['season', 'week', 'home_team'], how='left')
        
        # Add a helper column for your specific UI code
        df['ui_yards'] = df[yard_col]
        
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
        selected_player = st.selectbox("Search Player", player_list, 
                                       index=player_list.index("Jordan Love") if "Jordan Love" in player_list else 0)
        
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

# --- 4. MAIN DASHBOARD (REVERTED) ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player]
    latest = p_data.iloc[-1]
    
    st.header(f"📊 {selected_player} Analytics")
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season Avg", f"{p_data['ui_yards'].mean():.1f} Yds")
    m2.metric("Temp", f"{latest.get('temp', 'N/A')}°F")
    m3.metric("Wind", f"{latest.get('wind', 0)} mph")
    m4.metric("Spread", latest.get('spread_line', 'N/A'))

    # GRAPH
    st.plotly_chart(px.line(p_data, x='week', y='ui_yards', markers=True, 
                            title="Weekly Yardage Trend"), use_container_width=True)
    
    # FOOTER INFO
    st.info(f"Field Surface: {latest.get('surface', 'Turf').title()} | O/U: {latest.get('total_line', 'N/A')}")
else:
    st.warning("Dashboard syncing... please refresh in 30 seconds.")
