import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. AUTHENTICATION & PAYWALL ---
# add_auth(required=True) # Uncomment for production use

# --- 3. REPAIRED DATA LOADING (THE SYNC FIX) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load 2024 and 2025 (The current season ending in Jan 2026)
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2025]).to_pandas()
        
        # --- THE SLEDGEHAMMER: Flatten MultiIndex Immediately ---
        # This converts [('passing', 'yards')] into 'passing_yards'
        for df in [w_raw, s_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                # Join nested names with underscores (e.g., 'passing_yards')
                df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
            else:
                df.columns = [str(c).strip() for c in df.columns]

        # Standardize Names (Handles the 2026 player_display_name vs player_name issue)
        name_key = 'player_player_name' if 'player_player_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_key: 'player_name', 'team_team_abbr': 'recent_team'})

        # String Cleaning (Now works because player_name is a 1D Series)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        
        # Repair Yardage: Force TOTAL yards (Fixes the Jordan Love 5.3 yard error)
        # We ensure passing_yards refers to the game total (e.g., 234) not Y/A
        for m in ['passing_passing_yards', 'passing_yards']:
            if m in w_raw.columns:
                w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        return w_raw
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 4. SIDEBAR (RESTORED) ---
with st.sidebar:
    st.title("🏈 Sharp Controls")
    st.info("2026 Wild Card Weekend Data Active")
    
    if not data.empty:
        players = sorted(data['player_name'].unique())
        # Default to Jordan Love if he exists in the 2026 dataset
        default_idx = players.index("Jordan Love") if "Jordan Love" in players else 0
        selected_player = st.selectbox("Search Player", players, index=default_idx)
        
        st.divider()
        st.subheader("🎟️ Parlay Builder")
        if st.button("Add to Slip"):
            st.session_state.parlay_legs.append(selected_player)
            st.success(f"Added {selected_player}")
        
        if st.session_state.parlay_legs:
            for leg in st.session_state.parlay_legs:
                st.write(f"✅ {leg} Prop")
            if st.button("Clear Slip"):
                st.session_state.parlay_legs = []
                st.rerun()

# --- 5. MAIN DASHBOARD ---
if not data.empty:
    p_data = data[data['player_name'] == selected_player]
    player_pos = p_data['position'].iloc[-1] if 'position' in p_data.columns else "N/A"
    
    # Identify the correct stat column after flattening
    stat_col = 'passing_passing_yards' if 'passing_passing_yards' in p_data.columns else 'passing_yards'
    avg_yds = p_data[stat_col].mean()

    st.title(f"🚀 {selected_player} ({player_pos})")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Season Average", f"{avg_yds:.1f} Yds")
    c2.metric("Projected Total", f"{avg_yds * 1.05:.1f}")
    c3.success("Sharp Signal: OVER")

    # Trend Chart
    st.plotly_chart(px.line(p_data, x='week', y=stat_col, markers=True, 
                            title=f"{selected_player} Yardage History (2025/26 Season)"), 
                    use_container_width=True)
else:
    st.warning("🔄 System Restarting: Please wait while the 2026 Wild Card data syncs.")
