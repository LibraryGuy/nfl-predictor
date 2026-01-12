import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
from nfl_stadiums import NFLStadiums
import requests

# --- 1. SETTINGS ---
st.set_page_config(page_title="NFL Sharp: Logic Fix", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_base_data():
    """Loads only the core stats to ensure the player list never breaks."""
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Flatten MultiIndex immediately
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        # Robust naming: find the name column regardless of what it's called
        name_options = ['player_display_name', 'player_name', 'player']
        for opt in name_options:
            if opt in df.columns:
                df = df.rename(columns={opt: 'player_name'})
                break
        
        # Ensure we don't have duplicate 'player_name' columns
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Base Load Failed: {e}")
        return pd.DataFrame()

# Load the core data
data = load_base_data()

# --- 2. PLAYER SELECTION (THE PART THAT WAS CRASHING) ---
if not data.empty and 'player_name' in data.columns:
    # We use list(set()) as a backup to .unique() for extra safety
    raw_names = data['player_name'].dropna().unique()
    clean_names = sorted([str(p) for p in raw_names if str(p) not in ['nan', 'None', 'Unknown']])
    
    st.title("🏈 NFL Genius: Logic Pro")
    selected_p = st.selectbox("Select Player", clean_names)
    
    # Filter to current player
    p_df = data[data['player_name'] == selected_p].copy()
    p_pos = p_df['position'].iloc[-1] if not p_df.empty else 'WR'
    
    # --- 3. ON-DEMAND WEATHER & DEFENSE ---
    # We only do the complex stuff once a player is chosen
    st.sidebar.header("Matchup Context")
    stadiums = NFLStadiums()
    sel_stad = st.sidebar.selectbox("Venue", sorted(stadiums.get_list_of_stadium_names()))
    
    # Simple calculation for projection
    t_stat = 'passing_yards' if p_pos == 'QB' else 'receiving_yards' if p_pos in ['WR', 'TE'] else 'rushing_yards'
    avg_val = p_df[t_stat].mean()
    
    # Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{selected_p} ({p_pos}) Trends")
        st.plotly_chart(px.line(p_df, x='week', y=t_stat, markers=True), use_container_width=True)
    
    with col2:
        st.metric("Season Average", f"{avg_val:.1f} Yds")
        v_line = st.number_input("Market Line", value=float(round(avg_val)))
        edge = ((avg_val - v_line) / v_line) * 100 if v_line > 0 else 0
        st.metric("Model Edge", f"{edge:.1f}%")

else:
    st.warning("Data is still syncing or the source schema has changed. Please refresh.")
