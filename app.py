import streamlit as st
import pandas as pd
import nflreadpy as nfl

st.set_page_config(page_title="NFL Predictor 2026", layout="wide")

# --- 1. LIGHTWEIGHT INDEX LOADING ---
@st.cache_data
def get_player_list():
    """Only loads player names to populate the sidebar dropdown"""
    # This just gets the 'roster' which is tiny compared to 'stats'
    roster = nfl.load_players().to_pandas()
    return sorted(roster['display_name'].dropna().unique())

# --- 2. ON-DEMAND STAT LOADING ---
@st.cache_data
def load_player_specific_stats(player_name):
    """Only runs when a player is selected"""
    # Load stats for current and previous season
    df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
    
    # Filter immediately to save memory
    player_data = df[df['player_display_name'] == player_name].copy()
    return player_data

# --- 3. UI LOGIC ---
st.title("🏈 NFL Performance Predictor")

players = get_player_list()
selected_player = st.selectbox("Search and Select a Player:", players)

if selected_player:
    with st.spinner(f"Fetching stats for {selected_player}..."):
        stats_df = load_player_specific_stats(selected_player)
    
    if not stats_df.empty:
        st.success(f"Loaded {len(stats_df)} games for {selected_player}")
        
        # Now your existing prediction logic goes here
        # Example: Show recent passing yards
        if 'passing_yards' in stats_df.columns:
            st.line_chart(stats_df.set_index('week')['passing_yards'])
    else:
        st.warning("No 2024-2025 stats found for this player.")
