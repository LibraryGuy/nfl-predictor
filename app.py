import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. DATA LOADING WITH ID MAPPING ---
@st.cache_data(ttl=3600)
def get_player_registry_with_ids():
    try:
        # load_players() is the "Source of Truth" for IDs in 2025/2026
        players = nfl.load_players().to_pandas()
        
        # Standardize columns
        name_col = 'display_name' if 'display_name' in players.columns else 'player_name'
        players = players.rename(columns={name_col: 'full_name'})
        
        # Filter for fantasy skill positions
        active = players[players['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        
        # Create a dictionary for the selectbox: { "Full Name (POS)": "gsis_id" }
        registry = {
            f"{row['full_name']} ({row['position']})": row['gsis_id'] 
            for _, row in active.iterrows() 
            if pd.notna(row['gsis_id'])
        }
        return registry
    except Exception as e:
        st.error(f"ID Registry Error: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_stats_by_id(gsis_id):
    try:
        curr_season = nfl.get_current_season()
        # Request WEEKLY level for the current season
        df = nfl.load_player_stats(seasons=[curr_season], summary_level='week').to_pandas()
        
        # Filter by ID (The most reliable way)
        p_df = df[df['player_id'] == gsis_id].copy()
        
        # Fallback: If current season is empty, pull 2024 data
        if p_df.empty:
            df_prev = nfl.load_player_stats(seasons=[curr_season-1], summary_level='week').to_pandas()
            p_df = df_prev[df_prev['player_id'] == gsis_id].copy()
            
        return p_df
    except Exception as e:
        return pd.DataFrame()

# --- 2. DASHBOARD UI ---
st.set_page_config(page_title="NFL Sharp: ID-Matched", layout="wide")

registry = get_player_registry_with_ids()
selected_label = st.sidebar.selectbox("Select Player", list(registry.keys()))
selected_id = registry.get(selected_label)

if selected_id:
    p_data = get_stats_by_id(selected_id)
    
    if not p_data.empty:
        st.title(f"📊 {selected_label}")
        
        # Logic to handle the specific market
        pos = p_data['position'].iloc[-1]
        market_options = {
            'QB': ['passing_yards', 'passing_tds'],
            'RB': ['rushing_yards', 'rushing_tds'],
            'WR': ['receiving_yards', 'receiving_tds'],
            'TE': ['receiving_yards', 'receiving_tds']
        }
        
        stat_col = st.sidebar.selectbox("Market", market_options.get(pos, ['receiving_yards']))
        
        # Display Actual Gamelog
        st.subheader("Season Gamelog")
        display_df = p_data[['week', 'opponent_team', stat_col]].sort_values('week')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Run Prediction (Monte Carlo)
        avg_val = p_data[stat_col].mean()
        sims = np.random.poisson(avg_val, 10000) if "td" in stat_col else np.random.normal(avg_val, p_data[stat_col].std() or 5, 10000)
        
        st.metric("Model Projection", f"{np.mean(sims):.1f}")
        
    else:
        st.warning(f"No game-level stats found for {selected_label} in the current or previous season data.")
        st.info("Debugging Info: This can happen for rookies who haven't played a regular season snap yet.")
