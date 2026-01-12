import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Genius Pro", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA ENGINE (Stabilized) ---
@st.cache_data(ttl=3600)
def load_and_fix_data():
    try:
        # Load data (nflreadpy handles 2024-2025)
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # COLUMN SAFETY: Rename common variations to our 'standard' names
        # This prevents the AttributeError by ensuring these keys exist
        rename_map = {
            'player_display_name': 'player_name',
            'opponent_team': 'opponent',
            'recent_team': 'team',
            'rushing_tds': 'rush_td',
            'receiving_tds': 'rec_td'
        }
        # Only rename if the source column actually exists
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # DEFENSE ENGINE: Calculate avg yards allowed by team + position
        # This powers your "Defense Option"
        def_df = df.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return df.fillna(0), def_df
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, def_data = load_fix_data()

# --- 3. THE ANALYTICS ---
def calc_td_prob(avg_td, modifier):
    # Poisson Distribution: Probability of at least 1 TD
    lam = avg_td * modifier
    return (1 - exp(-lam)) * 100

# --- 4. THE INTERFACE ---
st.title("🏈 NFL Genius: Parlay & Defense Suite")

# Only run UI if data successfully loaded
if not data.empty and 'player_name' in data.columns:
    with st.sidebar:
        st.header("🛒 Parlay Builder")
        if st.session_state.parlay_legs:
            for leg in st.session_state.parlay_legs:
                st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
            if st.button("Clear All Legs"):
                st.session_state.parlay_legs = []
                st.rerun()
        else:
            st.info("No legs added yet.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Player Selection
        players = sorted(data['player_name'].unique())
        p_name = st.selectbox("Select Player", players)
        p_data = data[data['player_name'] == p_name]
        p_pos = p_data['position'].iloc[-1]
        
        # Defense Selection
        opponents = sorted(data['opponent'].unique())
        target_opp = st.selectbox("Select Opponent Defense", opponents)
        
        # DEFENSE MATCHUP LOGIC
        d_match = def_data[(def_data['def_team'] == target_opp) & (def_data['position'] == p_pos)]
        # If defense is "leaky" (avg > 55 yds to this pos), we give a 15% boost
        mod = 1.15 if (not d_match.empty and d_match['receiving_yards'].iloc[0] > 55) else 0.92

    with col2:
        st.subheader("🎯 Sharp Metrics")
        mkt_line = st.number_input("Market Yardage Line", value=50.5)
        
        # Yardage Proj
        avg_yds = p_data['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_data['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - mkt_line) / mkt_line) * 100
        
        # TD Probability Proj
        avg_td = (p_data['rush_td'].mean() + p_data['rec_td'].mean())
        td_p = calc_td_prob(avg_td, mod)
        
        st.metric("Model Projection", f"{proj_yds:.1f} Yds", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD Prob", f"{td_p:.1f}%")

    # --- ADD TO PARLAY ---
    st.divider()
    if st.button("➕ Add to Parlay Leg", use_container_width=True):
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{'Over' if edge > 0 else 'Under'} {mkt_line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Added {p_name} to parlay!")

    # Performance Graph
    st.plotly_chart(px.line(p_data, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title=f"Performance Trend: {p_name}", markers=True), use_container_width=True)
else:
    st.error("The app could not find the 'player_name' column. This usually means the NFL data feed is down or being updated.")
