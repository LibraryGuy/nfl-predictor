import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Genius Pro", layout="wide", page_icon="🏈")
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE BULLETPROOF DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_full_nfl_data():
    try:
        # Load 2024-2025 Stats
        raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. COLUMN MAPPING (Standardizing for 2026 schema)
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player'],
            'team': ['recent_team', 'team_abbr', 'team'],
            'opponent': ['opponent_team', 'opponent', 'opp'],
            'rush_td': ['rushing_tds', 'rush_td'],
            'rec_td': ['receiving_tds', 'rec_td']
        }
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in raw.columns), None)
            if found: raw = raw.rename(columns={found: target})
        
        # B. DEFENSE ENGINE: Calculate yards allowed per position
        # This creates the "Defense Option" you lost
        def_stats = raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return raw.fillna(0), def_stats
    except Exception as e:
        st.error(f"Critical Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, def_data = load_full_nfl_data()

# --- 3. ANALYTICS HELPERS ---
def get_td_prob(avg_td, match_mod):
    """Calculates Anytime TD Probability ($1 - e^{-\lambda}$)"""
    lam = avg_td * match_mod
    return (1 - exp(-lam)) * 100

# --- 4. INTERFACE ---
st.title("💰 NFL Genius: Parlay & Prop Suite")

if not data.empty:
    with st.sidebar:
        st.header("📋 Parlay Builder")
        if st.session_state.parlay_legs:
            for leg in st.session_state.parlay_legs:
                st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
            if st.button("Reset Parlay"):
                st.session_state.parlay_legs = []
                st.rerun()
        else:
            st.info("Add a leg to start building.")

    # Main Selection Row
    col1, col2 = st.columns([2, 1])
    with col1:
        p_name = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        p_sub = data[data['player_name'] == p_name]
        p_pos = p_sub['position'].iloc[-1]
        
        # Defense Selection
        opp_list = sorted(data['opponent'].unique())
        target_def = st.selectbox("Vs. Defense", opp_list)
        
        # Calculation Defense Modifier
        d_sub = def_data[(def_data['def_team'] == target_def) & (def_data['position'] == p_pos)]
        mod = 1.15 if (not d_sub.empty and d_sub['receiving_yards'].iloc[0] > 55) else 0.90

    with col2:
        st.subheader("📊 Model Projections")
        v_line = st.number_input("Vegas Line", value=60.5)
        
        # Yardage Logic
        avg_yds = p_sub['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_sub['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - v_line) / v_line) * 100
        
        # TD Logic
        td_prob = get_td_prob((p_sub['rush_td'].mean() + p_sub['rec_td'].mean()), mod)
        
        st.metric("Proj. Yards", f"{proj_yds:.1f}", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD %", f"{td_prob:.1f}%")

    # --- ACTION BUTTONS ---
    st.divider()
    if st.button("➕ Add to Parlay Leg"):
        pick_type = "Over" if edge > 0 else "Under"
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{pick_type} {v_line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Added {p_name} to parlay!")

    # Performance Visual
    st.plotly_chart(px.line(p_sub, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title="Season Trend", markers=True), use_container_width=True)
else:
    st.warning("Data loading... If this persists, verify your 'nflreadpy' installation.")
