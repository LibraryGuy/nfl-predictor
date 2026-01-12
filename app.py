import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. SETTINGS & SESSION ---
st.set_page_config(page_title="NFL Genius: Pro Suite", layout="wide", page_icon="💰")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. THE DATA NORMALIZER (Fixes the AttributeError) ---
@st.cache_data(ttl=3600)
def load_and_clean_data():
    try:
        # Load latest 2025/2026 data
        raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Mapping dict to handle fluctuating nflverse column names
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player', 'displayName'],
            'team': ['recent_team', 'team_abbr', 'team'],
            'opponent': ['opponent_team', 'opponent', 'opp'],
            'rush_td': ['rushing_tds', 'rush_td'],
            'rec_td': ['receiving_tds', 'rec_td']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in raw.columns), None)
            if found:
                raw = raw.rename(columns={found: target})
            elif target not in raw.columns:
                raw[target] = "Unknown" if 'name' in target or 'team' in target else 0

        # DEFENSE ENGINE: Yards allowed by opponent per position
        def_stats = raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return raw.fillna(0), def_stats
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, def_data = load_and_clean_data()

# --- 3. THE INTERFACE ---
st.title("🏈 NFL Genius: Pro Parlay & Defense Suite")

if not data.empty:
    with st.sidebar:
        st.header("📋 Your Parlay")
        if st.session_state.parlay_legs:
            for i, leg in enumerate(st.session_state.parlay_legs):
                st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
            if st.button("Reset Parlay"):
                st.session_state.parlay_legs = []
                st.rerun()
        else:
            st.info("No legs added yet.")

    # Dashboard Columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Player Selection (AttributeError Fix)
        players = sorted([p for p in data['player_name'].unique() if p != "Unknown"])
        p_name = st.selectbox("Select Target Player", players)
        p_sub = data[data['player_name'] == p_name]
        p_pos = p_sub['position'].iloc[-1]
        
        # Defense Matchup Selection
        opp_list = sorted([o for o in data['opponent'].unique() if o != "Unknown"])
        target_def = st.selectbox("Select Opponent Defense", opp_list)
        
        # Calc Defense Multiplier
        d_impact = def_data[(def_data['def_team'] == target_def) & (def_data['position'] == p_pos)]
        # If defense allows > 55 yards avg to this pos, boost projection by 15%
        mod = 1.15 if (not d_impact.empty and d_impact['receiving_yards'].iloc[0] > 55) else 0.90

    with col2:
        st.subheader("🎯 Sharp Metrics")
        v_line = st.number_input("Sportsbook Line", value=50.5)
        
        # Projections
        avg_yds = p_sub['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_sub['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - v_line) / v_line) * 100
        
        # TD Probability (Poisson Distribution: 1 - e^-lambda)
        avg_td = (p_sub['rush_td'].mean() + p_sub['rec_td'].mean())
        td_prob = (1 - exp(-(avg_td * mod))) * 100
        
        st.metric("Model Proj", f"{proj_yds:.1f} Yds", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD Prob", f"{td_prob:.1f}%")

    # --- PARLAY BUILDER ACTION ---
    st.divider()
    if st.button("➕ Add to Parlay Builder", use_container_width=True):
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{'Over' if edge > 0 else 'Under'} {v_line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Locked in {p_name}!")

    # Performance Plot
    st.plotly_chart(px.line(p_sub, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title=f"Season Momentum: {p_name}", markers=True), use_container_width=True)

else:
    st.error("Data could not be loaded. Check your internet or 'nflreadpy' library version.")
