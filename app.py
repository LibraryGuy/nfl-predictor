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

# --- 2. THE CLEAN SWEEP DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_and_fix_data():
    try:
        # Load 2025/2026 Stats
        raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. FLATTEN MULTI-INDEX HEADERS
        # This is the likely cause of your AttributeError. 
        # It turns ('offense', 'player_name') into just 'player_name'.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[1] if col[1] else col[0] for col in raw.columns.values]
        
        # B. 2026 SCHEMA MAPPING
        # Mapping variants to a standard set of names your UI expects
        rename_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent',
            'rushing_tds': 'rush_td',
            'receiving_tds': 'rec_td'
        }
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

        # C. DEFENSE ANALYTICS ENGINE
        # Grouping by opponent and position to find "leaky" defenses
        def_stats = raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return raw.fillna(0), def_stats
    except Exception as e:
        # If this fails, return empty DataFrames so the rest of the app doesn't crash
        return pd.DataFrame(columns=['player_name', 'opponent', 'position']), pd.DataFrame()

# Execute loading
data, def_data = load_and_fix_data()

# --- 3. ANALYTICS ---
def get_td_prob(avg_td, match_mod):
    # Poisson probability: P(at least 1 TD) = 1 - e^(-lambda)
    lam = avg_td * match_mod
    return (1 - exp(-lam)) * 100

# --- 4. THE INTERFACE ---
st.title("🏈 NFL Genius: Pro Build")

# SIDEBAR: Placed early so it renders even if data is still loading
with st.sidebar:
    st.header("📋 Parlay Builder")
    if st.session_state.parlay_legs:
        for leg in st.session_state.parlay_legs:
            st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
        if st.button("Reset Parlay"):
            st.session_state.parlay_legs = []
            st.rerun()
    else:
        st.info("No legs added yet.")

# Main Page Logic
if not data.empty and 'player_name' in data.columns:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Player Selection (Bracket notation is safer than dot notation)
        players = sorted(data['player_name'].unique())
        p_name = st.selectbox("Select Target Player", players)
        p_sub = data[data['player_name'] == p_name]
        p_pos = p_sub['position'].iloc[-1]
        
        # Defense Selection
        opp_list = sorted(data['opponent'].unique())
        target_def = st.selectbox("Vs. Defense", opp_list)
        
        # Calculate Defense Modifier
        d_sub = def_data[(def_data['def_team'] == target_def) & (def_data['position'] == p_pos)]
        # If defense allows >55 yards avg to this pos, apply a 15% "boost"
        mod = 1.15 if (not d_sub.empty and d_sub['receiving_yards'].iloc[0] > 55) else 0.90

    with col2:
        st.subheader("📊 Sharp Metrics")
        v_line = st.number_input("Vegas Line", value=50.5)
        
        # Yardage Proj
        avg_yds = p_sub['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_sub['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - v_line) / v_line) * 100
        
        # TD Proj
        avg_td = (p_sub['rush_td'].mean() + p_sub['rec_td'].mean())
        td_prob = get_td_prob(avg_td, mod)
        
        st.metric("Model Proj", f"{proj_yds:.1f} Yds", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD %", f"{td_prob:.1f}%")

    st.divider()
    if st.button("➕ Add to Parlay", use_container_width=True):
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{'Over' if edge > 0 else 'Under'} {v_line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Locked in {p_name}!")

    # Performance Visual
    st.plotly_chart(px.line(p_sub, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title=f"Trend: {p_name}", markers=True), use_container_width=True)
else:
    st.warning("🔄 Data Sync in progress... Please refresh in a moment.")
