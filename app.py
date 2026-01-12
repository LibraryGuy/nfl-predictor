import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Genius Pro: Full Suite", layout="wide", page_icon="💰")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA ENGINE (FUZZY + DEFENSE) ---
@st.cache_data(ttl=3600)
def load_full_nfl_data():
    try:
        # Load 2024-2025 Stats
        raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Fuzzy Column Mapping
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player'],
            'team': ['recent_team', 'team_abbr', 'team'],
            'rush_td': ['rushing_tds', 'rush_td'],
            'rec_td': ['receiving_tds', 'rec_td'],
            'pass_td': ['passing_tds', 'pass_td']
        }
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in raw.columns), None)
            if found: raw = raw.rename(columns={found: target})
        
        # Calculate Defense Strength (EPA/Yards allowed per team)
        # In the new schema, we group by 'opponent' to see defensive performance
        def_stats = raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return raw.fillna(0), def_stats
    except Exception as e:
        st.error(f"Sync Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, def_data = load_full_nfl_data()

# --- 3. LOGIC ENGINES ---
def calc_td_prob(player_avg_td, def_epa_mod):
    """
    Calculates Anytime TD Probability using a Poisson Distribution.
    $P(X \ge 1) = 1 - e^{-\lambda}$
    """
    lam = player_avg_td * def_epa_mod
    prob = (1 - exp(-lam)) * 100
    return max(min(prob, 99.0), 1.0)

# --- 4. THE INTERFACE ---
st.title("🏈 NFL Genius: Pro Parlay Builder")

if not data.empty:
    with st.sidebar:
        st.header("🛒 Your Parlay")
        if st.session_state.parlay_legs:
            for i, leg in enumerate(st.session_state.parlay_legs):
                st.success(f"**{leg['player']}**: {leg['prop']} ({leg['edge']}% Edge)")
            if st.button("Clear Parlay"):
                st.session_state.parlay_legs = []
                st.rerun()
        else:
            st.write("No legs added yet.")

    # Main Dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        players = sorted(data['player_name'].unique())
        p_name = st.selectbox("Select Target Player", players)
        p_data = data[data['player_name'] == p_name]
        p_pos = p_data['position'].iloc[-1]
        p_team = p_data['team'].iloc[-1]
        
        opp_team = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        # Defense Impact Calculation
        def_impact = def_data[(def_data['def_team'] == opp_team) & (def_data['position'] == p_pos)]
        mod = 1.1 if not def_impact.empty and def_impact['receiving_yards'].iloc[0] > 60 else 0.95

    with col2:
        st.subheader("🔥 Prop Analytics")
        market_line = st.number_input("Market Yardage Line", value=55.5)
        
        # Prediction Logic
        avg_yds = p_data['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_data['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - market_line) / market_line) * 100
        
        # TD Prob Logic
        avg_td = (p_data['rush_td'].mean() + p_data['rec_td'].mean())
        td_p = calc_td_prob(avg_td, mod)
        
        st.metric("Proj. Yards", f"{proj_yds:.1f}", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD Prob", f"{td_p:.1f}%")

    # --- SHARP RECOMMENDATION ---
    st.divider()
    rec_col, btn_col = st.columns([3, 1])
    
    with rec_col:
        if edge > 15:
            st.info(f"**SHARP REC:** High value on **{p_name} OVER**. Defense allows high volume to {p_pos}s.")
        elif edge < -10:
            st.warning(f"**SHARP REC:** Significant value on **{p_name} UNDER**. Defensive matchup is elite.")
        else:
            st.write("Standard market pricing. No significant edge detected.")

    with btn_col:
        if st.button("➕ Add to Parlay", use_container_width=True):
            st.session_state.parlay_legs.append({
                "player": p_name,
                "prop": f"{'Over' if edge > 0 else 'Under'} {market_line} Yds",
                "edge": round(edge, 1)
            })
            st.toast(f"Added {p_name} to parlay!")

    # Performance Visual
    fig = px.bar(p_data, x='week', y=['rushing_yards', 'receiving_yards'], 
                 title=f"{p_name} - Historical Performance", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Could not load nflverse data. Please check your internet connection or API limits.")
