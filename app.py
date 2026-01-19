import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson, lognorm

# --- 1. DATA ACCESS & DEFENSIVE MODELING ---
@st.cache_data(ttl=3600)
def get_advanced_data():
    """Loads current season stats and calculates defensive strengths."""
    try:
        curr_season = nfl.get_current_season()
        # Load player stats (weekly)
        raw_stats = nfl.load_player_stats(seasons=[curr_season], summary_level='week').to_pandas()
        
        # Load team stats to calculate Defense vs Position (DvP)
        team_stats = nfl.load_team_stats(seasons=[curr_season]).to_pandas()
        
        # We need to find 'opponent_team' defensive averages
        # Columns: 'opp_passing_yards', 'opp_rushing_yards', etc.
        def_cols = {
            'passing_yards': 'opp_passing_yards',
            'rushing_yards': 'opp_rushing_yards',
            'passing_tds': 'opp_pass_tds',
            'rushing_tds': 'opp_rush_tds'
        }
        
        # Create defensive lookup table
        def_lookup = team_stats.groupby('team').mean(numeric_only=True)
        league_means = team_stats.mean(numeric_only=True)
        
        return raw_stats, def_lookup, league_means
    except Exception as e:
        st.error(f"Engine Data Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

@st.cache_data(ttl=3600)
def get_registry():
    """Maps display names to GSIS IDs to solve the 'First Initial' problem."""
    players = nfl.load_players().to_pandas()
    active = players[players['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    return {f"{row['display_name']} ({row['position']})": row['gsis_id'] for _, row in active.iterrows()}

# --- 2. MATCHUP LOGIC ---
def calculate_matchup_multiplier(opponent, stat_type, def_lookup, league_means):
    """Returns a float representing the defensive difficulty (e.g., 1.15 for weak def)."""
    # Map player stat to the defensive column in team_stats
    mapping = {
        'passing_yards': 'opp_passing_yards',
        'rushing_yards': 'opp_rushing_yards',
        'receiving_yards': 'opp_passing_yards', # WRs/TEs face Pass Def
        'passing_tds': 'opp_pass_tds',
        'rushing_tds': 'opp_rush_tds',
        'receiving_tds': 'opp_pass_tds'
    }
    
    col = mapping.get(stat_type)
    if opponent in def_lookup.index and col in def_lookup.columns:
        opp_avg = def_lookup.loc[opponent, col]
        lg_avg = league_means[col]
        return opp_avg / lg_avg if lg_avg > 0 else 1.0
    return 1.0

# --- 3. UI & SIMULATION ---
st.set_page_config(page_title="NFL Sharp Predictor v3", layout="wide")

# Load Data
registry = get_registry()
raw_stats, def_lookup, lg_means = get_advanced_data()

# Sidebar Setup
st.sidebar.title("🏈 Model Controls")
selected_label = st.sidebar.selectbox("Target Player", list(registry.keys()))
player_id = registry[selected_label]

if player_id and not raw_stats.empty:
    # Filter for the specific player using ID (solves name mismatch)
    p_df = raw_stats[raw_stats['player_id'] == player_id].copy()
    
    if not p_df.empty:
        pos = p_df['position'].iloc[-1]
        
        # Market Selection
        market_map = {
            'QB': ['passing_yards', 'passing_tds'],
            'RB': ['rushing_yards', 'rushing_tds'],
            'WR': ['receiving_yards', 'receiving_tds'],
            'TE': ['receiving_yards', 'receiving_tds']
        }
        stat_col = st.sidebar.selectbox("Market", market_map.get(pos, ['receiving_yards']))
        line = st.sidebar.number_input("Sportsbook Line", value=45.5 if "yards" in stat_col else 0.5)
        
        # NEXT OPPONENT (Logic to find next team or manual override)
        teams = sorted(def_lookup.index.tolist())
        opp = st.sidebar.selectbox("Next Opponent", teams)
        
        # Multiplier Calculation
        match_mult = calculate_matchup_multiplier(opp, stat_col, def_lookup, lg_means)
        
        # --- MONTE CARLO ENGINE ---
        iterations = 10000
        base_avg = p_df[stat_col].mean()
        adj_avg = base_avg * match_mult
        
        if "tds" in stat_col:
            sims = np.random.poisson(max(adj_avg, 0.01), iterations)
        else:
            std = p_df[stat_col].std() if p_df[stat_col].std() > 0 else (base_avg * 0.4)
            sigma = np.sqrt(np.log(1 + (std**2 / (adj_avg**2 + 1e-9))))
            mu = np.log(adj_avg + 1e-9) - (sigma**2 / 2)
            sims = np.random.lognormal(mu, sigma, iterations)

        # UI LAYOUT
        st.title(f"Model: {selected_label}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Matchup Multiplier", f"{match_mult:.2f}x", delta=f"{match_mult-1:.2f}", delta_color="inverse")
        m2.metric("Projected Median", f"{np.median(sims):.1f}")
        m3.metric("Over Probability", f"{(np.sum(sims > line)/10000)*100:.1f}%")

        # Visuals
        fig = go.Figure(go.Histogram(x=sims, nbinsx=40, marker_color='#00f2ff', opacity=0.6))
        fig.add_vline(x=line, line_color="red", line_dash="dash", annotation_text="Line")
        fig.update_layout(title=f"10,000 Iterations: {stat_col.replace('_',' ')} vs {opp}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Form Table
        st.subheader("Recent Form (2025-26)")
        st.dataframe(p_df[['week', 'opponent_team', stat_col]].tail(5), use_container_width=True, hide_index=True)
    else:
        st.warning(f"No game logs found for ID {player_id}. Player may be inactive this season.")
