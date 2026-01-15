import streamlit as st
import pandas as pd
import numpy as np
import nflreadpy as nfl
import plotly.graph_objects as go
from scipy.stats import poisson, lognorm

# --- 1. SETTINGS & REFINED CONSTANTS ---
st.set_page_config(page_title="NFL Sharp Pro v1.0", layout="wide", page_icon="🏈")

# --- 2. ROBUST DATA LOADING (Memory Optimized) ---
@st.cache_data(ttl=3600)
def load_nfl_data():
    try:
        # Load only 2024-2025 to keep the app fast and stable
        # nflreadpy returns Polars by default; we convert to Pandas for your logic
        raw_df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Mapping NFL columns to match your "Stat Category" logic
        name_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent'
        }
        df = raw_df.rename(columns=name_map)
        
        # Fill NaNs in stats to avoid math errors
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets', 'passing_tds', 'rushing_tds', 'receiving_tds']
        df[stat_cols] = df[stat_cols].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Data Sync Failure: {e}")
        return pd.DataFrame()

# --- 3. CORE LOGIC ENGINE (NBA-Enhanced) ---

def calculate_nfl_usage(p_df, pos, injury_list):
    """Ported from your NBA 'Dynamic Usage' logic"""
    boost = 1.0
    # In NFL, if a lead WR is out, the others get a 'Target Share' boost
    # This simulates your Ja Morant/Donovan Mitchell logic for football
    if pos in ['WR', 'TE'] and any(player in injury_list for player in ["Tyreek Hill", "Justin Jefferson"]): 
        boost += 0.15
    return boost

def get_dvp_multiplier(pos, opp):
    """NBA-style Defense vs Position (DvP) mapping"""
    dvp_map = {
        'QB': {'CLE': 0.85, 'NYJ': 0.88, 'JAC': 1.25},
        'RB': {'BAL': 0.78, 'KC': 0.82, 'DAL': 1.30},
        'WR': {'HOU': 0.84, 'PHI': 0.90, 'WAS': 1.18}
    }
    return dvp_map.get(pos, {}).get(opp, 1.0)

# --- 4. THE PROJECTION ENGINE ---

def run_monte_carlo(mu, line, is_td):
    """
    UPGRADE: Unlike NBA points, NFL yards have high 'Skew'.
    We use Log-Normal for yards to account for the 'Big Play' tail.
    """
    iterations = 10000
    if is_td:
        # Touchdowns are discrete events (Poisson)
        return np.random.poisson(mu, iterations)
    else:
        # Yards are continuous with explosive potential (Log-Normal)
        sigma = 0.45  # Volatility constant
        return np.random.lognormal(np.log(mu) - (sigma**2 / 2), sigma, iterations)

# --- 5. MAIN UI ---
df = load_nfl_data()

if not df.empty:
    with st.sidebar:
        st.title("🚀 NFL Sharp Pro")
        player_list = sorted(df['player_name'].unique())
        selected_p = st.selectbox("Select Player", player_list)
        
        # Filter for player info
        p_df = df[df['player_name'] == selected_p]
        p_pos = p_df['position'].iloc[-1]
        p_team = p_df['team'].iloc[-1]
        
        market = st.radio("Market", ["Yards", "Touchdowns"])
        
        # Match NBA-style category selection
        if p_pos == 'QB': stat = 'passing_yards' if market == "Yards" else 'passing_tds'
        elif p_pos == 'RB': stat = 'rushing_yards' if market == "Yards" else 'rushing_tds'
        else: stat = 'receiving_yards' if market == "Yards" else 'receiving_tds'
        
        line = st.number_input("Market Line", value=50.0 if market == "Yards" else 0.5)
        recency_weight = st.slider("Recency Bias", 0.0, 1.0, 0.3)

    # Calculation logic
    opp = p_df['opponent'].iloc[-1]
    dvp_m = get_dvp_multiplier(p_pos, opp)
    usage_m = calculate_nfl_usage(p_df, p_pos, [])
    
    # Weighted Average (Last 3 vs Season)
    avg_full = p_df[stat].mean()
    avg_recent = p_df[stat].tail(3).mean()
    mu = ((avg_recent * recency_weight) + (avg_full * (1 - recency_weight))) * dvp_m * usage_m
    
    # Run Simulations
    sims = run_monte_carlo(mu, line, market == "Touchdowns")
    win_prob = (np.sum(sims > line) / 10000) * 100

    # Render
    st.title(f"🏈 {selected_p} vs {opp}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Projection", round(mu, 1))
    c2.metric("Win Probability", f"{round(win_prob, 1)}%")
    c3.metric("Matchup Edge", f"{round(dvp_m, 2)}x")

    fig = go.Figure(data=[go.Histogram(x=sims, marker_color='#00CC96', opacity=0.7)])
    fig.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="Market Line")
    fig.update_layout(template="plotly_dark", title="Simulated Outcomes (10,000 runs)")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Wait... The NFL data is still loading or the connection was refused. Check your logs.")
