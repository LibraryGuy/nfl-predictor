import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import lognorm, poisson

# --- 1. DATA RECOVERY & STANDARDIZATION ---
@st.cache_data(ttl=3600)
def load_base_data():
    try:
        # Load 2024-2025 stats
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        # Robust column mapping
        mapping = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # Ensure numeric types for modeling
        stats = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'attempts', 'targets']
        for s in stats:
            if s in df.columns:
                df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Data Sync Failed: {e}")
        return pd.DataFrame()

# --- 2. THE ENGINE: MONTE CARLO SIMULATOR ---
def run_advanced_simulation(player_stats, matchup_mult, weather_mult, market_type):
    """
    Runs 10,000 iterations using distribution curves tailored to NFL stat types.
    """
    iterations = 10000
    
    if "Yards" in market_type:
        # Yards are modeled with Lognormal distribution (cannot be negative, right-skewed)
        # We calculate the player's mean and variance (volatility)
        mean_val = player_stats.mean() * matchup_mult * weather_mult
        std_val = player_stats.std() if player_stats.std() > 0 else (mean_val * 0.4)
        
        # Convert to Lognormal parameters (mu and sigma)
        sigma = np.sqrt(np.log(1 + (std_val**2 / (mean_val**2 + 1e-9))))
        mu = np.log(mean_val + 1e-9) - (sigma**2 / 2)
        
        sims = np.random.lognormal(mu, sigma, iterations)
    else:
        # Touchdowns are discrete events, modeled with Poisson
        lam = max(0.01, player_stats.mean() * matchup_mult * weather_mult)
        sims = np.random.poisson(lam, iterations)
        
    return sims

# --- 3. UI & ANALYTICS ---
data = load_base_data()

if not data.empty:
    st.sidebar.header("🔍 Intelligence Setup")
    
    # Selection Guard
    all_players = sorted(data['player_name'].unique())
    selected_player = st.sidebar.selectbox("Target Player", all_players)
    
    # Filter specific player data
    p_df = data[data['player_name'] == selected_player].copy()
    p_pos = p_df['position'].iloc[-1]
    
    # Market Logic
    if p_pos == 'QB':
        market = st.sidebar.selectbox("Market", ["Passing Yards", "Passing TDs"])
        stat_col = 'passing_yards' if "Yards" in market else 'passing_tds'
    elif p_pos == 'RB':
        market = st.sidebar.selectbox("Market", ["Rushing Yards", "Rushing TDs"])
        stat_col = 'rushing_yards' if "Yards" in market else 'rushing_tds'
    else:
        market = st.sidebar.selectbox("Market", ["Receiving Yards", "Receiving TDs"])
        stat_col = 'receiving_yards' if "Yards" in market else 'receiving_tds'

    line = st.sidebar.number_input("Sportsbook Line", value=50.0 if "Yards" in market else 0.5)

    # Contextual Multipliers (In a production app, these pull from a DvP API)
    st.sidebar.subheader("Adjustments")
    matchup_adj = st.sidebar.slider("Matchup Strength (DvP)", 0.70, 1.30, 1.00, help="0.9 = Strong Defense, 1.1 = Weak Defense")
    weather_adj = st.sidebar.slider("Weather Impact", 0.80, 1.00, 1.00, help="Wind/Rain reduction factor")

    # --- EXECUTION ---
    sim_results = run_advanced_simulation(p_df[stat_col], matchup_adj, weather_adj, market)
    
    # Probability Math
    win_prob = (np.sum(sim_results > line) / 10000) * 100
    expected_val = np.mean(sim_results)
    edge = ((win_prob/100) * (1.91)) - 1 # Simple Edge calc for -110 odds

    # Display Results
    st.title(f"🏈 {selected_player} ({p_pos})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Mean", f"{expected_val:.1f}")
    col2.metric("Win Probability", f"{win_prob:.1f}%")
    col3.metric("Est. Edge", f"{edge*100:.1f}%", delta_color="normal")

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim_results, name='Simulated Outcomes', marker_color='#00f2ff', opacity=0.75))
    fig.add_vline(x=line, line_dash="dash", line_color="red", annotation_text=f"Line: {line}")
    fig.update_layout(title=f"10,000 Iteration Distribution: {market}", template="plotly_dark", xaxis_title=market)
    st.plotly_chart(fig, use_container_width=True)

    # Probability Ladder
    st.subheader("🎯 Probability Ladder")
    targets = [expected_val * 0.75, expected_val, expected_val * 1.25, expected_val * 1.5] if "Yards" in market else [0.5, 1.5, 2.5]
    ladder = [{"Target": f"{round(t,1)}+", "Prob": f"{(np.sum(sim_results >= t)/10000)*100:.1f}%"} for t in targets]
    st.table(pd.DataFrame(ladder))

else:
    st.warning("Awaiting Data Stream...")
