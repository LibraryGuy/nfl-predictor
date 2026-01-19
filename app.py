import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson, lognorm

# --- 1. THE REGISTRY (Lightweight Name Fetcher) ---
@st.cache_data(ttl=3600)
def get_active_player_list():
    """Loads only rosters to avoid memory crashes and column errors on boot."""
    try:
        # Load 2024 rosters for the dropdown
        roster = nfl.load_rosters(seasons=[2024]).to_pandas()
        
        # NFLVerse roster column is typically 'full_name' or 'player_name'
        name_col = next((c for c in ['full_name', 'player_name', 'p_name'] if c in roster.columns), None)
        
        if not name_col:
            st.error(f"Roster Error: Name column not found. Available: {roster.columns.tolist()}")
            return []
            
        # Filter for fantasy-relevant positions
        skill_df = roster[roster['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        return sorted(skill_df[name_col].dropna().unique().tolist())
    except Exception as e:
        st.sidebar.error(f"Registry Sync Error: {e}")
        return []

# --- 2. THE ANALYTICS ENGINE (Deep Data Loading) ---
@st.cache_data(ttl=3600)
def get_player_deep_stats(player_name):
    """Loads detailed game logs only for the chosen player."""
    try:
        # Load stats for the prediction years
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Standardize the player name column immediately
        name_map = {'player_display_name': 'player_name', 'player': 'player_name'}
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        
        if 'player_name' not in df.columns:
            return pd.DataFrame()
            
        return df[df['player_name'] == player_name].copy()
    except Exception:
        return pd.DataFrame()

# --- 3. THE PREDICTION LOGIC ---
def run_monte_carlo(data_series, market_type, matchup_adj=1.0):
    """Professional 10,000-iteration sim tailored to stat distributions."""
    iterations = 10000
    avg = data_series.mean() * matchup_adj
    
    if "TD" in market_type:
        # Poisson is best for discrete 'counting' events like Touchdowns
        return np.random.poisson(max(avg, 0.01), iterations)
    else:
        # Lognormal is best for yardage (avoids negatives, allows 'big play' outliers)
        std = data_series.std() if data_series.std() > 0 else (avg * 0.4)
        sigma = np.sqrt(np.log(1 + (std**2 / (avg**2 + 1e-9))))
        mu = np.log(avg + 1e-9) - (sigma**2 / 2)
        return np.random.lognormal(mu, sigma, iterations)

# --- 4. DASHBOARD UI ---
st.set_page_config(page_title="NFL Sharp Predictor", layout="wide")
st.sidebar.title("🏈 Sharp Intel")

player_list = get_active_player_list()
selected_p = st.sidebar.selectbox("Select Player", player_list)

if selected_p:
    # LAZY LOAD: We only go get the heavy stats now
    p_data = get_player_deep_stats(selected_p)
    
    if not p_data.empty:
        st.title(f"📊 {selected_p} Intelligence")
        
        # Dynamic Market Selection based on position
        pos = p_data['position'].iloc[-1]
        market_map = {
            'QB': ['Passing Yards', 'Passing TDs'],
            'RB': ['Rushing Yards', 'Rushing TDs'],
            'WR': ['Receiving Yards', 'Receiving TDs'],
            'TE': ['Receiving Yards', 'Receiving TDs']
        }
        market = st.sidebar.selectbox("Market", market_map.get(pos, ['Receiving Yards']))
        stat_col = market.lower().replace(" ", "_")
        line = st.sidebar.number_input("Sportsbook Line", value=0.5 if "TD" in market else 50.0)

        # Run Prediction
        sims = run_monte_carlo(p_data[stat_col], market)
        win_prob = (np.sum(sims > line) / 10000) * 100

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected Mean", f"{np.mean(sims):.1f}")
        c2.metric("Win Probability", f"{win_prob:.1f}%")
        c3.metric("Volatility (StdDev)", f"{p_data[stat_col].std():.1f}")

        # Visualization
        fig = go.Figure(go.Histogram(x=sims, nbinsx=30, marker_color='#00ff96'))
        fig.add_vline(x=line, line_dash="dash", line_color="red")
        fig.update_layout(title=f"10,000 Iteration Result: {market}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not find historical stats for {selected_p} in 2024/2025.")
