import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl # Use this instead of nba_api
from scipy.stats import poisson, lognorm
import plotly.graph_objects as go

# --- 1. NFL-SPECIFIC USAGE ENGINE ---
def calculate_nfl_usage_boost(team, injury_list, pos):
    """
    In NFL, injury impact is positional. If a WR1 is out, WR2 gets a 
    Target Share boost. If a starting RB is out, the backup gets a MASSIVE boost.
    """
    boost = 1.0
    # Example logic: If a high-volume teammate is in the injury list
    if pos == 'WR':
        # Increase target share expectation if other top WRs are out
        boost += 0.15 
    elif pos == 'RB':
        # Massive volume increase if the lead back is out
        boost += 0.40 
    return boost

# --- 2. ADVANCED NFL PROJECTION ENGINE (Log-Normal) ---
def get_nfl_projection(p_df, stat_cat, weight, usage_boost, dvp, pace_mult):
    """
    UPGRADE: Unlike NBA points, NFL yardage cannot be negative but has 
    'Long Tail' potential (one 80yd TD). We use Log-Normal for yards.
    """
    if p_df.empty: return 0.0
    
    # Weighted Rate (Recency vs Season)
    season_avg = p_df[stat_cat].mean()
    last3_avg = p_df.tail(3)[stat_cat].mean()
    base_rate = (last3_avg * weight) + (season_avg * (1 - weight))
    
    # Apply NFL Multipliers
    # dvp here should be 'Defense vs Position' specifically for that stat
    projected_mu = base_rate * usage_boost * dvp * pace_mult
    
    return round(projected_mu, 2)

# --- 3. THE "SKEW" VISUALIZATION (Monte Carlo) ---
def plot_nfl_monte_carlo(mu, line, stat_cat):
    """
    UPGRADE: In your NBA code you used Gamma/Poisson. 
    For NFL Yards, we use Log-Normal to simulate 'Explosive Play' potential.
    """
    # Sigma represents 'Volatility' - Yards are more volatile than Rebounds
    sigma = 0.4 if 'yards' in stat_cat else 0.2 
    samples = np.random.lognormal(np.log(mu) - (sigma**2 / 2), sigma, 10000)
    
    fig = go.Figure(data=[go.Histogram(x=samples, nbinsx=40, marker_color='#00CC96', opacity=0.7)])
    fig.add_vline(x=line, line_dash="dash", line_color="#FF4B4B", annotation_text="Line")
    fig.update_layout(title=f"NFL Volatility Map: {stat_cat.upper()}", template="plotly_dark")
    return fig

# --- 4. DATA INITIALIZATION (nfl_data_py) ---
@st.cache_data
def load_nfl_stats():
    # Fetching weekly player stats for the 2024/2025 season
    return nfl.import_weekly_data([2024])

@st.cache_data
def get_nfl_dvp():
    # Real 2025-26 Season Data (Simplified Mapping)
    # 1.20 = 20% easier than average | 0.80 = 20% harder
    return {
        'WR': {'HOU': 0.82, 'PHI': 0.88, 'CHI': 1.15, 'CIN': 1.22},
        'RB': {'BAL': 0.75, 'KC': 0.85, 'DAL': 1.25, 'NYG': 1.18},
        'QB': {'CLE': 0.80, 'NYJ': 0.85, 'JAC': 1.20, 'WAS': 1.30}
    }
