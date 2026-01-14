import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, time
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & CONSTANTS ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")
LEAGUE_AVG_PLAYS = 63.5  # Crucial for NBA-style Pace logic

# --- 2. DATA LOADING (STRENGTHENED & FIXED) ---
@st.cache_data(ttl=3600)
def load_data_pro():
    """
    Rewritten for safety: Handles Polars-to-Pandas conversion and 
    missing column issues which cause the 'Line 78' crash.
    """
    try:
        # 1. Fetch data for current and previous season
        raw_result = nfl.load_player_stats(seasons=[2024, 2025])
        
        # 2. POLARS SAFETY CHECK: nflreadpy returns Polars by default
        if hasattr(raw_result, 'to_pandas'):
            df = raw_result.to_pandas()
        else:
            df = pd.DataFrame(raw_result)
            
        if df.empty:
            st.error("Data Source returned an empty dataset.")
            return pd.DataFrame()

        # 3. COLUMN STANDARDIZATION
        # Map various API naming conventions to our internal logic
        name_map = {
            'player_display_name': 'player_name', 
            'player': 'player_name',
            'recent_team': 'team', 
            'opponent_team': 'opponent'
        }
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        
        # 4. DATA CLEANING
        required_stats = [
            'passing_yards', 'rushing_yards', 'receiving_yards', 
            'attempts', 'carries', 'targets', 
            'passing_tds', 'rushing_tds', 'receiving_tds'
        ]
        for col in required_stats:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0 # Create dummy column to prevent engine crash
                
        return df.dropna(subset=['player_name'])
    
    except Exception as e:
        st.error(f"Sync Failure in load_data_pro: {str(e)}")
        return pd.DataFrame()

# --- 3. INTELLIGENT MATCHUP & PACE LOGIC ---
def get_matchup_context(data, opponent, p_pos, stat_col):
    opp_def_stats = data[(data['opponent'] == opponent) & (data['position'] == p_pos)]
    if opp_def_stats.empty: return 1.0, "Neutral"
    
    league_avg = data[data['position'] == p_pos][stat_col].mean()
    opp_avg = opp_def_stats[stat_col].mean()
    m_ratio = opp_avg / league_avg if league_avg > 0 else 1.0
    status = "Shutdown" if m_ratio < 0.88 else "Vulnerable" if m_ratio > 1.12 else "Neutral"
    return round(m_ratio, 2), status

@st.cache_data
def get_pace_map(data):
    """NBA-style Pace Logic for NFL: Average plays run per team per game."""
    # Group by team and game to count total plays
    team_plays = data.groupby(['team', 'game_id']).size().reset_index(name='plays')
    return team_plays.groupby('team')['plays'].mean().to_dict()

# --- 4. CALCULATION ENGINE (PACE INTEGRATED) ---
def run_usage_monte_carlo(avg_volume, avg_efficiency, eff_std, m_mult, pace_mult, is_td, iterations=10000):
    # Adjusted Volume = (Season Avg Snaps * Weather Penalty) * Pace Factor
    adj_vol = avg_volume * pace_mult
    if adj_vol <= 0: return np.zeros(iterations)
    
    if is_td:
        # Defensive matchup directly impacts TD probability (Poisson Lambda)
        return np.random.poisson(adj_vol * m_mult, iterations)
    else:
        # Yardage = Volume * Efficiency (Matchup impacts Efficiency)
        sim_vol = np.random.poisson(adj_vol, iterations)
        mu = np.log(max(avg_efficiency * m_mult, 0.01)) - (0.4**2 / 2)
        sim_eff = np.random.lognormal(mu, 0.4, iterations)
        return sim_vol * sim_eff

# --- 5. INITIALIZE DATA ---
data = load_data_pro()
stadium_client = NFLStadiums()

# --- 6. UI RENDER ---
if not data.empty:
    pace_lookup = get_pace_map(data)
    
    with st.sidebar:
        st.header("🎯 Settings")
        bankroll = st.number_input("Purse ($)", value=1000)
        kelly_fraction = st.slider("Kelly Risk", 0.1, 1.0, 0.25)
        st.divider()
        selected_p = st.selectbox("Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent", sorted(data['opponent'].unique()))
        selected_market = st.radio("Market", ["Yards", "Touchdowns"])
        market_line = st.number_input("Line", value=0.5 if selected_market == "Touchdowns" else 50.0)
        
    # --- LOGIC INTEGRATION ---
    p_df = data[data['player_name'] == selected_p].copy()
    p_pos = p_df['position'].iloc[-1]
    p_team = p_df['team'].iloc[-1]
    
    # Calculate PACE Factor
    t_pace = pace_lookup.get(p_team, LEAGUE_AVG_PLAYS)
    o_pace = pace_lookup.get(selected_opp, LEAGUE_AVG_PLAYS)
    matchup_pace = (t_pace + o_pace) / 2
    pace_mult = matchup_pace / LEAGUE_AVG_PLAYS
    
    # Run Engine
    stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if selected_market == "Yards" else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
    vol_col = 'attempts' if p_pos == 'QB' else 'carries' if p_pos == 'RB' else 'targets'
    
    m_ratio, m_status = get_matchup_context(data, selected_opp, p_pos, stat_col)
    sim_results = run_usage_monte_carlo(p_df[vol_col].mean(), (p_df[stat_col]/p_df[vol_col].replace(0,1)).mean(), 0.4, m_ratio, pace_mult, (selected_market == "Touchdowns"))
    
    # Render Metrics
    st.title(f"📊 {selected_p} Insights")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Proj", round(np.mean(sim_results), 1))
    m2.metric("Pace Mult", f"{round(pace_mult, 2)}x", f"{round(matchup_pace, 1)} Plays/G")
    m3.metric("Matchup", m_status, f"{m_ratio}x Adj")
    
    st.plotly_chart(go.Figure(go.Histogram(x=sim_results, marker_color='#00ff96')).update_layout(template="plotly_dark"))
else:
    st.error("Unable to load data. Ensure nflreadpy and pandas are installed.")
