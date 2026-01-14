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
LEAGUE_AVG_PLAYS = 63.5 

# --- 2. DATA LOADING (POLARS-TO-PANDAS + GAME_ID FIX) ---
@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        # Fetch 2024 (historical) and 2025 (current/recent) data
        raw_result = nfl.load_player_stats(seasons=[2024, 2025])
        
        # Polars Safety Guard
        if hasattr(raw_result, 'to_pandas'):
            df = raw_result.to_pandas()
        else:
            df = pd.DataFrame(raw_result)
            
        if df.empty:
            st.warning("Data Source returned an empty dataset.")
            return pd.DataFrame()

        # Standardizing Column Names (Mapping nflverse keys to Dashboard keys)
        name_map = {
            'player_display_name': 'player_name', 
            'player': 'player_name',
            'recent_team': 'team', 
            'opponent_team': 'opponent',
            'gsis_game_id': 'game_id'  # CRITICAL FIX for the KeyError
        }
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        
        # Robust Game ID Fallback
        if 'game_id' not in df.columns:
            # Create a unique string if the API didn't provide one
            df['game_id'] = (df['season'].astype(str) + "_" + 
                             df['week'].astype(str) + "_" + 
                             df['team'] + "_" + df['opponent'])

        # Numeric Enforcement for Analytics Engine
        required_stats = [
            'passing_yards', 'rushing_yards', 'receiving_yards', 
            'attempts', 'carries', 'targets', 
            'passing_tds', 'rushing_tds', 'receiving_tds'
        ]
        for col in required_stats:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0 
                
        return df.dropna(subset=['player_name'])
    
    except Exception as e:
        st.error(f"Critical Sync Failure: {str(e)}")
        return pd.DataFrame()

# --- 3. ADVANCED ANALYTICS FUNCTIONS ---
@st.cache_data
def get_pace_map(data):
    """Calculates team play volume per game (The 'NBA Pace' logic)."""
    if 'game_id' not in data.columns:
        return {team: LEAGUE_AVG_PLAYS for team in data['team'].unique()}
    
    # Count rows per team/game to find total offensive plays run
    team_plays = data.groupby(['team', 'game_id']).size().reset_index(name='plays')
    return team_plays.groupby('team')['plays'].mean().to_dict()

def get_matchup_context(data, opponent, p_pos, stat_col):
    opp_def_stats = data[(data['opponent'] == opponent) & (data['position'] == p_pos)]
    if opp_def_stats.empty: return 1.0, "Neutral"
    
    league_avg = data[data['position'] == p_pos][stat_col].mean()
    opp_avg = opp_def_stats[stat_col].mean()
    m_ratio = opp_avg / league_avg if league_avg > 0 else 1.0
    status = "Shutdown" if m_ratio < 0.88 else "Vulnerable" if m_ratio > 1.12 else "Neutral"
    return round(m_ratio, 2), status

def run_usage_monte_carlo(avg_vol, avg_eff, m_mult, pace_mult, is_td, iterations=10000):
    # Core Engine: Combines Season Avg Volume * Pace Multiplier
    adj_vol = avg_vol * pace_mult
    if adj_vol <= 0: return np.zeros(iterations)
    
    if is_td:
        # TD Logic uses Poisson centered on Matchup-Adjusted Volume
        return np.random.poisson(adj_vol * m_mult, iterations)
    else:
        # Yardage Logic: Volume (Poisson) * Efficiency (Log-Normal)
        sim_vol = np.random.poisson(adj_vol, iterations)
        # Matchup multiplier specifically affects efficiency (yards per attempt)
        mu = np.log(max(avg_eff * m_mult, 0.01)) - (0.4**2 / 2)
        sim_eff = np.random.lognormal(mu, 0.4, iterations)
        return sim_vol * sim_eff

def generate_prob_ladder(sim_results, is_td):
    thresholds = [1, 2, 3] if is_td else [25, 50, 75, 100, 125, 150]
    ladder = []
    for t in thresholds:
        prob = (np.sum(sim_results >= t) / len(sim_results)) * 100
        ladder.append({"Threshold": f"{t}+", "Prob": f"{prob:.1f}%"})
    return pd.DataFrame(ladder)

# --- 4. DASHBOARD INITIALIZATION ---
data = load_data_pro()
stadium_client = NFLStadiums()

if not data.empty:
    pace_lookup = get_pace_map(data)
    
    with st.sidebar:
        st.header("🛡️ Strategy & Risk")
        bankroll = st.number_input("Total Bankroll ($)", value=1000)
        kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25)
        st.divider()
        
        # Player Selection Logic
        player_list = sorted(data['player_name'].unique())
        selected_p = st.selectbox("Search Player", player_list)
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if 'position' in p_df.columns else "WR"
        p_team = p_df['team'].iloc[-1]
        
        opp_list = sorted(data['opponent'].unique())
        selected_opp = st.selectbox("Vs. Defense", opp_list)
        
        market_type = st.radio("Market", ["Yards", "Touchdowns"])
        market_line = st.number_input("Sportsbook Line", value=50.0 if market_type == "Yards" else 0.5)

    # --- 5. ANALYTICAL CALCULATION ---
    # Determine Pace Multiplier (Matchup Pace vs League Avg)
    t_pace = pace_lookup.get(p_team, LEAGUE_AVG_PLAYS)
    o_pace = pace_lookup.get(selected_opp, LEAGUE_AVG_PLAYS)
    matchup_pace = (t_pace + o_pace) / 2
    pace_mult = matchup_pace / LEAGUE_AVG_PLAYS

    # Define Stats for specific position/market
    if market_type == "Yards":
        stat_col = 'passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards'
        vol_col = 'attempts' if p_pos == 'QB' else 'carries' if p_pos == 'RB' else 'targets'
        is_td = False
    else:
        stat_col = 'passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds'
        vol_col = 'attempts' if p_pos == 'QB' else 'carries' if p_pos == 'RB' else 'targets'
        is_td = True

    m_ratio, m_status = get_matchup_context(data, selected_opp, p_pos, stat_col)
    
    # Efficiency calculation (avoiding divide-by-zero)
    avg_vol = p_df[vol_col].mean()
    avg_eff = (p_df[stat_col] / p_df[vol_col].replace(0, 1)).mean() if not is_td else 1.0
    
    # Run the Engine
    sim_results = run_usage_monte_carlo(avg_vol, avg_eff, m_ratio, pace_mult, is_td)
    win_prob = (np.sum(sim_results >= market_line) / 10000) * 100

    # --- 6. DASHBOARD RENDERING ---
    st.title(f"🏈 {selected_p} ({p_pos}) Intelligence Hub")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Projection", round(np.mean(sim_results), 1), f"{round(win_prob,1)}% Win")
    col2.metric("Pace Factor", f"{round(pace_mult, 2)}x", f"{round(matchup_pace,1)} plays")
    col3.metric("Matchup Strength", m_status, f"{m_ratio}x Adj")
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # Plot Outcome Distribution
        fig = go.Figure(go.Histogram(x=sim_results, marker_color='#00ff96', nbinsx=40))
        fig.add_vline(x=market_line, line_dash="dash", line_color="red", annotation_text="Market Line")
        fig.update_layout(title="Monte Carlo Outcome Distribution", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("📈 Probability Ladder")
        st.table(generate_prob_ladder(sim_results, is_td))
        
        # Kelly Criterion Calculation
        st.subheader("💰 Bankroll Strategy")
        edge = (win_prob / 100) - (1 / 1.91) # Assuming -110 standard odds
        if edge > 0:
            kelly_stake = (bankroll * edge * kelly_fraction)
            st.success(f"Suggested Stake: ${round(kelly_stake, 2)}")
        else:
            st.error("No Mathematical Edge Found.")

else:
    st.error("Data Source Offline. Please check nflreadpy connectivity.")
