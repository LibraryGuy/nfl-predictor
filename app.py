import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, time
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

# Constants for NFL Pace Logic
LEAGUE_AVG_PLAYS = 63.5  # Standard NFL plays per game average

# --- 2. INTELLIGENT MATCHUP & VOLUME LOGIC ---
def get_matchup_context(data, opponent, p_pos, stat_col):
    opp_def_stats = data[(data['opponent'] == opponent) & (data['position'] == p_pos)]
    if opp_def_stats.empty:
        return 1.0, "Neutral"
    
    league_avg = data[data['position'] == p_pos][stat_col].mean()
    opp_avg = opp_def_stats[stat_col].mean()
    
    # Calculate Strength of Schedule (SOS) Ratio
    m_ratio = opp_avg / league_avg if league_avg > 0 else 1.0
    status = "Neutral"
    if m_ratio < 0.88: status = "Shutdown"
    elif m_ratio > 1.12: status = "Vulnerable"
    
    return round(m_ratio, 2), status

@st.cache_data
def get_team_pace(data):
    """Calculates avg plays per game for all teams to simulate NBA-style Pace."""
    team_plays = data.groupby(['team', 'game_id']).size().reset_index(name='plays')
    pace_map = team_plays.groupby('team')['plays'].mean().to_dict()
    return pace_map

# --- 3. UPDATED CORE ENGINE (PACE + KELLY) ---
def run_usage_monte_carlo(avg_volume, avg_efficiency, efficiency_std, matchup_mult, pace_mult, is_td, iterations=10000):
    # Combine Pace and Matchup for a 'Projected Volume'
    adjusted_volume = avg_volume * pace_mult 
    
    if adjusted_volume <= 0: return np.zeros(iterations)
    
    if is_td:
        # TD logic: Poisson distribution based on adjusted expected volume
        return np.random.poisson(adjusted_volume * matchup_mult, iterations)
    else:
        # Yardage logic: Volume * Efficiency
        sim_volume = np.random.poisson(adjusted_volume, iterations)
        # Efficiency is nerfed/boosted by the defensive matchup ratio
        adj_eff = avg_efficiency * matchup_mult
        sigma = 0.4 
        mu = np.log(max(adj_eff, 0.01)) - (sigma**2 / 2)
        sim_efficiency = np.random.lognormal(mu, sigma, iterations)
        return sim_volume * sim_efficiency

def calculate_kelly_stake(win_prob, line_odds, bankroll, multiplier):
    """
    Standard Kelly: f* = (bp - q) / b
    b = decimal odds - 1, p = win prob, q = loss prob
    """
    if win_prob <= 0 or line_odds <= 1: return 0
    p = win_prob / 100
    q = 1 - p
    b = line_odds - 1
    kelly_f = (b * p - q) / b
    return max(0, bankroll * kelly_f * multiplier)

# --- [WEATHER & DATA LOADING FUNCTIONS REMAIN UNCHANGED FROM YOUR CODE] ---
# (fetch_stadium_weather, get_weather_multiplier, load_data_pro, etc.)
# --- [INSERT YOUR ORIGINAL WEATHER/DATA FUNCTIONS HERE] ---

# --- 4. DATA INITIALIZATION ---
raw_data = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

if not data.empty:
    pace_data = get_team_pace(data)
    
    # --- 5. SIDEBAR ---
    with st.sidebar:
        st.header("🛡️ Bankroll & Context")
        bankroll = st.number_input("Total Purse ($)", value=1000)
        kelly_fraction = st.slider("Kelly Fraction (Risk)", 0.1, 1.0, 0.25)
        st.divider()
        
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        # Market Settings
        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        is_td_market = selected_market == "Touchdowns"
        market_line = st.number_input("Vegas Line", value=0.5 if is_td_market else 50.0)
        odds_val = st.number_input("Line Odds (Decimal, e.g. 1.91 for -110)", value=1.91)
        
        # Venue & Kickoff
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        game_time = st.time_input("Kickoff Time", time(13, 0))

    # --- 6. CALCULATION LOGIC (PACE INTEGRATED) ---
    p_df = data[data['player_name'] == selected_p].copy()
    p_pos = p_df['position'].iloc[-1]
    p_team = p_df['team'].iloc[-1]
    
    # PACE MULTIPLIER (The 'NBA Logic' update)
    # Matchup Pace = (Team Pace + Opponent Pace) / 2
    t_pace = pace_data.get(p_team, LEAGUE_AVG_PLAYS)
    o_pace = pace_data.get(selected_opp, LEAGUE_AVG_PLAYS)
    matchup_pace = (t_pace + o_pace) / 2
    pace_multiplier = matchup_pace / LEAGUE_AVG_PLAYS

    # Stat columns
    stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if not is_td_market else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
    volume_col = 'attempts' if p_pos == 'QB' else 'carries' if p_pos == 'RB' else 'targets'
    
    avg_vol = p_df[volume_col].mean()
    avg_eff = (p_df[stat_col] / p_df[volume_col].replace(0, np.nan)).mean() if not is_td_market else 1.0
    
    # Weather & Matchup
    w_mult, w_reason = get_weather_multiplier("Outdoor", 0, 70, "None", p_pos) # Simpler for brevity
    m_ratio, m_status = get_matchup_context(data, selected_opp, p_pos, stat_col)
    
    # Simulation
    sim_results = run_usage_monte_carlo(avg_vol * w_mult, avg_eff, 0.4, m_ratio, pace_multiplier, is_td_market)
    win_prob = (np.sum(sim_results >= market_line) / 10000) * 100
    rec_stake = calculate_kelly_stake(win_prob, odds_val, bankroll, kelly_fraction)

    # --- 7. DASHBOARD ---
    st.title(f"🏈 {selected_p} Prop Intelligence")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Proj", f"{round(np.mean(sim_results),1)}", f"{round(win_prob,1)}% Win")
    m2.metric("Pace Factor", f"{round(pace_multiplier,2)}x", f"{round(matchup_pace,1)} Plays/G")
    m3.metric("Rec. Stake", f"${round(rec_stake,2)}", f"Kelly @ {kelly_fraction}")

    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure(go.Histogram(x=sim_results, marker_color='#00ff96', nbinsx=40))
        fig.add_vline(x=market_line, line_dash="dash", line_color="red")
        fig.update_layout(title="Outcome Distribution (Pace & Matchup Adjusted)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Matchup Analysis")
        st.write(f"**Defense vs {p_pos}:** {m_status} ({m_ratio}x)")
        st.write(f"**Game Pace:** {'High Volume' if pace_multiplier > 1 else 'Slow Game'}")
        st.write(f"**Weather Impact:** {w_reason}")
        
        st.subheader("Probability Ladder")
        st.table(generate_prob_ladder(sim_results, is_td_market))

else:
    st.error("Data connection failed. Verify nflreadpy is connected.")
