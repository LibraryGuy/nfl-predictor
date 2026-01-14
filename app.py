import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

# --- 1. SETTINGS & CONSTANTS ---
st.set_page_config(page_title="NFL Sharp 2026", layout="wide", page_icon="🏈")
LEAGUE_AVG_PLAYS = 63.5 

# --- 2. DATA LOADING (ULTRA-ROBUST VERSION) ---
@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        # Fetching multiple seasons to ensure we have a baseline
        raw_result = nfl.load_player_stats(seasons=[2024, 2025])
        
        # Polars to Pandas conversion
        df = raw_result.to_pandas() if hasattr(raw_result, 'to_pandas') else pd.DataFrame(raw_result)
            
        if df.empty:
            st.error("Data source is empty. Check internet or API status.")
            return pd.DataFrame()

        # --- THE FIX: Aggressive Column Normalization ---
        # We rename everything to our internal 'gold standard' names
        cols_to_fix = {
            'player_display_name': 'player_name',
            'player': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent',
            'gsis_game_id': 'game_id',
            'gsis_id': 'player_id'
        }
        df = df.rename(columns=cols_to_fix)

        # Safety Check: If 'player_name' is still missing, we find the closest match
        if 'player_name' not in df.columns:
            possible_names = [c for c in df.columns if 'name' in c.lower()]
            if possible_names:
                df = df.rename(columns={possible_names[0]: 'player_name'})

        # Clean numerical data
        numerical_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        return df.dropna(subset=['player_name'])
    
    except Exception as e:
        st.error(f"Data Load Error: {str(e)}")
        return pd.DataFrame()

# --- 3. ANALYTICS ENGINE ---
@st.cache_data
def get_pace_map(data):
    if 'game_id' not in data.columns or 'team' not in data.columns:
        return {}
    team_plays = data.groupby(['team', 'game_id']).size().reset_index(name='plays')
    return team_plays.groupby('team')['plays'].mean().to_dict()

def run_monte_carlo(avg_vol, avg_eff, m_ratio, pace_mult, is_td):
    adj_vol = max(avg_vol * pace_mult, 0.1)
    iterations = 10000
    if is_td:
        return np.random.poisson(adj_vol * m_ratio, iterations)
    else:
        sim_vol = np.random.poisson(adj_vol, iterations)
        # Efficiency adjustment via Log-Normal distribution
        mu = np.log(max(avg_eff * m_ratio, 0.01)) - (0.4**2 / 2)
        return sim_vol * np.random.lognormal(mu, 0.4, iterations)

# --- 4. MAIN APP LOGIC ---
data = load_data_pro()

if not data.empty and 'player_name' in data.columns:
    pace_lookup = get_pace_map(data)
    
    with st.sidebar:
        st.header("Parameters")
        player_list = sorted(data['player_name'].unique()) # LINE 126 FIX: Ensured column exists
        selected_p = st.selectbox("Select Player", player_list)
        
        # Filter data for selected player
        p_df = data[data['player_name'] == selected_p].copy()
        p_team = p_df['team'].iloc[-1] if 'team' in p_df.columns else "N/A"
        
        selected_opp = st.selectbox("Opponent", sorted(data['opponent'].unique() if 'opponent' in data.columns else ["NFL"]))
        market_type = st.radio("Market", ["Yards", "Touchdowns"])
        line = st.number_input("Line", value=50.0 if market_type == "Yards" else 0.5)

    # --- CALCULATIONS ---
    t_pace = pace_lookup.get(p_team, LEAGUE_AVG_PLAYS)
    o_pace = pace_lookup.get(selected_opp, LEAGUE_AVG_PLAYS)
    pace_mult = ((t_pace + o_pace) / 2) / LEAGUE_AVG_PLAYS
    
    # Run simulation (Simplified for display)
    sim_results = run_monte_carlo(10, 5, 1.0, pace_mult, (market_type == "Touchdowns"))
    win_prob = (np.sum(sim_results >= line) / 10000) * 100

    # --- RENDER ---
    st.title(f"Analysis: {selected_p}")
    c1, c2 = st.columns(2)
    c1.metric("Win Probability", f"{round(win_prob, 1)}%")
    c2.metric("Pace Multiplier", f"{round(pace_mult, 2)}x")
    
    fig = go.Figure(go.Histogram(x=sim_results, marker_color='#00ff96'))
    fig.add_vline(x=line, line_color="red")
    st.plotly_chart(fig)
else:
    st.error("Critical Error: The dataset does not contain a 'player_name' column. Please verify your data source.")
