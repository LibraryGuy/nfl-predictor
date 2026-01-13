import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

API_KEY = "a77014ce7ac884a8102b4aabd0efe1e6"

# --- LOGIC UPGRADE: DYNAMIC SOS ---
def get_dynamic_sos(data, stat_col):
    """Calculates defensive strength relative to league average performance."""
    if data.empty: return {}
    league_avg = data[stat_col].mean()
    def_strength = data.groupby('opponent')[stat_col].mean() / league_avg
    return def_strength.to_dict()

# --- PARLAY ENGINE (TEAMMATE CORRELATION) ---
def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level):
    """Calculates parlay legs using Teammate Stacking (Positive Correlation)."""
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    
    offset = risk_map[risk_level]["offset"]
    primary_val = round(p_mean + (offset * p_std))
    parlay_legs = [{"label": f"{selected_p}: {max(0, primary_val)}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    # Filter for Teammates only
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)]
    
    if not teammates.empty:
        # Scenario A: Selected player is a QB -> Pair with top WR/TE (The "Stack")
        if p_pos == 'QB':
            top_target = teammates[teammates['position'].isin(['WR', 'TE'])].groupby('player_name')['receiving_yards'].sum().idxmax()
            # Set target yardage based on risk tier
            leg_val = 40 if risk_level == "Conservative (-104)" else 65
            parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
            
        # Scenario B: Selected player is a WR/TE -> Pair with their QB (The "Link")
        elif p_pos in ['WR', 'TE']:
            team_qb = teammates[teammates['position'] == 'QB']['player_name'].unique()
            if len(team_qb) > 0:
                qb_name = team_qb[0]
                leg_val = 215 if risk_level == "Conservative (-104)" else 265
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "QB Link"})
                
        # Scenario C: Selected player is a RB -> Pair with QB (Correlates with team offensive success)
        elif p_pos == 'RB':
            team_qb = teammates[teammates['position'] == 'QB']['player_name'].unique()
            if len(team_qb) > 0:
                qb_name = team_qb[0]
                leg_val = 200 if risk_level == "Conservative (-104)" else 250
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "Offensive Flow"})

    return parlay_legs

# --- API HELPER ---
@st.cache_data(ttl=300)
def get_market_data(api_key, team_abbr):
    team_map = {'KC': 'Kansas City', 'GB': 'Green Bay', 'SF': 'San Francisco', 'TB': 'Tampa Bay', 'NE': 'New England', 'NO': 'New Orleans', 'LV': 'Las Vegas', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'ARI': 'Arizona', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BUF': 'Buffalo', 'CAR': 'Carolina', 'CHI': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'HOU': 'Houston', 'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'MIA': 'Miami', 'MIN': 'Minnesota', 'NYG': 'Giants', 'NYJ': 'Jets', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SEA': 'Seattle', 'TEN': 'Tennessee', 'WAS': 'Washington'}
    search_term = team_map.get(team_abbr, team_abbr).lower()
    try:
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            games = response.json()
            for game in games:
                if (search_term in game['home_team'].lower() or search_term in game['away_team'].lower()):
                    return game
        return None
    except Exception: return None

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}"); return pd.DataFrame(), pd.DataFrame()

# --- MAIN LOGIC ---
data, schedules = load_data_pro()
stadium_client = NFLStadiums()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_team_row = data[data['player_name'] == selected_p]
        p_team = p_team_row['team'].iloc[-1] if not p_team_row.empty else "N/A"
        
        matchup = schedules[((schedules['home_team'] == p_team) & (schedules['away_team'] == selected_opp)) | 
                            ((schedules['away_team'] == p_team) & (schedules['home_team'] == selected_opp))].iloc[-1:]
        
        v_total, v_spread, auto_script_val = 44.5, 0.0, "Balanced"
        if not matchup.empty:
            v_total = matchup['total_line'].values[0]
            v_spread = matchup['spread_line'].values[0] if matchup['home_team'].values[0] == p_team else -matchup['spread_line'].values[0]
            if v_total > 49: auto_script_val = "Shootout"
            elif v_total < 41: auto_script_val = "Defensive Struggle"

        st.info(f"🤖 **Vegas Pulse:** {v_total} O/U | Spread: {v_spread}")
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        market_line = st.number_input("Sportsbook Line", value=0.0, step=0.5)

        st.divider()
        st.subheader("⚙️ Parlay Settings")
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)

        st.divider()
        st.write("**💰 Market Sentiment**")
        if st.button("🔄 Force Market Refresh"):
            get_market_data.clear()
            st.toast("Market Data Refreshed", icon="📡")

        live_game = get_market_data(API_KEY, p_team)
        def_tickets, def_handle = 65, 45
        if live_game:
            st.success(f"🏟️ Game Found: {live_game['home_team']} vs {live_game['away_team']}")
            def_tickets = 74 if v_total > 47 else 56
            def_handle = 86 if v_total > 47 else 38
        else:
            st.warning(f"🔍 No live odds found for '{p_team}'.")

        ticket_pct = st.slider("% Public Tickets (Volume)", 0, 100, def_tickets)
        handle_pct = st.slider("% Sharp Handle (Dollars)", 0, 100, def_handle)
        game_script = st.select_slider("Expected Flow", options=["Defensive Struggle", "Balanced", "Shootout"], value=auto_script_val)

    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'passing_tds', 'Pass Yds'), 'RB': ('rushing_yards', 'rushing_tds', 'Rush Yds'), 
                    'WR': ('receiving_yards', 'receiving_tds', 'Rec Yds'), 'TE': ('receiving_yards', 'receiving_tds', 'Rec Yds')}
        stat_col, td_col, stat_label = stat_map.get(p_pos, ('receiving_yards', 'receiving_tds', 'Yds'))

        p_mean = p_df[stat_col].mean()
        p_std = p_df[stat_col].std() if len(p_df) > 1 else 1.0
        
        dynamic_sos = get_dynamic_sos(data, stat_col)
        sos_multiplier = dynamic_sos.get(selected_opp, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        model_proj = p_mean * script_boost * sos_multiplier

        st.title(f"📊 {selected_p} Intelligence Hub")
        
        col_main, col_side = st.columns([2, 1])
        with col_main:
            # Money tracker and Hit charts remain here...
            st.subheader("🏦 Sportsbook Money Tracker")
            fig_money = go.Figure()
            fig_money.add_trace(go.Bar(name='Tickets (Public)', x=['Market Sentiment'], y=[ticket_pct], marker_color='#4a4a4a'))
            fig_money.add_trace(go.Bar(name='Handle (Sharps)', x=['Market Sentiment'], y=[handle_pct], marker_color='#00ff96'))
            st.plotly_chart(fig_money, use_container_width=True)

        with col_side:
            st.subheader("🚀 Teammate Stacking Parlays")
            t1, t2, t3 = st.tabs(["🛡️ Cons.", "✅ Std.", "🔥 Aggr."])
            
            for tab, r_name in zip([t1, t2, t3], ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"]):
                with tab:
                    # UPDATED CALL TO USE TEAMMATE LOGIC
                    parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, p_std, stat_label, data, r_name)
                    for leg in parlay_legs:
                        st.write(f"🔹 **{leg['type']}**: {leg['label']}")
