import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson, lognorm

# --- 1. DATA ACCESS & DEFENSIVE MODELING ---
@st.cache_data(ttl=3600)
def get_advanced_data():
    """Loads 6 seasons of data to ensure robust head-to-head history."""
    try:
        curr_season = nfl.get_current_season()
        # Loading from 2020 to current (2025/26) captures true H2H trends
        seasons_to_load = list(range(curr_season - 5, curr_season + 1))
        
        # Load player stats (weekly)
        raw_stats = nfl.load_player_stats(seasons=seasons_to_load, summary_level='week').to_pandas()
        
        # Load team stats for current season Matchup Multipliers
        team_stats = nfl.load_team_stats(seasons=[curr_season]).to_pandas()
        
        # Create defensive lookup table
        def_lookup = team_stats.groupby('team').mean(numeric_only=True)
        league_means = team_stats.mean(numeric_only=True)
        
        return raw_stats, def_lookup, league_means
    except Exception as e:
        st.error(f"Engine Data Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

@st.cache_data(ttl=3600)
def get_registry():
    """Maps display names to GSIS IDs for reliable data joins."""
    players = nfl.load_players().to_pandas()
    active = players[players['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    return {f"{row['display_name']} ({row['position']})": row['gsis_id'] for _, row in active.iterrows()}

# --- 2. MATCHUP LOGIC ---
def calculate_matchup_multiplier(opponent, stat_type, def_lookup, league_means):
    mapping = {
        'passing_yards': 'opp_passing_yards',
        'rushing_yards': 'opp_rushing_yards',
        'receiving_yards': 'opp_passing_yards',
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
st.set_page_config(page_title="NFL Sharp Predictor v5", layout="wide")

registry = get_registry()
raw_stats, def_lookup, lg_means = get_advanced_data()

st.sidebar.title("🏈 Model Controls")
selected_label = st.sidebar.selectbox("Target Player", list(registry.keys()))
player_id = registry[selected_label]

if player_id and not raw_stats.empty:
    p_df_all = raw_stats[raw_stats['player_id'] == player_id].copy()
    
    if not p_df_all.empty:
        pos = p_df_all['position'].iloc[-1]
        market_map = {
            'QB': ['passing_yards', 'passing_tds'],
            'RB': ['rushing_yards', 'rushing_tds'],
            'WR': ['receiving_yards', 'receiving_tds'],
            'TE': ['receiving_yards', 'receiving_tds']
        }
        stat_col = st.sidebar.selectbox("Market", market_map.get(pos, ['receiving_yards']))
        line = st.sidebar.number_input("Sportsbook Line", value=45.5 if "yards" in stat_col else 0.5)
        
        teams = sorted(def_lookup.index.tolist())
        opp = st.sidebar.selectbox("Next Opponent", teams)
        
        # --- MONTE CARLO ENGINE (Current Season Only for Projections) ---
        curr_season = nfl.get_current_season()
        p_df_curr = p_df_all[p_df_all['season'] == curr_season]
        model_data = p_df_curr if len(p_df_curr) >= 3 else p_df_all
        
        match_mult = calculate_matchup_multiplier(opp, stat_col, def_lookup, lg_means)
        base_avg = model_data[stat_col].mean()
        adj_avg = base_avg * match_mult
        
        iterations = 10000
        if "tds" in stat_col:
            sims = np.random.poisson(max(adj_avg, 0.01), iterations)
        else:
            std = model_data[stat_col].std() if model_data[stat_col].std() > 0 else (base_avg * 0.4)
            sigma = np.sqrt(np.log(1 + (std**2 / (adj_avg**2 + 1e-9))))
            mu = np.log(adj_avg + 1e-9) - (sigma**2 / 2)
            sims = np.random.lognormal(mu, sigma, iterations)

        # MAIN DASHBOARD
        st.title(f"Model Intelligence: {selected_label}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Matchup Multiplier", f"{match_mult:.2f}x", delta=f"{match_mult-1:.2f}", delta_color="inverse")
        m2.metric("Projected Median", f"{np.median(sims):.1f}")
        m3.metric("Over Probability", f"{(np.sum(sims > line)/10000)*100:.1f}%")

        # Distribution Graph
        fig_dist = go.Figure(go.Histogram(x=sims, nbinsx=40, marker_color='#00f2ff', opacity=0.6))
        fig_dist.add_vline(x=line, line_color="red", line_dash="dash", annotation_text="Line")
        fig_dist.update_layout(title=f"10,000 Iteration Forecast vs {opp}", template="plotly_dark")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # --- PLAYER LAST 5 GAMES (Form Chart) ---
        st.divider()
        st.subheader(f"📈 {selected_label}: Recent Performance (Last 5 Games)")
        last_5_overall = p_df_all.sort_values(['season', 'week'], ascending=True).tail(5)
        last_5_overall['game_label'] = "S" + last_5_overall['season'].astype(str) + " W" + last_5_overall['week'].astype(str)
        
        fig_form = px.bar(
            last_5_overall, 
            x='game_label', 
            y=stat_col, 
            color=stat_col,
            color_continuous_scale='GnBu',
            text_auto=True,
            labels={'game_label': 'Game', stat_col: stat_col.replace('_', ' ').title()}
        )
        fig_form.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_form, use_container_width=True)
        

        # --- HEAD-TO-HEAD HISTORY (All time vs Opponent up to 5) ---
        st.subheader(f"🏟️ Historical Performance vs {opp}")
        # Search the entire 6-season database for games against this specific opponent
        h2h_all_time = p_df_all[p_df_all['opponent_team'] == opp].sort_values(['season', 'week'], ascending=False).head(5)
        
        if not h2h_all_time.empty:
            st.write(f"Displaying the last {len(h2h_all_time)} matchups against the {opp}:")
            cols_to_show = ['season', 'week', 'opponent_team', stat_col, 'completions', 'attempts', 'receptions', 'targets']
            # Filter columns to only those that exist in the data (e.g., skip 'receptions' for QBs)
            available_cols = [c for c in cols_to_show if c in h2h_all_time.columns]
            
            # Using dataframe for a cleaner, scrollable table
            st.dataframe(
                h2h_all_time[available_cols].rename(columns=lambda x: x.replace('_', ' ').title()),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No historical games found for {selected_label} against the {opp} in the current dataset (2020-2025).")

    else:
        st.warning(f"No game logs found for ID {player_id}.")
