import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="NFL Genius: Matchup Pro", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_comprehensive_data():
    try:
        # Load core player stats
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Standardize Columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        # Map essential columns
        rename_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        # Calculate Base Scrimmage Yards
        df['total_yds'] = df.get('rushing_yards', 0).fillna(0) + df.get('receiving_yards', 0).fillna(0)
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Sync Error: {e}")
        return pd.DataFrame()

data = load_comprehensive_data()

# --- 2. DEFENSE ANALYTICS ENGINE ---
def get_defense_metrics(df):
    """Calculates how many yards each defense allows by position."""
    # League averages for benchmarking
    league_avg = df.groupby('position')['total_yds'].mean().to_dict()
    
    # Team-specific averages
    def_stats = df.groupby(['opponent', 'position'])['total_yds'].mean().reset_index()
    return def_stats, league_avg

if not data.empty:
    def_data, league_benchmarks = get_defense_metrics(data)
    
    # --- 3. UI: PLAYER & OPPONENT SELECTION ---
    st.title("🏈 NFL Genius: Matchup Predictor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        players = sorted(data['player_name'].unique())
        selected_p = st.selectbox("1. Choose Player", players)
    
    with col_b:
        opponents = sorted(data['opponent'].unique())
        selected_opp = st.selectbox("2. Choose Opponent Defense", opponents)

    # --- 4. PREDICTION LOGIC ---
    p_df = data[data['player_name'] == selected_p]
    p_pos = p_df['position'].iloc[-1]
    p_avg = p_df['total_yds'].mean()
    
    # Calculate Defense Modifier
    # Logic: (Opponent Avg Allowed to Pos) / (League Avg Allowed to Pos)
    opp_allowed = def_data[(def_data['opponent'] == selected_opp) & 
                           (def_data['position'] == p_pos)]['total_yds']
    
    bench = league_benchmarks.get(p_pos, 1)
    # If we have data on the opponent, use it; otherwise, neutral (1.0)
    matchup_mod = (opp_allowed.iloc[0] / bench) if not opp_allowed.empty else 1.0
    
    # Final Calculation
    proj_yds = p_avg * matchup_mod
    
    # --- 5. DASHBOARD DISPLAY ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Player Season Avg", f"{p_avg:.1f} Yds")
    
    # Color code the modifier
    mod_delta = "Favorable" if matchup_mod > 1.05 else "Difficult" if matchup_mod < 0.95 else "Neutral"
    m2.metric("Matchup Difficulty", f"{matchup_mod:.2f}x", delta=mod_delta)
    
    m3.metric("Projected Performance", f"{proj_yds:.1f} Yds", 
              delta=f"{proj_yds - p_avg:+.1f} vs Avg", delta_color="normal")

    # Visualizing Trend vs Defense
    st.subheader(f"Recent Performance: {selected_p}")
    fig = px.bar(p_df.tail(8), x='week', y='total_yds', color='total_yds',
                 title="Total Yards - Last 8 Games",
                 labels={'total_yds': 'Yards', 'week': 'Week'})
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please wait for the data to sync...")
