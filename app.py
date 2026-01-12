import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Automated Intelligence", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_betting_environment():
    try:
        # Load Player Stats
        df = nfl.load_player_stats(seasons=[2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        # Load Vegas Lines (Schedules)
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        
        # Rename for consistency
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        
        return df, sched
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, schedules = load_betting_environment()
stadium_client = NFLStadiums()

# --- 2. AUTOMATED SCRIPT CALCULATION ---
def get_automated_script(team, opp_team, sched_df):
    # Find the game in the schedule
    game = sched_df[((sched_df['home_team'] == team) & (sched_df['away_team'] == opp_team)) | 
                    ((sched_df['away_team'] == team) & (sched_df['home_team'] == opp_team))].iloc[-1:]
    
    if game.empty: return "Balanced", 1.0, 44.0, 0.0 # Default fallback
    
    total = game['total_line'].values[0]
    spread = game['spread_line'].values[0] # Home spread
    is_home = (game['home_team'].values[0] == team)
    
    # Adjust spread for the specific player's team
    team_spread = spread if is_home else -spread
    
    # Logic Engine
    if total > 49: script = "Shootout"
    elif total < 41: script = "Defensive Struggle"
    elif team_spread < -7: script = "Heavy Lead"
    elif team_spread > 7: script = "Playing From Behind"
    else: script = "Balanced"
    
    # Multiplier impact
    impact = {"Defensive Struggle": 0.88, "Balanced": 1.0, "Shootout": 1.20, 
              "Heavy Lead": 0.92, "Playing From Behind": 1.15}[script]
    
    return script, impact, total, team_spread

# --- 3. THE INTERFACE ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        p_team = data[data['player_name'] == selected_p]['team'].iloc[-1]
        selected_opp = st.selectbox("Opponent", sorted(data['opponent'].unique()))
        
        # --- AUTO SCRIPT TRIGGER ---
        auto_script, script_mult, v_total, v_spread = get_automated_script(p_team, selected_opp, schedules)
        
        st.success(f"**Auto-Script:** {auto_script}")
        st.caption(f"Based on Vegas Total: {v_total} | Spread: {v_spread}")

        st.divider()
        st.subheader("🏥 Injury Impact")
        teammate_out = st.checkbox("Key Teammate OUT")
        injury_mult = 1.18 if teammate_out else 1.0

    # --- 4. PREDICTION LOGIC ---
    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_col = 'passing_yards' if p_pos == 'QB' else ('rushing_yards' if p_pos == 'RB' else 'receiving_yards')
        
        # Combine all automated factors
        model_proj = p_df[stat_col].mean() * script_mult * injury_mult
        
        st.title(f"🚀 {selected_p} | {p_team} vs {selected_opp}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Automated Script", auto_script, delta=f"{script_mult}x Impact")
        c2.metric("Injury Boost", "Active" if teammate_out else "None", delta=f"{injury_mult}x" if teammate_out else None)
        c3.metric("Final Sharp Projection", f"{model_proj:.1f} Yards")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p_df[stat_col], mode='lines+markers', name="Last Games", line=dict(color='#00ff88')))
        fig.add_hline(y=model_proj, line_dash="dash", line_color="red", annotation_text="Model Target")
        fig.update_layout(template="plotly_dark", title="Player Performance Trend")
        st.plotly_chart(fig, use_container_width=True)
