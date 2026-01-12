import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

DEF_STATS_2025 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92}, 'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}, 
}

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

data, schedules = load_data_pro()
stadium_client = NFLStadiums()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        p_team = data[data['player_name'] == selected_p]['team'].iloc[-1]
        
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
        
        # --- NEW: MONEY TRACKER INPUTS ---
        st.divider()
        st.write("**💰 Market Sentiment**")
        ticket_pct = st.slider("% Public Tickets (Volume)", 0, 100, 65)
        handle_pct = st.slider("% Sharp Handle (Dollars)", 0, 100, 45)
        
        game_script = st.select_slider("Expected Flow", options=["Defensive Struggle", "Balanced", "Shootout"], value=auto_script_val)
        st.toast(f"Active Script: {game_script}", icon="🎭")

    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'passing_tds', 'Pass Yds'), 'RB': ('rushing_yards', 'rushing_tds', 'Rush Yds'), 
                    'WR': ('receiving_yards', 'receiving_tds', 'Rec Yds'), 'TE': ('receiving_yards', 'receiving_tds', 'Rec Yds')}
        stat_col, td_col, stat_label = stat_map.get(p_pos, ('receiving_yards', 'receiving_tds', 'Yds'))
        adj_key = 'pass_adj' if p_pos != 'RB' else 'rush_adj'

        hist_avg = p_df[stat_col].mean()
        td_avg = p_df[td_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        
        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier
        safe_leg_val = round(model_proj * 0.85)

        st.title(f"📊 {selected_p} Intelligence Hub")
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            # --- NEW: SPORTSBOOK MONEY TRACKER CHART ---
            st.subheader("🏦 Sportsbook Money Tracker")
            fig_money = go.Figure()
            fig_money.add_trace(go.Bar(name='Tickets (Public)', x=['Over/Under'], y=[ticket_pct], marker_color='#4a4a4a'))
            fig_money.add_trace(go.Bar(name='Handle (Sharps)', x=['Over/Under'], y=[handle_pct], marker_color='#00ff96'))
            
            # Highlight Sharp Action
            if handle_pct > ticket_pct + 10:
                money_msg = "🔥 Significant Sharp Money Detected"
            elif ticket_pct > handle_pct + 15:
                money_msg = "⚠️ Public Heavy - Possible Trap"
            else:
                money_msg = "⚖️ Balanced Market Action"
                
            fig_money.update_layout(barmode='group', template="plotly_dark", height=250, margin=dict(t=20, b=20),
                                  title=dict(text=money_msg, font=dict(size=14, color="#00ff96")))
            st.plotly_chart(fig_money, use_container_width=True)

            # Previous Charts
            last_5 = p_df.tail(5).copy()
            last_5['hit'] = last_5[stat_col] >= safe_leg_val
            fig_hits = go.Figure(go.Bar(x=[f"Wk {w}" for w in last_5['week']], y=last_5[stat_col], 
                                       marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_5['hit']], text=last_5[stat_col], textposition='auto'))
            fig_hits.add_hline(y=safe_leg_val, line_dash="dash", line_color="#ff4b4b")
            fig_hits.update_layout(title=f"Performance vs Target ({safe_leg_val}+)", template="plotly_dark", height=250)
            st.plotly_chart(fig_hits, use_container_width=True)

        with col_side:
            st.subheader("📋 Historical Averages")
            avg_data = {"Metric": [stat_label, "Touchdowns"], "Season": [round(hist_avg, 1), round(td_avg, 2)], "Last 5": [round(last_5[stat_col].mean(), 1), round(last_5[td_col].mean(), 2)]}
            st.table(pd.DataFrame(avg_data))
            
            st.divider()
            st.subheader("🔥 Parlay Recommendations")
            # Logic check for Sharp Money in recommendations
            if handle_pct > ticket_pct + 5:
                st.success(f"💎 **Sharp Pick:** {selected_p} {safe_leg_val}+ {stat_label}")
            else:
                st.success(f"**Safe Leg:** {selected_p} {safe_leg_val}+ {stat_label}")
            
            if last_5[td_col].mean() >= 0.6: st.info(f"**High Value TD:** {selected_p} Anytime TD Scorer")
            if game_script == "Shootout": st.warning(f"**Correlated Leg:** Over {v_total - 1.5} Total Pts")
            
            st.divider()
            hit_count = last_5['hit'].sum()
            st.write(f"**Consistency:** {hit_count}/5 Games Hit")
            st.progress(hit_count / 5)
