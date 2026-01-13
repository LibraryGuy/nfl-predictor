import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Try-except for the stadium library as it is a specialized local dependency
try:
    from nfl_stadiums import NFLStadiums
except ImportError:
    NFLStadiums = None

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

# --- 2. CORE LOGIC ---
@st.cache_resource
def get_stadium_client():
    return NFLStadiums() if NFLStadiums else None

def get_dynamic_sos(data, stat_col):
    if data.empty or stat_col not in data.columns: return {}
    league_avg = data[stat_col].mean()
    if league_avg == 0: return {}
    # Higher number = easier defense (allows more yards than avg)
    def_strength = data.groupby('opponent')[stat_col].mean() / league_avg
    return def_strength.to_dict()

def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    offset = risk_map[risk_level]["offset"]
    primary_val = round(p_mean + (offset * p_std))
    parlay_legs = [{"label": f"{selected_p}: {max(0, primary_val)}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)]
    if not teammates.empty:
        if p_pos == 'QB':
            targets = teammates[teammates['position'].isin(['WR', 'TE'])]
            if not targets.empty:
                top_target = targets.groupby('player_name')['receiving_yards'].sum().idxmax()
                leg_val = 40 if risk_level == "Conservative (-104)" else 60
                parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
        elif p_pos in ['WR', 'TE', 'RB']:
            team_qb_list = teammates[teammates['position'] == 'QB']['player_name'].unique()
            if len(team_qb_list) > 0:
                qb_name = team_qb_list[0]
                leg_val = 200 if risk_level == "Conservative (-104)" else 245
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "QB Link" if p_pos != 'RB' else "Team Success"})
    return parlay_legs

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        # Loading current and previous season for better baseline stats
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df.dropna(subset=['player_name', 'opponent', 'position']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. UI RENDERING ---
data, schedules = load_data_pro()
stadium_client = get_stadium_client()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        # Safe extraction of team and stats
        p_df = data[data['player_name'] == selected_p].copy()
        p_team = p_df['team'].iloc[-1] if not p_df.empty else "N/A"
        p_pos = p_df['position'].iloc[-1] if not p_df.empty else "WR"
        
        matchup = schedules[((schedules['home_team'] == p_team) & (schedules['away_team'] == selected_opp)) | 
                            ((schedules['away_team'] == p_team) & (schedules['home_team'] == selected_opp))].iloc[-1:]
        
        v_total = matchup['total_line'].values[0] if not matchup.empty else 44.5
        v_spread = matchup['spread_line'].values[0] if not matchup.empty else 0.0
        
        st.info(f"🤖 **Vegas Pulse:** {v_total} O/U | Spread: {v_spread}")
        market_line = st.number_input("Sportsbook Line", value=0.0, step=0.5)

        st.divider()
        st.subheader("⚙️ Parlay Settings")
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
        game_script = st.select_slider("Expected Flow", options=["Defensive Struggle", "Balanced", "Shootout"], value="Balanced")

    # DATA PROCESSING
    stat_map = {'QB': ('passing_yards', 'passing_tds', 'Pass Yds'), 
                'RB': ('rushing_yards', 'rushing_tds', 'Rush Yds'), 
                'WR': ('receiving_yards', 'receiving_tds', 'Rec Yds'), 
                'TE': ('receiving_yards', 'receiving_tds', 'Rec Yds')}
    
    stat_col, td_col, stat_label = stat_map.get(p_pos, ('receiving_yards', 'receiving_tds', 'Yds'))

    p_mean = p_df[stat_col].mean()
    # Guard against zero variance for players with 1 game or identical stats
    p_std = p_df[stat_col].std() if len(p_df) > 1 else 1.0
    if p_std == 0 or np.isnan(p_std): p_std = 1.0 
    
    dynamic_sos = get_dynamic_sos(data, stat_col).get(selected_opp, 1.0)
    script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
    model_proj = p_mean * script_boost * dynamic_sos

    # WIN PROBABILITY (Using Normal Distribution)
    win_prob = (1 - norm.cdf(market_line, loc=model_proj, scale=p_std)) * 100 if market_line > 0 else 0
    edge_msg = f"🔥 {round(win_prob)}% Hit Probability" if win_prob > 60 else f"⚖️ {round(win_prob)}% Win Probability"

    st.title(f"📊 {selected_p} Intelligence Hub")
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("🏦 Sportsbook Money Tracker")
        fig_money = go.Figure()
        fig_money.add_trace(go.Bar(name='Tickets', x=['Market Sentiment'], y=[65], marker_color='#4a4a4a'))
        fig_money.add_trace(go.Bar(name='Handle', x=['Market Sentiment'], y=[85 if win_prob > 55 else 40], marker_color='#00ff96'))
        st.plotly_chart(fig_money, use_container_width=True)

        target_line = round(model_proj + ({"Conservative (-104)": -0.6, "Standard (+105)": 0, "Aggressive (+200)": 0.6}[risk_pref] * p_std))
        last_5 = p_df.tail(5).copy()
        last_5['hit'] = last_5[stat_col] >= target_line
        
        fig_hits = go.Figure(go.Bar(
            x=[f"Wk {int(w)}" for w in last_5['week']], 
            y=last_5[stat_col], 
            marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_5['hit']]
        ))
        fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Target")
        fig_hits.update_layout(title=f"Last {len(last_5)} Games vs {risk_pref} Target")
        st.plotly_chart(fig_hits, use_container_width=True)

    with col_side:
        st.subheader("🎯 Primary Edge")
        with st.container(border=True):
            st.metric(label=f"Projected {stat_label}", value=round(model_proj, 1), delta=f"{round(model_proj - market_line, 1)} vs Line")
            st.write(f"**{edge_msg}**")
            st.caption(f"Recommendation: {'SHARP PICK' if win_prob > 62 else 'SAFE BASE' if win_prob > 52 else 'AVOID'}")

        st.subheader("🛡️ Correlated Add-ons")
        with st.expander("Filter Additional Legs", expanded=True):
            td_rate = p_df[td_col].mean()
            st.write(f"🏈 **Anytime TD:** {round(td_rate * 100)}% Prob.")
            st.checkbox(f"Add {selected_p} Anytime TD", value=td_rate > 0.4)
            st.checkbox(f"Game Total: {'OVER' if v_total > 45 else 'UNDER'} {v_total}", value=True)

        st.divider()
        st.subheader("🚀 Teammate Stacking")
        t1, t2, t3 = st.tabs(["🛡️ Cons.", "✅ Std.", "🔥 Aggr."])
        for tab, r_name in zip([t1, t2, t3], ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"]):
            with tab:
                parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, p_std, stat_label, data, r_name)
                for leg in parlay_legs:
                    st.write(f"🔹 **{leg['type']}**: {leg['label']}")
                
                tier_v = round(model_proj + ({"Conservative (-104)": -0.6, "Standard (+105)": 0, "Aggressive (+200)": 0.6}[r_name] * p_std))
                actual_hits = (last_5[stat_col] >= tier_v).sum()
                st.caption(f"Historical Hit Rate: {actual_hits}/{len(last_5)} Games")

        # Dynamic progress bar based on actual data points available
        if not last_5.empty:
            st.progress(last_5['hit'].sum() / len(last_5))
else:
    st.warning("Data sync in progress or no data available for the selected parameters.")
