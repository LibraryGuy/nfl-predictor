import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
import requests
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

API_KEY = "a77014ce7ac884a8102b4aabd0efe1e6"

DEF_STATS_2025 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92}, 'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}, 
}

# --- PARLAY ENGINE ---
def generate_risk_parlay(selected_p, p_pos, p_team, model_proj, stat_label, data, risk_level):
    """Calculates a parlay based on the user's risk/odds preference."""
    risk_map = {
        "Conservative (-104)": {"scale": 0.65, "label": "Floor"},
        "Standard (+105)": {"scale": 0.85, "label": "Safe"},
        "Aggressive (+200)": {"scale": 1.05, "label": "Median"}
    }
    
    scale = risk_map[risk_level]["scale"]
    primary_val = round(model_proj * scale)
    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    # Logic for the 2nd Leg (Correlation)
    team_players = data[data['team'] == p_team]
    if p_pos == 'QB' and not team_players.empty:
        wrs = team_players[team_players['position'] == 'WR']
        if not wrs.empty:
            top_target = wrs.groupby('player_name')['receiving_yards'].sum().idxmax()
            leg_val = round(50 * (scale + 0.15)) 
            parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Stack"})
    elif p_pos in ['WR', 'TE'] and not team_players.empty:
        qbs = team_players[team_players['position'] == 'QB']
        if not qbs.empty:
            qb_name = qbs['player_name'].iloc[-1]
            leg_val = round(215 * scale)
            parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "Correlation"})
    
    return parlay_legs

# --- API HELPER ---
@st.cache_data(ttl=300)
def get_market_data(api_key, team_abbr):
    team_map = {
        'KC': 'Kansas City', 'GB': 'Green Bay', 'SF': 'San Francisco', 
        'TB': 'Tampa Bay', 'NE': 'New England', 'NO': 'New Orleans', 
        'LV': 'Las Vegas', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
        'ARI': 'Arizona', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BUF': 'Buffalo',
        'CAR': 'Carolina', 'CHI': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland',
        'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'HOU': 'Houston',
        'IND': 'Indianapolis', 'JAX': 'Jacksonville', 'MIA': 'Miami', 'MIN': 'Minnesota',
        'NYG': 'Giants', 'NYJ': 'Jets', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
        'SEA': 'Seattle', 'TEN': 'Tennessee', 'WAS': 'Washington'
    }
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
    except Exception:
        return None

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
        risk_pref = st.radio("Target Odds Profile", 
                             ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], 
                             index=1)

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
        adj_key = 'pass_adj' if p_pos != 'RB' else 'rush_adj'

        hist_avg = p_df[stat_col].mean()
        td_avg = p_df[td_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        
        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier

        st.title(f"📊 {selected_p} Intelligence Hub")
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.subheader("🏦 Sportsbook Money Tracker")
            fig_money = go.Figure()
            fig_money.add_trace(go.Bar(name='Tickets (Public)', x=['Market Sentiment'], y=[ticket_pct], marker_color='#4a4a4a'))
            fig_money.add_trace(go.Bar(name='Handle (Sharps)', x=['Market Sentiment'], y=[handle_pct], marker_color='#00ff96'))
            
            money_msg = "⚖️ Balanced Market Action"
            if handle_pct > ticket_pct + 10: money_msg = "🔥 Significant Sharp Money Detected"
            elif ticket_pct > handle_pct + 15: money_msg = "⚠️ Public Heavy - Possible Trap"
                
            fig_money.update_layout(barmode='group', template="plotly_dark", height=250, 
                                  title=dict(text=money_msg, font=dict(size=14, color="#00ff96")))
            st.plotly_chart(fig_money, use_container_width=True)

            current_risk_scale = {"Conservative (-104)": 0.65, "Standard (+105)": 0.85, "Aggressive (+200)": 1.05}[risk_pref]
            target_line = round(model_proj * current_risk_scale)
            
            last_5 = p_df.tail(5).copy()
            last_5['hit'] = last_5[stat_col] >= target_line
            fig_hits = go.Figure(go.Bar(x=[f"Wk {w}" for w in last_5['week']], y=last_5[stat_col], 
                                       marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_5['hit']]))
            fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b")
            fig_hits.update_layout(title=f"Last 5 Games vs {risk_pref} Target ({target_line}+)", template="plotly_dark", height=250)
            st.plotly_chart(fig_hits, use_container_width=True)

        with col_side:
            st.subheader("📋 Historical Averages")
            avg_data = {"Metric": [stat_label, "Touchdowns"], 
                        "Season": [round(hist_avg, 1), round(td_avg, 2)], 
                        "Last 5": [round(last_5[stat_col].mean(), 1), round(last_5[td_col].mean(), 2)]}
            st.table(pd.DataFrame(avg_data))
            
            st.divider()
            
            # --- START UPDATED INTELLIGENCE HUB SECTION ---
            st.subheader("🎯 Primary Edge")
            # Logic for Sharp vs Safe recommendation
            is_sharp = handle_pct > ticket_pct + 5
            safe_val = round(model_proj * 0.85)
            
            if is_sharp:
                st.success(f"💎 **Sharp Pick:** {selected_p} {safe_val}+ {stat_label}")
                st.caption("Sharps are backing this floor despite public volume.")
            else:
                st.info(f"🟢 **Safe Base:** {selected_p} {safe_val}+ {stat_label}")

            st.divider()
            st.subheader("🚀 Risk-Adjusted Parlays")
            
            # Display adjusted odds recommendations in Tabs alongside the primary leg
            t1, t2, t3 = st.tabs(["🛡️ Cons.", "✅ Std.", "🔥 Aggr."])
            tiers = [
                (t1, "Conservative (-104)", 0.65),
                (t2, "Standard (+105)", 0.85),
                (t3, "Aggressive (+200)", 1.05)
            ]
            
            for tab, r_name, scale in tiers:
                with tab:
                    parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, stat_label, data, r_name)
                    for leg in parlay_legs:
                        st.write(f"🔹 **{leg['type']}**: {leg['label']}")
                    
                    # Tier-specific hit rate context
                    tier_v = round(model_proj * scale)
                    tier_h = (last_5[stat_col] >= tier_v).sum()
                    st.caption(f"Historical Hit Rate: {tier_h}/5 Games")

            # Correlated Betting Add-ons
            if last_5[td_col].mean() >= 0.6 or game_script == "Shootout":
                st.divider()
                st.subheader("🔗 Correlated Add-ons")
                if last_5[td_col].mean() >= 0.6: 
                    st.write(f"🏈 **Value TD:** {selected_p} Anytime TD")
                if game_script == "Shootout": 
                    st.write(f"📈 **Game Script:** Over {v_total - 1.5} Total Pts")

            st.divider()
            hit_count = last_5['hit'].sum()
            st.write(f"**Consistency ({risk_pref.split(' ')[0]}):** {hit_count}/5 Games Hit")
            st.progress(hit_count / 5)
