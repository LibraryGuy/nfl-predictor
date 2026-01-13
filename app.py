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

# --- NEW: WEATHER IMPACT LOGIC ---
def get_weather_multiplier(roof_type, wind, temp, precip, p_pos):
    """Calculates a multiplier based on weather severity and stadium type."""
    if roof_type in ['Dome', 'Closed', 'Indoor']:
        return 1.0, "Dome (No Impact)"
    
    multiplier = 1.0
    impact_reasons = []

    if wind >= 15:
        if p_pos in ['QB', 'WR', 'TE']:
            penalty = 0.05 if wind < 20 else 0.12
            multiplier -= penalty
            impact_reasons.append(f"High Wind (-{int(penalty*100)}%)")
        elif p_pos == 'RB':
            multiplier += 0.03
            impact_reasons.append("Wind Vol. Boost (+3%)")

    if precip in ['Rain', 'Snow']:
        multiplier -= 0.05
        impact_reasons.append(f"{precip} (-5%)")

    if temp <= 20:
        multiplier -= 0.03
        impact_reasons.append("Extreme Cold (-3%)")

    reason_str = " + ".join(impact_reasons) if impact_reasons else "Fair Weather"
    return round(multiplier, 2), reason_str

# --- CORE LOGIC ---
def get_dynamic_sos(data, stat_col):
    if data.empty: return {}
    league_avg = data[stat_col].mean()
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
    
    # --- FIXED TEAMMATE SELECTION (PRIORITIZING CURRENT STARTER) ---
    # Sort by season and week descending to ensure we find the QB playing NOW
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)].sort_values(['season', 'week'], ascending=False)
    
    if not teammates.empty:
        latest_season = teammates['season'].max()
        
        # 1. QB Selected -> Find Top WR/TE based on the latest available data
        if p_pos == 'QB':
            seasonal_teammates = teammates[teammates['season'] == latest_season]
            valid_targets = seasonal_teammates[seasonal_teammates['position'].isin(['WR', 'TE'])]
            
            if not valid_targets.empty:
                top_target = valid_targets.groupby('player_name')['receiving_yards'].sum().idxmax()
                leg_val = 40 if risk_level == "Conservative (-104)" else 60
                parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
            
        # 2. WR/TE/RB Selected -> Find CURRENT QB
        elif p_pos in ['WR', 'TE', 'RB']:
            current_qbs = teammates[(teammates['position'] == 'QB') & (teammates['season'] == latest_season)]
            
            if not current_qbs.empty:
                # Pick the QB from the most recent game played
                last_week = current_qbs['week'].max()
                qb_name = current_qbs[current_qbs['week'] == last_week]['player_name'].iloc[0]
                
                if p_pos == 'RB':
                    leg_val = 195 if risk_level == "Conservative (-104)" else 240
                    parlay_type = "Team Success"
                else:
                    leg_val = 215 if risk_level == "Conservative (-104)" else 255
                    parlay_type = "QB Link"
                    
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": parlay_type})
                
    return parlay_legs

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

# --- 3. UI RENDERING ---
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
        
        st.divider()
        st.subheader("🌦️ Venue & Weather")
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type = stad_obj.get('roof_type', 'Outdoor') if stad_obj else 'Outdoor'
        
        if roof_type in ['Dome', 'Closed', 'Indoor']:
            st.success(f"🏟️ {roof_type} Facility: Weather Ignored")
            w_wind, w_temp, w_precip = 0, 70, "None"
        else:
            w_wind = st.slider("Wind Speed (MPH)", 0, 40, 5)
            w_temp = st.slider("Temperature (F)", -10, 100, 55)
            w_precip = st.selectbox("Precipitation", ["None", "Rain", "Snow"])

        market_line = st.number_input("Sportsbook Line", value=0.0, step=0.5)

        st.divider()
        st.subheader("⚙️ Parlay Settings")
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
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
        
        weather_multiplier, weather_reason = get_weather_multiplier(roof_type, w_wind, w_temp, w_precip, p_pos)
        model_proj = p_mean * script_boost * sos_multiplier * weather_multiplier

        st.title(f"📊 {selected_p} Intelligence Hub")
        
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            if weather_multiplier != 1.0:
                st.warning(f"⚠️ **Weather Adjustment:** {weather_reason} (x{weather_multiplier})")
            else:
                st.caption(f"☀️ **Weather Condition:** {weather_reason}")

            st.subheader("🏦 Sportsbook Money Tracker")
            fig_money = go.Figure()
            fig_money.add_trace(go.Bar(name='Tickets', x=['Market Sentiment'], y=[65], marker_color='#4a4a4a'))
            fig_money.add_trace(go.Bar(name='Handle', x=['Market Sentiment'], y=[45], marker_color='#00ff96'))
            st.plotly_chart(fig_money, use_container_width=True)

            risk_offsets = {"Conservative (-104)": -0.6, "Standard (+105)": 0, "Aggressive (+200)": 0.6}
            target_line = round(model_proj + (risk_offsets[risk_pref] * p_std))
            
            last_5 = p_df.tail(5).copy()
            last_5['hit'] = last_5[stat_col] >= target_line
            fig_hits = go.Figure(go.Bar(x=[f"Wk {w}" for w in last_5['week']], y=last_5[stat_col], marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_5['hit']]))
            fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b")
            fig_hits.update_layout(title=f"Last 5 Games vs {risk_pref} Target ({target_line}+)", template="plotly_dark", height=250)
            st.plotly_chart(fig_hits, use_container_width=True)

        with col_side:
            st.subheader("📋 Historical Averages")
            avg_data = {"Metric": [stat_label, "Touchdowns"], "Season": [round(p_mean, 1), round(p_df[td_col].mean(), 2)], "Last 5": [round(last_5[stat_col].mean(), 1), round(last_5[td_col].mean(), 2)]}
            st.table(pd.DataFrame(avg_data))
            
            st.divider()
            st.subheader("🚀 Teammate Stacking Parlays")
            t1, t2, t3 = st.tabs(["🛡️ Cons.", "✅ Std.", "🔥 Aggr."])
            
            for tab, r_name in zip([t1, t2, t3], ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"]):
                with tab:
                    parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, p_std, stat_label, data, r_name)
                    for leg in parlay_legs:
                        st.write(f"🔹 **{leg['type']}**: {leg['label']}")
                    
                    tier_v = round(model_proj + (risk_offsets[r_name] * p_std))
                    tier_h = (last_5[stat_col] >= tier_v).sum()
                    st.caption(f"Historical Hit Rate: {tier_h}/5 Games")

            st.divider()
            hit_count = last_5['hit'].sum()
            st.write(f"**Consistency ({risk_pref.split(' ')[0]}):** {hit_count}/5 Games Hit")
            st.progress(hit_count / 5)
