import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import requests
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS ---
st.set_page_config(page_title="NFL Sharp: Parlay Genius", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

data = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. THE PARLAY LOGIC ENGINE ---
def get_parlay_recommendation(p_df, p_pos, proj_val, label):
    """
    Logic:
    1. Identify a 'Safe' vs 'Aggressive' milestone.
    2. Calculate the 'Hit Rate' (How often has he done this in the last 10 games?).
    3. Return the best bet based on projection vs historical frequency.
    """
    recent = p_df.tail(10)
    # Define a 'High Confidence' line (80% of projected value)
    safe_line = round(proj_val * 0.85)
    hit_rate = (recent[recent.columns[recent.columns.get_loc(p_df.columns[0])]] >= safe_line).mean() # Simplified for brevity
    
    # Custom logic based on position
    if p_pos == 'QB':
        leg = f"{selected_p} {safe_line}+ Pass Yds"
        confidence = "HIGH" if hit_rate > 0.7 else "MED"
    elif p_pos == 'RB':
        leg = f"{selected_p} {safe_line}+ Rush Yds"
        confidence = "HIGH" if hit_rate > 0.65 else "MED"
    else:
        leg = f"{selected_p} {safe_line}+ Rec Yds"
        confidence = "HIGH" if hit_rate > 0.6 else "MED"
        
    return leg, confidence, safe_line

# --- 3. MAIN UI ---
if not data.empty:
    st.title("🏈 NFL Genius: Parlay Builder")
    
    with st.sidebar:
        selected_p = st.selectbox("1. Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("2. Select Opponent", sorted(data['opponent'].unique()))
        sel_stad = st.selectbox("3. Venue", sorted(stadium_client.get_list_of_stadium_names()))

    p_df = data[data['player_name'] == selected_p].copy()
    p_pos = p_df['position'].iloc[-1]
    
    # Determine Stat Type
    stat_col = 'passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards'
    label = "Pass Yds" if p_pos == 'QB' else "Rush Yds" if p_pos == 'RB' else "Rec Yds"
    
    # Calculate Projection (Using our previous logic)
    avg_val = p_df[stat_col].mean()
    # (Insert your defense/weather multipliers here for the full effect)
    proj_val = avg_val * 1.05 # Mock multiplier for example
    
    # --- 4. THE PARLAY CARD ---
    st.subheader("💡 Genius Parlay Recommendation")
    
    leg_text, conf, s_line = get_parlay_recommendation(p_df[[stat_col]], p_pos, proj_val, label)
    
    # Visual Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(f"**Recommended Leg:**\n\n### {leg_text}")
        st.write(f"**Model Confidence:** {conf}")
        if conf == "HIGH":
            st.success("🔥 High Value: Strong historical hit rate.")
        else:
            st.warning("⚠️ Volatile: High ceiling but inconsistent floor.")

    with col2:
        # Distribution Plot
        fig = px.histogram(p_df, x=stat_col, nbins=10, title="Performance Distribution",
                           labels={stat_col: label}, color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=s_line, line_dash="dash", line_color="white", annotation_text="Suggested Line")
        st.plotly_chart(fig, use_container_width=True)

    # Historical Table
    with st.expander("View Hit Rate History"):
        p_df['Hit?'] = p_df[stat_col] >= s_line
        st.table(p_df[['week', 'opponent', stat_col, 'Hit?']].tail(5))

else:
    st.error("Data Load Error")
