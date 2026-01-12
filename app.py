import streamlit as st
import nflreadpy as nfl
import pandas as pd
from math import exp
import plotly.express as px

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Genius: Lazy Loader", layout="wide")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. SELECT BEFORE LOADING ---
st.title("🏈 NFL Genius: Pro Builder")

with st.sidebar:
    st.header("📋 Parlay Builder")
    for leg in st.session_state.parlay_legs:
        st.success(f"{leg['player']}: {leg['pick']}")
    if st.button("Reset"):
        st.session_state.parlay_legs = []
        st.rerun()

# Step 1: Input the player name manually or from a simple list
# This prevents the app from crashing trying to "find" players in a broken file
p_name = st.text_input("Enter Player Name (e.g., Justin Jefferson)", "Justin Jefferson")

# --- 3. ON-DEMAND DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_player_data(name):
    try:
        # We only pull 2024-2025 data
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Immediate Flattening to prevent the AttributeError
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
        # Standardize 'player_name'
        if 'player_display_name' in df.columns:
            df = df.rename(columns={'player_display_name': 'player_name'})
            
        # Filter strictly for the chosen player
        p_df = df[df['player_name'].str.contains(name, case=False, na=False)]
        
        # Calculate Defense Stats on the fly for their position
        if not p_df.empty:
            pos = p_df['position'].iloc[-1]
            def_df = df.groupby(['opponent', 'position']).agg({
                'receiving_yards': 'mean',
                'rushing_yards': 'mean',
                'rushing_tds': 'mean',
                'receiving_tds': 'mean'
            }).reset_index()
            return p_df, def_df
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

# Step 2: Only load when the user is ready
if p_name:
    data, def_data = get_player_data(p_name)
    
    if not data.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Analyzing: {p_name}")
            opp = st.selectbox("Opponent", sorted(data['opponent'].unique()))
            line = st.number_input("Vegas Line", value=75.5)
            
            # Defense Mod Logic
            p_pos = data['position'].iloc[-1]
            d_match = def_data[(def_data['opponent'] == opp) & (def_data['position'] == p_pos)]
            mod = 1.15 if (not d_match.empty and d_match['receiving_yards'].iloc[0] > 55) else 0.95
            
            # Projections
            avg_yds = data['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else data['rushing_yards'].mean()
            proj = avg_yds * mod
            edge = ((proj - line) / line) * 100
            
            st.metric("Model Projection", f"{proj:.1f} Yds", delta=f"{edge:.1f}% Edge")
            
            if st.button("Add to Parlay"):
                st.session_state.parlay_legs.append({"player": p_name, "pick": f"Over {line}"})
                st.rerun()

        with col2:
            st.plotly_chart(px.bar(data, x='week', y='receiving_yards', title="Weekly Yards"))
    else:
        st.warning(f"No data found for '{p_name}'. Please check the spelling.")
