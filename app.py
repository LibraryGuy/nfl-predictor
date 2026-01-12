import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
from math import exp

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Genius Debugger", layout="wide")

st.title("🏈 NFL Genius: Pro Builder")

# Sidebar for Parlay
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

with st.sidebar:
    st.header("📋 Parlay Builder")
    for leg in st.session_state.parlay_legs:
        st.success(f"{leg['player']}: {leg['pick']}")
    if st.button("Reset Parlay"):
        st.session_state.parlay_legs = []
        st.rerun()

# --- 2. INPUT & SEARCH ---
col_search, col_debug = st.columns([2, 1])
with col_search:
    p_input = st.text_input("Player Name (Try 'Jefferson' or 'Mahomes')", "Jefferson")
with col_debug:
    show_debug = st.checkbox("🐞 Peek at Raw Data")

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_nfl_data():
    try:
        # Loading 2024 and 2025
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # FIX 1: Flatten MultiIndex (The AttributeError culprit)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
        
        # FIX 2: Standardize Name Columns
        # nflreadpy often uses 'player_display_name'
        if 'player_display_name' in df.columns and 'player_name' not in df.columns:
            df['player_name'] = df['player_display_name']
            
        return df.fillna(0)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_nfl_data()

# --- 4. DEBUGGER & FILTER ---
if show_debug and not data.empty:
    st.write("### Data Preview (First 5 Rows)")
    st.dataframe(data.head())
    st.write("### Available Columns:")
    st.write(list(data.columns))

if not data.empty and p_input:
    # We use a case-insensitive 'contains' search to find "Justin Jefferson" even if you type "Jefferson"
    results = data[data['player_name'].str.contains(p_input, case=False, na=False)]
    
    if not results.empty:
        # If multiple players match (e.g., 'Jones'), let user pick one
        unique_matches = results['player_name'].unique()
        selected_p = st.selectbox("Confirm Player", unique_matches)
        
        # Filter down to the specific player
        p_df = data[data['player_name'] == selected_p]
        p_pos = p_df['position'].iloc[-1]
        
        # --- UI LAYOUT ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"Analytics for {selected_p}")
            opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
            line = st.number_input("Market Line (Yards)", value=70.5)
            
            # Simple Math: Project Yards based on Average
            avg_yds = p_df['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_df['rushing_yards'].mean()
            st.metric("Season Average", f"{avg_yds:.1f} Yds")
            
            if st.button("Add to Parlay"):
                st.session_state.parlay_legs.append({"player": selected_p, "pick": f"Over {line}"})
                st.rerun()

        with c2:
            fig = px.bar(p_df, x='week', y='receiving_yards' if p_pos in ['WR', 'TE'] else 'rushing_yards', 
                         title=f"Weekly Production")
            st.plotly_chart(fig)
    else:
        st.error(f"Could not find '{p_input}' in the dataset. Try a shorter name (e.g., just 'Jefferson').")
