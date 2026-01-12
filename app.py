import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Player Stats", layout="wide")
st.title("🏈 NFL Player Performance Tracker")

# --- 2. DATA LOADING ---
@st.cache_data(ttl=3600)
def load_simple_data():
    try:
        # Load recent seasons
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Flatten columns if they are MultiIndex (common in nflreadpy 2025/2026)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
        
        # Standardize the name column
        # Mapping common variants to 'player_name'
        name_cols = ['player_display_name', 'player_name', 'player']
        for col in name_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'player_name'})
                break
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

data = load_simple_data()

# --- 3. SEARCH & FILTER ---
if not data.empty:
    # Sidebar for simple navigation
    st.sidebar.header("Settings")
    
    # Player Search Input
    p_query = st.text_input("Search for a Player (e.g., Jefferson)", "Jefferson")
    
    if p_query:
        # Case-insensitive fuzzy search
        matches = data[data['player_name'].str.contains(p_query, case=False, na=False)]
        
        if not matches.empty:
            # Let user select the exact player from search results
            player_list = sorted(matches['player_name'].unique())
            selected_player = st.selectbox("Select Player", player_list)
            
            # Filter data for selected player
            p_df = data[data['player_name'] == selected_player]
            
            # --- 4. DISPLAY STATS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Stats for {selected_player}")
                # Determine position to show relevant yards
                pos = p_df['position'].iloc[-1]
                yard_col = 'receiving_yards' if pos in ['WR', 'TE'] else 'rushing_yards'
                
                avg_yards = p_df[yard_col].mean()
                st.metric(f"Avg {yard_col.replace('_', ' ').title()}", f"{avg_yards:.1f}")
                
                st.write(p_df[['week', 'opponent_team', yard_col]])

            with col2:
                # Simple visual trend
                fig = px.line(p_df, x='week', y=yard_col, 
                             title=f"{selected_player} {yard_col.replace('_', ' ').title()} Trend",
                             markers=True)
                st.plotly_chart(fig)
        else:
            st.warning(f"No players found matching '{p_query}'")
else:
    st.info("Waiting for NFL data to sync...")
