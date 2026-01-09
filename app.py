@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    # 1. Load Data
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # 2. Fix Team Column Name (Handle KeyError)
    # Check what the team column is actually named in this version
    if 'recent_team' not in weekly.columns:
        if 'team' in weekly.columns:
            weekly = weekly.rename(columns={'team': 'recent_team'})
        elif 'team_abbr' in weekly.columns:
            weekly = weekly.rename(columns={'team_abbr': 'recent_team'})
    
    # 3. Clean and Prep
    weekly = weekly.dropna(subset=['player_name'])
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
    
    # Calculate rough Target Share if not present
    team_tgts = weekly.groupby(['recent_team', 'season', 'week'])['targets'].transform('sum')
    weekly['target_share'] = (weekly['targets'] / team_tgts).fillna(0)
    weekly['wopr'] = weekly['target_share'] * 2.0 

    # 4. Chronological Sorting for Rolling Averages
    weekly = weekly.sort_values(['player_name', 'season', 'week'])
    for col in ['passing_yards', 'total_scrimmage_yards', 'target_share']:
        weekly[f'{col}_roll3'] = weekly.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())
    
    # 5. Merge with Schedule (Environmental Data)
    # Ensure 'wind' exists in schedule before selecting
    sched_cols = ['season', 'week', 'home_team', 'temp', 'surface']
    if 'wind' in sched.columns:
        sched_cols.append('wind')
    
    df = weekly.merge(sched[sched_cols], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], 
                      how='left')
    
    # 6. Final Clean-up
    if 'wind' not in df.columns: df['wind'] = 0
    df['wind'] = df['wind'].fillna(0)
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    
    return df
