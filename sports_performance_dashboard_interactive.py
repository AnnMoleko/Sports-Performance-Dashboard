# sports_performance_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

# ========================
# SETUP & DATA LOADING
# ========================
st.set_page_config(
    page_title="🏆 Sports Performance Dashboard",
    layout="wide",
    page_icon="⚽"
)

@st.cache_data
def load_data():
    """Load and validate the dataset."""
    data_file = "dataset_real_players_teams.csv"
    
    if not os.path.exists(data_file):
        st.error(f"❌ Data file not found: {data_file}")
        st.stop()
    
    try:
        df = pd.read_csv(data_file)
        expected_columns = [
            'Player', 'Team', 'Position', 'Device', 'GameID', 
            'HeartRateAvg', 'DistanceCoveredKM', 'TopSpeedKMH',
            'Goals', 'Assists', 'PassesCompleted', 'Tackles',
            'Saves', 'MatchRating'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            st.stop()
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

df = load_data()

# ========================
# SESSION STATE & FILTERS
# ========================
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None
if 'selected_team' not in st.session_state:
    st.session_state.selected_team = None
if 'selected_position' not in st.session_state:
    st.session_state.selected_position = None

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_teams = st.sidebar.multiselect(
    "Select Teams", 
    options=df['Team'].unique(), 
    default=df['Team'].unique()
)
selected_positions = st.sidebar.multiselect(
    "Select Positions", 
    options=df['Position'].unique(), 
    default=df['Position'].unique()
)
selected_devices = st.sidebar.multiselect(
    "Select Devices", 
    options=df['Device'].unique(), 
    default=df['Device'].unique()
)

filtered_df = df[
    (df['Team'].isin(selected_teams)) & 
    (df['Position'].isin(selected_positions)) & 
    (df['Device'].isin(selected_devices))
]

# ========================
# MAIN DASHBOARD LAYOUT
# ========================
st.title("🏆 Interactive Sports Performance Dashboard")
st.markdown("""
Explore player and team performance metrics.  
**Click on charts to filter data dynamically!**
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "👤 Player Analysis", 
    "📈 Team Comparison", 
    "🔮 Predictions"
])

# ========================
# TAB 1: OVERVIEW
# ========================
with tab1:
    st.header("📊 Dataset Overview")
    
    # Key Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Players", len(filtered_df))
        st.metric("Avg. Match Rating", f"{filtered_df['MatchRating'].mean():.2f}")
    with col2:
        st.metric("Avg. Distance Covered", f"{filtered_df['DistanceCoveredKM'].mean():.2f} km")
        st.metric("Avg. Top Speed", f"{filtered_df['TopSpeedKMH'].mean():.2f} km/h")
    
    # Interactive Data Table
    st.subheader("🔍 Click on a Player to Analyze")
    selected_rows = st.dataframe(
        filtered_df.head(10), 
        key="player_table",
        on_select="rerun"
    )
    
    if st.session_state.get('player_table_selected_rows'):
        selected_row = st.session_state.player_table_selected_rows[0]
        st.session_state.selected_player = filtered_df.iloc[selected_row]['Player']
        st.session_state.selected_team = filtered_df.iloc[selected_row]['Team']
        st.session_state.selected_position = filtered_df.iloc[selected_row]['Position']
        st.rerun()
    
    # Interactive Distribution Chart
    st.subheader("📊 Player Distribution by Position & Team")
    fig = px.histogram(
        filtered_df, 
        x='Position', 
        color='Team', 
        barmode='group',
        title="<b>Click on bars to filter players</b>"
    )
    fig.update_layout(clickmode='event+select')
    st.plotly_chart(fig, use_container_width=True, key="position_chart")

# ========================
# TAB 2: PLAYER ANALYSIS
# ========================
with tab2:
    st.header("👤 Individual Player Performance")
    
    # Player Selection
    if st.session_state.selected_player:
        default_player = st.session_state.selected_player
    else:
        default_player = filtered_df['Player'].iloc[0]
    
    selected_player = st.selectbox(
        "Select Player", 
        options=filtered_df['Player'].unique(), 
        index=filtered_df['Player'].tolist().index(default_player) 
        if default_player in filtered_df['Player'].tolist() else 0
    )
    
    player_data = filtered_df[filtered_df['Player'] == selected_player]
    
    if not player_data.empty:
        col1, col2 = st.columns(2)
        
        # Performance Metrics
        with col1:
            st.subheader(f"⚡ {selected_player}'s Key Stats")
            metrics = {
                'Match Rating': player_data['MatchRating'].mean(),
                'Goals': player_data['Goals'].sum(),
                'Assists': player_data['Assists'].sum(),
                'Distance Covered': player_data['DistanceCoveredKM'].mean(),
                'Top Speed': player_data['TopSpeedKMH'].max()
            }
            
            for metric, value in metrics.items():
                unit = 'km/h' if metric == 'Top Speed' else ('km' if metric == 'Distance Covered' else '')
                st.metric(metric, f"{value:.2f} {unit}")
        
        # Radar Chart
        with col2:
            st.subheader("📊 Performance Radar Chart")
            categories = [
                'HeartRateAvg', 'DistanceCoveredKM', 'TopSpeedKMH', 
                'PassesCompleted', 'Tackles', 'Saves'
            ]
            
            normalized_values = [
                (player_data[cat].values[0] / df[cat].max() * 100) 
                if not player_data.empty else 0 
                for cat in categories
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill='toself',
                name=selected_player,
                line_color='blue'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="<b>Normalized Performance Metrics</b>"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Trend
        st.subheader("📈 Match Rating Over Time")
        fig = px.line(
            player_data, 
            x='GameID', 
            y='MatchRating',
            title=f"<b>{selected_player}'s Performance Trend</b>"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available for selected player")

# ========================
# TAB 3: TEAM COMPARISON
# ========================
with tab3:
    st.header("📈 Team Performance Comparison")
    
    # Metric Selection
    metric_options = [
        'MatchRating', 'DistanceCoveredKM', 
        'TopSpeedKMH', 'Goals', 'Assists'
    ]
    selected_metric = st.selectbox(
        "Select Metric", 
        options=metric_options,
        index=0
    )
    
    # Team Comparison Box Plot
    st.subheader(f"📊 {selected_metric} Distribution by Team")
    fig = px.box(
        filtered_df, 
        x='Team', 
        y=selected_metric, 
        color='Position',
        title=f"<b>{selected_metric} Distribution</b>"
    )
    fig.update_layout(clickmode='event+select')
    st.plotly_chart(fig, use_container_width=True, key="team_chart")
    
    # Team Averages Table
    st.subheader("🏆 Team Averages")
    team_avg = filtered_df.groupby('Team').agg({
        'MatchRating': 'mean',
        'DistanceCoveredKM': 'mean',
        'TopSpeedKMH': 'mean',
        'Goals': 'sum',
        'Assists': 'sum'
    }).reset_index()
    
    st.dataframe(
        team_avg.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

# ========================
# TAB 4: PREDICTIVE ANALYTICS
# ========================
with tab4:
    st.header("🔮 Performance Predictor")
    st.markdown("""
    Predict a player's match rating based on their performance metrics.
    The model helps identify which factors influence performance most.
    """)
    
    # Model Training
    ml_df = filtered_df.copy()
    
    # Initialize LabelEncoders and fit them with ALL possible categories (not just filtered_df)
    le_dict = {
        'Position': LabelEncoder().fit(df['Position'].unique()),
        'Team': LabelEncoder().fit(df['Team'].unique()),
        'Device': LabelEncoder().fit(df['Device'].unique())
    }
    
    # Transform columns using pre-fitted encoders
    for col in ['Position', 'Team', 'Device']:
        ml_df[col] = le_dict[col].transform(ml_df[col])
    
    features = [
        'Position', 'Team', 'Device', 'HeartRateAvg', 'DistanceCoveredKM', 
        'TopSpeedKMH', 'Goals', 'Assists', 'PassesCompleted', 'Tackles', 'Saves'
    ]
    target = 'MatchRating'
    
    X = ml_df[features]
    y = ml_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    # Model Evaluation
    with col1:
        st.subheader("🧠 Model Insights")
        st.metric("Prediction Accuracy (MAE)", f"{mae:.2f}")
        
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="<b>Most Important Performance Factors</b>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Interface
    with col2:
        st.subheader("🔮 Make a Prediction")
        
        with st.form("prediction_form"):
            # Use previous selections if available
            default_position = st.session_state.selected_position if st.session_state.selected_position else df['Position'].iloc[0]
            default_team = st.session_state.selected_team if st.session_state.selected_team else df['Team'].iloc[0]
            
            position = st.selectbox(
                "Position", 
                options=df['Position'].unique(),
                index=list(df['Position'].unique()).index(default_position) 
                if default_position in df['Position'].unique() else 0
            )
            
            team = st.selectbox(
                "Team", 
                options=df['Team'].unique(),
                index=list(df['Team'].unique()).index(default_team) 
                if default_team in df['Team'].unique() else 0
            )
            
            device = st.selectbox("Device", options=df['Device'].unique())
            
            col1, col2 = st.columns(2)
            with col1:
                heart_rate = st.slider("Avg Heart Rate", 100, 200, 150)
                distance = st.slider("Distance (km)", 4.0, 12.0, 8.0)
                speed = st.slider("Top Speed (km/h)", 20, 35, 28)
            with col2:
                goals = st.number_input("Goals", 0, 10, 0)
                assists = st.number_input("Assists", 0, 10, 0)
                passes = st.number_input("Passes Completed", 10, 100, 50)
            
            tackles = st.slider("Tackles", 0, 20, 5)
            saves = st.slider("Saves", 0, 15, 3)
            
            if st.form_submit_button("Predict Rating"):
                input_data = pd.DataFrame({
                    'Position': [position],
                    'Team': [team],
                    'Device': [device],
                    'HeartRateAvg': [heart_rate],
                    'DistanceCoveredKM': [distance],
                    'TopSpeedKMH': [speed],
                    'Goals': [goals],
                    'Assists': [assists],
                    'PassesCompleted': [passes],
                    'Tackles': [tackles],
                    'Saves': [saves]
                })
                
                # Use the pre-fitted encoders
                for col in ['Position', 'Team', 'Device']:
                    input_data[col] = le_dict[col].transform(input_data[col])
                
                prediction = model.predict(input_data[features])[0]
                st.success(f"🎯 Predicted Match Rating: {prediction:.2f}/10")

# ========================
# STYLING
# ========================
st.markdown("""
<style>
    /* Main headers */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    /* Subheaders */
    h2 {
        color: #3498db;
        margin-top: 1.5rem !important;
    }
    
    /* Metrics cards */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        padding: 15px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    
    /* Hover effects */
    .stDataFrame tr:hover {
        background-color: #e8f4fc !important;
    }
</style>
""", unsafe_allow_html=True)