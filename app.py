import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add the recommendation_system to path
sys.path.append('.')

from recommendation_system import ShoeRecommendationEngine, PersonalizedServices

# Configure Streamlit page
st.set_page_config(
    page_title="Shoe Recommendation System",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main Layout & Typography */
    body {
        background: linear-gradient(135deg, #0a0c10 0%, #0d1117 100%);
        color: #c9d1d9;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #58a6ff, #1f6feb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }

    /* Enhanced Tab Styling */
    .stTabs {
        background: linear-gradient(180deg, #161b22, #0d1117);
        padding: 1.5rem 1rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(88, 166, 255, 0.1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        background: linear-gradient(180deg, #21262d, #1a1f24);
        padding: 0.75rem;
        border-radius: 1rem;
        gap: 8px;
        border: 1px solid rgba(88, 166, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 2.5rem;
        width: 100%;
        justify-content: center;
        padding: 1rem 2rem;
        background: linear-gradient(145deg, #1a1f24, #21262d);
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 1rem;
        color: #8b949e;
        transition: all 0.3s ease;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(145deg, #21262d, #2d333b);
        border-color: rgba(88, 166, 255, 0.3);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1f6feb, #58a6ff) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
    }

    /* Recommendation Cards with Glow Effects */
    .recommendation-card {
        background: linear-gradient(145deg, #161b22, #1a1f24);
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(88, 166, 255, 0.1);
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transition: all 0.4s ease;
    }

    .recommendation-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
        box-shadow: 0 12px 28px rgba(88, 166, 255, 0.2);
    }

    /* Enhanced Notification Styles */
    .notification-high {
        background: linear-gradient(135deg, #3d1f1f 0%, #161b22 100%);
        border-left: 5px solid #f85149;
        box-shadow: 0 4px 12px rgba(248, 81, 73, 0.2);
        padding: 1.5rem;
        border-radius: 0 1rem 1rem 0;
    }

    .notification-medium {
        background: linear-gradient(135deg, #3d331f 0%, #161b22 100%);
        border-left: 5px solid #d29922;
        box-shadow: 0 4px 12px rgba(210, 153, 34, 0.2);
        padding: 1.5rem;
        border-radius: 0 1rem 1rem 0;
    }

    .notification-low {
        background: linear-gradient(135deg, #1f3d25 0%, #161b22 100%);
        border-left: 5px solid #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.2);
        padding: 1.5rem;
        border-radius: 0 1rem 1rem 0;
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #1f6feb, #58a6ff);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(31, 111, 235, 0.4);
    }

    /* Metric Containers with Glass Effect */
    .metric-container {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(33, 38, 45, 0.9));
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(88, 166, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: linear-gradient(180deg, #0d1117, #161b22);
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #1f6feb, #58a6ff);
        border-radius: 6px;
        border: 3px solid #0d1117;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #58a6ff, #1f6feb);
    }

    /* Animation Keyframes */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(88, 166, 255, 0.2); }
        100% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.4); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
    st.session_state.services = None
    st.session_state.data_loaded = False

@st.cache_resource
def load_recommendation_engine():
    """Load and cache the recommendation engine."""
    engine = ShoeRecommendationEngine()
    engine.load_data()
    return engine

@st.cache_resource
def load_personalized_services():
    """Load and cache the personalized services."""
    services = PersonalizedServices()
    services.load_data()
    return services

def display_recommendation_card(rec):
    """Display a recommendation in a nice card format."""
    st.markdown(f"""
    <div class="recommendation-card">
        <h4>👟 {rec['brand']} {rec['model']}</h4>
        <p><strong>Type:</strong> {rec['type'].title()} | <strong>Material:</strong> {rec['material'].title()} | <strong>Color:</strong> {rec['color'].title()}</p>
        <p><strong>Price:</strong> ${rec['price']:.2f} | <strong>Score:</strong> {rec['score']:.3f}</p>
        <p><small><em>Algorithm: {rec['algorithm']}</em></small></p>
    </div>
    """, unsafe_allow_html=True)

def display_notification(notification):
    """Display a notification with appropriate styling."""
    priority = notification['priority']
    css_class = f"notification-{priority}"
    
    priority_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}
    
    st.markdown(f"""
    <div class="{css_class}">
        <strong>{priority_emoji.get(priority, '📌')} {notification['type'].replace('_', ' ').title()}</strong><br>
        {notification['message']}<br>
        <small><em>Recommended Action: {notification.get('recommended_action', 'N/A')}</em></small>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">👟 Shoe Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading recommendation system..."):
            st.session_state.engine = load_recommendation_engine()
            st.session_state.services = load_personalized_services()
            st.session_state.data_loaded = True
        st.success("✅ System loaded successfully!")
    
    engine = st.session_state.engine
    services = st.session_state.services
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # User selection
    user_ids = list(range(1, 21))  # Assuming 20 users in sample data
    selected_user = st.sidebar.selectbox("Select User ID", user_ids, index=0)
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", 
        "🎯 Recommendations", 
        "🔔 Notifications", 
        "💾 Database",  # Changed from Analytics to Database
        "⚙️ System Info"
    ])
    
    with tab1:
        st.header(f"📊 Dashboard for User {selected_user}")
        
        # User info (if available)
        if engine.users_df is not None and selected_user in engine.users_df['user_id'].values:
            user_info = engine.users_df[engine.users_df['user_id'] == selected_user].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", user_info.get('age', 'N/A'))
            with col2:
                st.metric("Gender", user_info.get('gender', 'N/A'))
            with col3:
                st.metric("Location", user_info.get('location', 'N/A'))
            with col4:
                st.metric("Shoe Size", user_info.get('shoe_size_preference', 'N/A'))
        
        st.divider()
        
        # Quick stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Quick Recommendations")
            try:
                quick_recs = engine.get_recommendations(selected_user, 3, 'hybrid')
                if quick_recs:
                    for rec in quick_recs:
                        st.write(f"👟 **{rec['brand']} {rec['model']}** - ${rec['price']:.2f} (Score: {rec['score']:.3f})")
                else:
                    st.info("No recommendations available")
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
        
        with col2:
            st.subheader("🔔 Priority Notifications")
            try:
                notifications = services.get_care_notifications(selected_user)
                high_priority = [n for n in notifications if n['priority'] == 'high']
                
                if high_priority:
                    for notif in high_priority[:3]:
                        st.warning(f"🚨 {notif['message']}")
                else:
                    st.success("✅ No urgent notifications")
            except Exception as e:
                st.error(f"Error getting notifications: {e}")
    
    with tab2:
        st.header("🎯 Shoe Recommendations")
        
        # Recommendation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_type = st.selectbox(
                "Recommendation Type", 
                ["hybrid", "collaborative", "content"],
                help="Choose the algorithm type for recommendations"
            )
        
        with col2:
            num_recs = st.slider("Number of Recommendations", 1, 20, 5)
        
        with col3:
            season = st.selectbox(
                "Season (Optional)", 
                ["None", "spring", "summer", "fall", "winter"]
            )
        
        # Generate recommendations button
        if st.button("🚀 Generate Recommendations", type="primary"):
            try:
                with st.spinner("Generating recommendations..."):
                    if season != "None":
                        recommendations = engine.get_seasonal_recommendations(
                            selected_user, season.lower(), num_recs
                        )
                    else:
                        recommendations = engine.get_recommendations(
                            selected_user, num_recs, rec_type
                        )
                
                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"#{i} - {rec['brand']} {rec['model']} (Score: {rec['score']:.3f})"):
                            display_recommendation_card(rec)
                else:
                    st.warning("No recommendations found for this user.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
        
        st.divider()
        
        # Similar shoes section
        st.subheader("🔍 Find Similar Shoes")
        
        if engine.shoes_df is not None:
            shoe_ids = engine.shoes_df['shoe_id'].tolist()
            selected_shoe = st.selectbox("Select a shoe to find similar ones", shoe_ids)
            
            if st.button("Find Similar Shoes"):
                try:
                    similar_shoes = engine.get_similar_shoes(selected_shoe, 5)
                    if similar_shoes:
                        st.write("**Similar shoes:**")
                        for shoe in similar_shoes:
                            st.write(f"👟 {shoe['brand']} {shoe['model']} - {shoe['type']} (Similarity: {shoe['similarity_score']:.3f})")
                    else:
                        st.info("No similar shoes found")
                except Exception as e:
                    st.error(f"Error finding similar shoes: {e}")
    
    with tab3:
        st.header("🔔 Personalized Notifications")
        
        # Care notifications
        st.subheader("🧽 Care Notifications")
        try:
            notifications = services.get_care_notifications(selected_user)
            
            if notifications:
                for notification in notifications:
                    display_notification(notification)
            else:
                st.success("✅ No care notifications at this time!")
                
        except Exception as e:
            st.error(f"Error getting care notifications: {e}")
        
        st.divider()
        
        # Replacement suggestions
        st.subheader("🔄 Replacement Suggestions")
        try:
            replacements = services.get_replacement_suggestions(selected_user)
            
            if replacements:
                for replacement in replacements:
                    urgency_color = "🔴" if replacement['replacement_urgency'] == 'high' else "🟡"
                    st.write(f"{urgency_color} **Replace:** {replacement['current_shoe']['brand']} {replacement['current_shoe']['type']}")
                    st.write(f"   Wear Level: {replacement['current_shoe']['wear_level']*100:.0f}%")
                    st.write(f"   Reason: {replacement['reason']}")
                    st.write(f"   Suggested Features: {', '.join(replacement['suggested_features'])}")
                    st.write("---")
            else:
                st.success("✅ No replacement suggestions at this time!")
                
        except Exception as e:
            st.error(f"Error getting replacement suggestions: {e}")
        
        st.divider()
        
        # Seasonal care recommendations
        st.subheader("🌤️ Seasonal Care Recommendations")
        season_select = st.selectbox("Select Season", ["spring", "summer", "fall", "winter"])
        
        if st.button("Get Seasonal Care Tips"):
            try:
                seasonal_care = services.get_seasonal_care_recommendations(selected_user, season_select)
                
                if seasonal_care:
                    for care_rec in seasonal_care[:3]:  # Show top 3
                        st.write(f"👟 **{care_rec['shoe_brand']} {care_rec['shoe_type']}**")
                        st.write(f"🎯 Focus: {care_rec['focus_area']}")
                        st.write(f"💡 {care_rec['message']}")
                        if care_rec['specific_tips']:
                            st.write("**Tips:**")
                            for tip in care_rec['specific_tips']:
                                st.write(f"  • {tip}")
                        st.write("---")
                else:
                    st.info("No seasonal care recommendations available")
                    
            except Exception as e:
                st.error(f"Error getting seasonal care recommendations: {e}")
    
    with tab4:
        st.markdown('<h2 style="text-align: center; color: #58a6ff;">💾 Database Schema</h2>', unsafe_allow_html=True)
        
        # Add enhanced CSS for SQL display
        st.markdown("""
        <style>
            .sql-container {
                background: linear-gradient(145deg, #161b22, #1a1f24);
                border-radius: 1rem;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(88, 166, 255, 0.1);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            .sql-title {
                color: #58a6ff;
                font-size: 1.5rem;
                margin-bottom: 1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .sql-code {
                background: #0d1117;
                padding: 1rem;
                border-radius: 0.5rem;
                font-family: 'Consolas', monospace;
                border: 1px solid #30363d;
                color: #e6edf3;
            }
            .keyword { color: #ff7b72; }
            .type { color: #79c0ff; }
            .constraint { color: #d2a8ff; }
            .symbol { color: #79c0ff; }
            .erd-container {
                background: linear-gradient(145deg, #161b22, #1a1f24);
                border-radius: 1rem;
                padding: 2rem;
                margin: 2rem 0;
                border: 1px solid rgba(88, 166, 255, 0.1);
            }
            .relationship-arrow {
                color: #58a6ff;
                font-size: 1.2rem;
            }
        </style>
        """, unsafe_allow_html=True)


        # Display SQL schemas with syntax highlighting
        col1, col2 = st.columns(2)
        
        with col1:
            # Users table
            st.markdown("""
            <div class="sql-container">
                <div class="sql-title">👤 Users Table</div>
                <div class="sql-code">
<span class="keyword">CREATE TABLE</span> users (
<br>    user_id         <span class="type">INTEGER</span> <span class="constraint">PRIMARY KEY</span>,
<br>    name           <span class="type">VARCHAR(100)</span>,
<br>    shoe_size      <span class="type">FLOAT</span>,
<br>    activity_level <span class="type">VARCHAR(50)</span>,
<br>    preferences    <span class="type">JSONB</span>,
<br>    created_at     <span class="type">TIMESTAMP</span> <span class="keyword">DEFAULT</span> CURRENT_TIMESTAMP
);</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations table
            st.markdown("""
            <div class="sql-container">
                <div class="sql-title">🎯 Recommendations Table</div>
                <div class="sql-code">
<span class="keyword">CREATE TABLE</span> recommendations (
<br>    recommendation_id <span class="type">INTEGER</span> <span class="constraint">PRIMARY KEY</span>,
<br>    user_id          <span class="type">INTEGER</span> <span class="constraint">REFERENCES</span> users(user_id),
<br>    shoe_id          <span class="type">INTEGER</span> <span class="constraint">REFERENCES</span> shoes(shoe_id),
<br>    score            <span class="type">FLOAT</span>,
<br>    algorithm        <span class="type">VARCHAR(50)</span>,
<br>    created_at       <span class="type">TIMESTAMP</span> <span class="keyword">DEFAULT</span> CURRENT_TIMESTAMP,
    <span class="constraint">UNIQUE</span>(user_id, shoe_id)
);</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Shoes table
            st.markdown("""
            <div class="sql-container">
                <div class="sql-title">👟 Shoes Table</div>
                <div class="sql-code">
<span class="keyword">CREATE TABLE</span> shoes (
<br>    shoe_id     <span class="type">INTEGER</span> <span class="constraint">PRIMARY KEY</span>,
<br>    brand      <span class="type">VARCHAR(100)</span>,
<br>    model      <span class="type">VARCHAR(100)</span>,
<br>    type       <span class="type">VARCHAR(50)</span>,
<br>    material   <span class="type">VARCHAR(50)</span>,
<br>    color      <span class="type">VARCHAR(50)</span>,
<br>    price      <span class="type">DECIMAL(10,2)</span>,
<br>    sizes      <span class="type">FLOAT[]</span>,
<br>    stock      <span class="type">INTEGER</span>,
<br>    created_at <span class="type">TIMESTAMP</span> <span class="keyword">DEFAULT</span> CURRENT_TIMESTAMP
);</div>
            </div>
            """, unsafe_allow_html=True)
            
            # User Interactions table
            st.markdown("""
            <div class="sql-container">
                <div class="sql-title">🤝 User Interactions Table</div>
                <div class="sql-code">
<span class="keyword">CREATE TABLE</span> user_interactions (
<br>    interaction_id <span class="type">INTEGER</span> <span class="constraint">PRIMARY KEY</span>,
<br>    user_id       <span class="type">INTEGER</span> <span class="constraint">REFERENCES</span> users(user_id),
<br>    shoe_id      <span class="type">INTEGER</span> <span class="constraint">REFERENCES</span> shoes(shoe_id),
<br>    action_type   <span class="type">VARCHAR(50)</span>,
<br>    rating       <span class="type">INTEGER</span>,
<br>    feedback     <span class="type">TEXT</span>,
<br>    created_at   <span class="type">TIMESTAMP</span> <span class="keyword">DEFAULT</span> CURRENT_TIMESTAMP
);</div>
            </div>
            """, unsafe_allow_html=True)

        # Display key relationships
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; background: #161b22; padding: 1.5rem; border-radius: 1rem; border: 1px solid rgba(88, 166, 255, 0.1);">
            <h3 style="color: #58a6ff; margin-bottom: 1rem;">Key Relationships</h3>
            <p style="color: #8b949e; margin-bottom: 0.5rem;">• Users → Recommendations → Shoes (Many-to-Many)</p>
            <p style="color: #8b949e; margin-bottom: 0.5rem;">• Users → Interactions → Shoes (Many-to-Many)</p>
            <p style="color: #8b949e;">• All tables include timestamps for temporal analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.header("⚙️ System Information")
        
        # Data overview
        st.subheader("📈 Data Overview")
        
        if engine.users_df is not None and engine.shoes_df is not None and engine.interactions_df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Users", len(engine.users_df))
                st.metric("Total Shoes", len(engine.shoes_df))
                
            with col2:
                st.metric("Total Interactions", len(engine.interactions_df))
                if services.care_history_df is not None:
                    st.metric("Care Records", len(services.care_history_df))
                else:
                    st.metric("Care Records", "N/A")
            
            with col3:
                # Show interaction type distribution
                interaction_counts = engine.interactions_df['interaction_type'].value_counts()
                most_common = interaction_counts.index[0]
                st.metric("Most Common Interaction", most_common.title())
        
        st.divider()
        
        # Algorithm information
        st.subheader("🧠 Algorithm Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recommendation Algorithms:**")
            st.write("• Hybrid (Collaborative + Content-based)")
            st.write("• Matrix Factorization (SVD)")
            st.write("• Content-based Filtering (TF-IDF)")
            st.write("• User-based Collaborative Filtering")
            
        with col2:
            st.write("**Personalized Services:**")
            st.write("• Proactive Care Notifications")
            st.write("• Wear Pattern Analysis")
            st.write("• Replacement Suggestions")
            st.write("• Seasonal Care Recommendations")
        
        st.divider()
        
        # Raw data inspection (expandable)
        with st.expander("🔍 Inspect Raw Data"):
            data_type = st.selectbox("Select Data Type", ["Users", "Shoes", "Interactions", "Care History"])
            
            if data_type == "Users" and engine.users_df is not None:
                st.dataframe(engine.users_df.head(10))
            elif data_type == "Shoes" and engine.shoes_df is not None:
                st.dataframe(engine.shoes_df.head(10))
            elif data_type == "Interactions" and engine.interactions_df is not None:
                st.dataframe(engine.interactions_df.head(10))
            elif data_type == "Care History" and services.care_history_df is not None:
                st.dataframe(services.care_history_df.head(10))
            else:
                st.info("Data not available")

if __name__ == "__main__":
    main()