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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .notification-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .notification-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .notification-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
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
        "📊 Analytics", 
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
        st.header("📊 Analytics & Insights")
        
        # Wear pattern analysis
        st.subheader("👁️ Wear Pattern Analysis")
        try:
            analysis = services.analyze_wear_patterns(selected_user)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Shoes Owned", analysis['total_shoes_owned'])
            with col2:
                st.metric("Care Frequency", analysis['care_frequency'].replace('_', ' ').title())
            with col3:
                if analysis['most_worn_types']:
                    most_worn = analysis['most_worn_types'][0]['type']
                    st.metric("Most Worn Type", most_worn.title())
                else:
                    st.metric("Most Worn Type", "N/A")
            
            # Most worn types chart
            if analysis['most_worn_types']:
                st.subheader("📊 Shoe Type Distribution")
                
                types_df = pd.DataFrame(analysis['most_worn_types'])
                
                fig = px.pie(
                    types_df, 
                    values='count', 
                    names='type',
                    title="Distribution of Shoe Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations from analysis
            if analysis['recommendations']:
                st.subheader("💡 Personalized Insights")
                for rec in analysis['recommendations']:
                    st.info(f"💡 {rec}")
                    
        except Exception as e:
            st.error(f"Error analyzing wear patterns: {e}")
        
        st.divider()
        
        # System performance metrics (mock data for demo)
        st.subheader("⚡ System Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Response Time", "0.23s", "↓ 15%")
        with col2:
            st.metric("Recommendation Accuracy", "87.3%", "↑ 2.1%")
        with col3:
            st.metric("User Engagement", "76.8%", "↑ 5.2%")
        with col4:
            st.metric("Click-through Rate", "12.4%", "↑ 3.1%")
    
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

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "👟 Shoe Recommendation System | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()