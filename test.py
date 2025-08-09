from recommendation_system import ShoeRecommendationEngine, PersonalizedServices

# Get recommendations
engine = ShoeRecommendationEngine()
engine.load_data()
recs = engine.get_recommendations(user_id=1, num_recommendations=5)

# Get personalized services
services = PersonalizedServices()
services.load_data()
notifications = services.get_care_notifications(user_id=1)
print("Recommendations:", recs)
print("Care Notifications:", notifications)