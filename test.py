from recommendation_system.recommendation_engine import ShoeRecommendationEngine
from recommendation_system.personalized_services import PersonalizedServices

# Initialize the recommendation engine
engine = ShoeRecommendationEngine()
engine.load_data()

# Get recommendations for a user
recommendations = engine.get_recommendations(user_id=1, num_recommendations=5)
print("Recommended shoes:", recommendations)

# Initialize personalized services
services = PersonalizedServices()
services.load_data()

# Get care notifications
notifications = services.get_care_notifications(user_id=1)
print("Care notifications:", notifications)