from recommendation_system.recommendation_engine import ShoeRecommendationEngine
from recommendation_system.personalized_services import PersonalizedServices

# Initialize the recommendation engine
engine = ShoeRecommendationEngine()
engine.load_data()

# User Input
print("Welcome to the Shoe Recommendation System!")
user_id = int(input("Enter your user ID: ( SAMPLE DATA RANGE: 1-20 ) "))

if user_id < 1 or user_id > 20:
    print("Invalid user ID. Please enter a user ID between 1 and 20.")
    exit()
    
n_o_r = int(input("Enter the number of recommendations you want: "))

# Get recommendations for a user
recommendations = engine.get_recommendations(user_id=user_id, num_recommendations=n_o_r)
if not recommendations:
    print("No recommendations found for the given user ID.")
else:
    print("Recommended shoes:", recommendations)

# Initialize personalized services
services = PersonalizedServices()
services.load_data()

# Get care notifications
notifications = services.get_care_notifications(user_id=user_id)
if not notifications:
    print("No care notifications found for the given user ID.")
else:
    print("Care notifications:", notifications)