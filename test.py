from recommendation_system.recommendation_engine import ShoeRecommendationEngine
from recommendation_system.personalized_services import PersonalizedServices

def main():
    # Initialize the recommendation engine
    engine = ShoeRecommendationEngine()
    engine.load_data()

    print("Welcome to the Shoe Recommendation System!")

    # User Input
    try:
        user_id = int(input("Enter your user ID: ( SAMPLE DATA RANGE: 1-20 ) "))
        if not 1 <= user_id <= 20:
            print("Invalid user ID. Please enter a user ID between 1 and 20.")
            return
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 20.")
        return

    try:
        n_o_r = int(input("Enter the number of recommendations you want: "))
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # Get recommendations for a user
    recommendations = engine.get_recommendations(user_id=user_id, num_recommendations=n_o_r)
    if not recommendations:
        print("No recommendations found for the given user ID.")
    else:
        print("Recommended shoes:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['brand']} {rec['model']} - {rec['type']} (Score: {rec['score']:.3f})")

    # Get similar shoes (example: shoe_id=1)
    similar_shoes = engine.get_similar_shoes(shoe_id=1, num_similar=3)
    print(f"\nShoes similar to shoe ID 1:")
    for shoe in similar_shoes:
        print(f"- {shoe['brand']} {shoe['model']} (Similarity: {shoe['similarity_score']:.3f})")

    # Seasonal recommendations example
    seasonal_recs = engine.get_seasonal_recommendations(user_id=user_id, season='winter', num_recommendations=3)
    print(f"\nWinter recommendations for User {user_id}:")
    for rec in seasonal_recs:
        print(f"- {rec['brand']} {rec['model']} - {rec['type']}")

    # Initialize personalized services
    services = PersonalizedServices()
    services.load_data()

    # Get care notifications
    notifications = services.get_care_notifications(user_id=user_id)
    if not notifications:
        print("\nNo care notifications found.")
    else:
        print("\nCare notifications:")
        for note in notifications:
            print(f"- {note['type']}: {note['message']}")

    # Analyze wear patterns
    analysis = services.analyze_wear_patterns(user_id=user_id)
    print(f"\nWear Pattern Analysis for User {user_id}:")
    print(f"Total shoes owned: {analysis['total_shoes_owned']}")
    print(f"Care frequency: {analysis['care_frequency']}")
    print("Most worn types:", analysis['most_worn_types'])

    # Replacement suggestions
    replacements = services.get_replacement_suggestions(user_id=user_id)
    print("\nReplacement Suggestions:")
    for suggestion in replacements:
        current = suggestion['current_shoe']
        print(f"- Replace {current['brand']} {current['type']} "
              f"(Wear: {current['wear_level']*100:.0f}%)")

    # Seasonal care recommendations
    seasonal_care = services.get_seasonal_care_recommendations(user_id=user_id, season='winter')
    print(f"\nWinter Care Recommendations:")
    for rec in seasonal_care[:2]:  # Only first 2
        print(f"- {rec['shoe_brand']} {rec['shoe_type']}: {rec['message']}")

if __name__ == "__main__":
    main()
