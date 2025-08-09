import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os


class PersonalizedServices:
    """
    Personalized service system for shoe care notifications, wear pattern analysis,
    and proactive recommendations based on user behavior patterns.
    """
    
    def __init__(self, data_path: str = 'data/'):
        self.data_path = data_path
        self.users_df = None
        self.shoes_df = None
        self.interactions_df = None
        self.care_history_df = None
        
        # Service parameters
        self.care_frequency_threshold = 30  # days
        self.wear_intensity_threshold = 0.7  # 70% wear level
        self.replacement_threshold = 0.85   # 85% wear level
        
    def load_data(self) -> None:
        """Load data required for personalized services."""
        try:
            # Load existing data from recommendation engine
            self.interactions_df = pd.read_csv(
                os.path.join(self.data_path, 'user_interaction.csv')
            )
            
            with open(os.path.join(self.data_path, 'shoe_catalog.json'), 'r') as f:
                shoes_data = json.load(f)
            self.shoes_df = pd.DataFrame(shoes_data)
            
            self.users_df = pd.read_csv(
                os.path.join(self.data_path, 'user_profiles.csv')
            )
            
            # Generate care history data
            self._generate_care_history()
            
        except Exception as e:
            print(f"Error loading data for personalized services: {e}")
            self._generate_sample_data()
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for personalized services."""
        print("Generating sample personalized services data...")
        
        # This would typically be loaded from the recommendation engine
        # For demo purposes, create minimal sample data
        users_data = []
        for i in range(1, 51):
            users_data.append({
                'user_id': i,
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F', 'Other'])
            })
        self.users_df = pd.DataFrame(users_data)
        
        # Minimal shoe data
        self.shoes_df = pd.DataFrame([
            {'shoe_id': i, 'brand': f'Brand_{i%5}', 'type': 'sneakers', 
             'care_requirements': ['regular_cleaning']} 
            for i in range(1, 101)
        ])
        
        # Sample interactions
        interactions_data = []
        for i in range(1, 501):
            interactions_data.append({
                'user_id': np.random.randint(1, 51),
                'shoe_id': np.random.randint(1, 101),
                'interaction_type': 'purchase',
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        self.interactions_df = pd.DataFrame(interactions_data)
        
        self._generate_care_history()
    
    def _generate_care_history(self) -> None:
        """Generate sample shoe care history data."""
        care_data = []
        care_types = ['cleaning', 'waterproofing', 'polishing', 'repair']
        
        # Get purchased shoes from interactions
        purchased_shoes = self.interactions_df[
            self.interactions_df['interaction_type'] == 'purchase'
        ]
        
        for _, interaction in purchased_shoes.iterrows():
            user_id = interaction['user_id']
            shoe_id = interaction['shoe_id']
            purchase_date = pd.to_datetime(interaction['timestamp'])
            
            # Generate care events for this shoe
            num_care_events = np.random.poisson(3)  # Average 3 care events per shoe
            
            for _ in range(num_care_events):
                care_date = purchase_date + timedelta(
                    days=np.random.randint(1, (datetime.now() - purchase_date).days + 1)
                )
                
                care_data.append({
                    'user_id': user_id,
                    'shoe_id': shoe_id,
                    'care_date': care_date,
                    'care_type': np.random.choice(care_types),
                    'days_since_last_care': np.random.randint(7, 60),
                    'estimated_wear_level': np.random.uniform(0.1, 0.9)
                })
        
        self.care_history_df = pd.DataFrame(care_data)
        if not self.care_history_df.empty:
            self.care_history_df['care_date'] = pd.to_datetime(self.care_history_df['care_date'])
    
    def get_care_notifications(self, user_id: int) -> List[Dict]:
        """
        Get proactive care notifications for user's shoes based on:
        - Time since last care
        - Estimated wear level
        - Shoe usage patterns
        """
        notifications = []
        
        if self.care_history_df is None or self.care_history_df.empty:
            return notifications
        
        # Get user's shoes and their care history
        user_shoes = self._get_user_shoes(user_id)
        
        for shoe_id in user_shoes:
            shoe_care_history = self.care_history_df[
                (self.care_history_df['user_id'] == user_id) &
                (self.care_history_df['shoe_id'] == shoe_id)
            ].sort_values('care_date')
            
            if shoe_care_history.empty:
                # No care history - recommend initial care
                notifications.append({
                    'type': 'initial_care',
                    'shoe_id': shoe_id,
                    'message': 'Consider setting up initial care routine for your new shoes',
                    'priority': 'medium',
                    'recommended_action': 'Schedule basic cleaning and protection treatment'
                })
                continue
            
            # Get latest care record
            latest_care = shoe_care_history.iloc[-1]
            days_since_last_care = (datetime.now() - latest_care['care_date']).days
            
            # Check if care is overdue
            if days_since_last_care > self.care_frequency_threshold:
                shoe_info = self._get_shoe_info(shoe_id)
                notifications.append({
                    'type': 'overdue_care',
                    'shoe_id': shoe_id,
                    'shoe_brand': shoe_info.get('brand', 'Unknown'),
                    'days_overdue': days_since_last_care - self.care_frequency_threshold,
                    'message': f'Your {shoe_info.get("brand", "")} shoes need care - {days_since_last_care} days since last maintenance',
                    'priority': 'high' if days_since_last_care > 45 else 'medium',
                    'recommended_action': self._get_recommended_care_action(shoe_id)
                })
            
            # Check wear level
            if latest_care['estimated_wear_level'] > self.wear_intensity_threshold:
                shoe_info = self._get_shoe_info(shoe_id)
                notifications.append({
                    'type': 'high_wear',
                    'shoe_id': shoe_id,
                    'shoe_brand': shoe_info.get('brand', 'Unknown'),
                    'wear_level': latest_care['estimated_wear_level'],
                    'message': f'High wear detected on your {shoe_info.get("brand", "")} shoes ({latest_care["estimated_wear_level"]*100:.0f}% worn)',
                    'priority': 'high',
                    'recommended_action': 'Consider professional maintenance or replacement evaluation'
                })
        
        return sorted(notifications, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    def analyze_wear_patterns(self, user_id: int) -> Dict:
        """
        Analyze user's shoe wear patterns to provide insights:
        - Most frequently worn shoe types
        - Seasonal preferences
        - Care habits
        - Replacement patterns
        """
        analysis = {
            'user_id': user_id,
            'total_shoes_owned': 0,
            'most_worn_types': [],
            'care_frequency': 'unknown',
            'replacement_pattern': 'unknown',
            'recommendations': []
        }
        
        # Get user's shoes
        user_shoes = self._get_user_shoes(user_id)
        analysis['total_shoes_owned'] = len(user_shoes)
        
        if not user_shoes:
            return analysis
        
        # Analyze shoe types
        shoe_types = []
        for shoe_id in user_shoes:
            shoe_info = self._get_shoe_info(shoe_id)
            if shoe_info:
                shoe_types.append(shoe_info.get('type', 'unknown'))
        
        if shoe_types:
            type_counts = pd.Series(shoe_types).value_counts()
            analysis['most_worn_types'] = [
                {'type': shoe_type, 'count': count} 
                for shoe_type, count in type_counts.head(3).items()
            ]
        
        # Analyze care frequency
        if not self.care_history_df.empty:
            user_care_history = self.care_history_df[
                self.care_history_df['user_id'] == user_id
            ]
            
            if not user_care_history.empty:
                avg_days_between_care = user_care_history['days_since_last_care'].mean()
                
                if avg_days_between_care <= 21:
                    analysis['care_frequency'] = 'excellent'
                elif avg_days_between_care <= 35:
                    analysis['care_frequency'] = 'good'
                elif avg_days_between_care <= 50:
                    analysis['care_frequency'] = 'moderate'
                else:
                    analysis['care_frequency'] = 'needs_improvement'
        
        # Generate recommendations based on patterns
        analysis['recommendations'] = self._generate_pattern_recommendations(user_id, analysis)
        
        return analysis
    
    def get_replacement_suggestions(self, user_id: int) -> List[Dict]:
        """
        Suggest shoe replacements based on:
        - High wear levels
        - Age of shoes
        - Usage frequency
        """
        suggestions = []
        
        if self.care_history_df is None or self.care_history_df.empty:
            return suggestions
        
        user_shoes = self._get_user_shoes(user_id)
        
        for shoe_id in user_shoes:
            shoe_care_history = self.care_history_df[
                (self.care_history_df['user_id'] == user_id) &
                (self.care_history_df['shoe_id'] == shoe_id)
            ]
            
            if not shoe_care_history.empty:
                latest_care = shoe_care_history.iloc[-1]
                
                # Check if replacement is needed
                if latest_care['estimated_wear_level'] > self.replacement_threshold:
                    shoe_info = self._get_shoe_info(shoe_id)
                    
                    suggestions.append({
                        'shoe_id': shoe_id,
                        'current_shoe': {
                            'brand': shoe_info.get('brand', 'Unknown'),
                            'type': shoe_info.get('type', 'Unknown'),
                            'wear_level': latest_care['estimated_wear_level']
                        },
                        'replacement_urgency': 'high' if latest_care['estimated_wear_level'] > 0.9 else 'medium',
                        'suggested_features': self._get_replacement_features(shoe_info),
                        'reason': f"Current wear level: {latest_care['estimated_wear_level']*100:.0f}%"
                    })
        
        return suggestions
    
    def get_seasonal_care_recommendations(self, user_id: int, season: str) -> List[Dict]:
        """
        Get seasonal care recommendations based on weather and shoe types.
        """
        seasonal_care_map = {
            'spring': {
                'focus': 'cleaning',
                'actions': ['deep_clean', 'waterproof', 'inspect_damage'],
                'message': 'Spring cleaning time! Prepare your shoes for the active season.'
            },
            'summer': {
                'focus': 'protection',
                'actions': ['uv_protection', 'ventilation_check', 'sweat_treatment'],
                'message': 'Protect your shoes from summer heat and moisture.'
            },
            'fall': {
                'focus': 'weatherproofing',
                'actions': ['waterproof', 'sole_check', 'material_treatment'],
                'message': 'Prepare your shoes for wet weather and changing conditions.'
            },
            'winter': {
                'focus': 'preservation',
                'actions': ['salt_protection', 'insulation_check', 'deep_condition'],
                'message': 'Protect your shoes from harsh winter conditions.'
            }
        }
        
        if season not in seasonal_care_map:
            return []
        
        recommendations = []
        user_shoes = self._get_user_shoes(user_id)
        seasonal_care = seasonal_care_map[season]
        
        for shoe_id in user_shoes:
            shoe_info = self._get_shoe_info(shoe_id)
            
            recommendations.append({
                'shoe_id': shoe_id,
                'shoe_brand': shoe_info.get('brand', 'Unknown'),
                'shoe_type': shoe_info.get('type', 'Unknown'),
                'season': season,
                'focus_area': seasonal_care['focus'],
                'recommended_actions': seasonal_care['actions'],
                'message': seasonal_care['message'],
                'specific_tips': self._get_seasonal_tips(shoe_info, season)
            })
        
        return recommendations
    
    def _get_user_shoes(self, user_id: int) -> List[int]:
        """Get list of shoes owned by a user."""
        if self.interactions_df is None:
            return []
        
        purchased_shoes = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ]['shoe_id'].unique().tolist()
        
        return purchased_shoes
    
    def _get_shoe_info(self, shoe_id: int) -> Dict:
        """Get shoe information by ID."""
        if self.shoes_df is None:
            return {}
        
        shoe_info = self.shoes_df[self.shoes_df['shoe_id'] == shoe_id]
        
        if not shoe_info.empty:
            return shoe_info.iloc[0].to_dict()
        return {}
    
    def _get_recommended_care_action(self, shoe_id: int) -> str:
        """Get recommended care action based on shoe type and requirements."""
        shoe_info = self._get_shoe_info(shoe_id)
        
        if not shoe_info:
            return "Basic cleaning and inspection"
        
        care_requirements = shoe_info.get('care_requirements', ['regular_cleaning'])
        
        if 'waterproofing' in care_requirements:
            return "Clean and apply waterproof treatment"
        elif 'polishing' in care_requirements:
            return "Clean and polish"
        else:
            return "Clean and condition"
    
    def _generate_pattern_recommendations(self, user_id: int, analysis: Dict) -> List[str]:
        """Generate recommendations based on wear pattern analysis."""
        recommendations = []
        
        # Care frequency recommendations
        if analysis['care_frequency'] == 'needs_improvement':
            recommendations.append(
                "Consider setting up a regular shoe care schedule - aim for maintenance every 2-3 weeks"
            )
        elif analysis['care_frequency'] == 'excellent':
            recommendations.append(
                "Great job maintaining your shoes! Your care routine is exemplary"
            )
        
        # Shoe type diversity recommendations
        if len(analysis['most_worn_types']) == 1:
            recommendations.append(
                f"You primarily wear {analysis['most_worn_types'][0]['type']} - consider diversifying your collection for different occasions"
            )
        
        # Collection size recommendations
        if analysis['total_shoes_owned'] < 3:
            recommendations.append(
                "Consider expanding your shoe collection to rotate wear and extend shoe lifespan"
            )
        elif analysis['total_shoes_owned'] > 20:
            recommendations.append(
                "You have an extensive collection! Focus on maintaining your favorites and consider donating unused pairs"
            )
        
        return recommendations
    
    def _get_replacement_features(self, current_shoe_info: Dict) -> List[str]:
        """Suggest features to look for in replacement shoes."""
        features = ['improved_durability']
        
        shoe_type = current_shoe_info.get('type', '')
        
        if shoe_type == 'running':
            features.extend(['better_cushioning', 'breathable_material'])
        elif shoe_type == 'boots':
            features.extend(['waterproof_construction', 'reinforced_sole'])
        elif shoe_type == 'dress':
            features.extend(['premium_leather', 'classic_design'])
        else:
            features.extend(['comfort_features', 'versatile_style'])
        
        return features
    
    def _get_seasonal_tips(self, shoe_info: Dict, season: str) -> List[str]:
        """Get specific seasonal care tips based on shoe type."""
        shoe_type = shoe_info.get('type', '')
        material = shoe_info.get('material', '')
        
        tips = []
        
        if season == 'winter':
            if 'leather' in material.lower():
                tips.append("Use leather conditioner to prevent cracking in cold weather")
            tips.append("Clean salt stains immediately to prevent damage")
            
        elif season == 'summer':
            tips.append("Allow shoes to air dry completely between wears")
            if shoe_type == 'sneakers':
                tips.append("Consider using antibacterial sprays to prevent odor")
                
        elif season == 'spring':
            tips.append("Deep clean to remove winter buildup")
            tips.append("Check for any damage from winter wear")
            
        elif season == 'fall':
            tips.append("Apply waterproofing treatment before wet weather")
            tips.append("Inspect soles for adequate tread")
        
        return tips


if __name__ == "__main__":
    # Example usage
    services = PersonalizedServices()
    services.load_data()
    
    # Get care notifications for user 1
    notifications = services.get_care_notifications(user_id=1)
    print("Care Notifications for User 1:")
    for notification in notifications:
        print(f"- {notification['type']}: {notification['message']}")
    
    # Analyze wear patterns
    analysis = services.analyze_wear_patterns(user_id=1)
    print(f"\nWear Pattern Analysis for User 1:")
    print(f"Total shoes owned: {analysis['total_shoes_owned']}")
    print(f"Care frequency: {analysis['care_frequency']}")
    print("Most worn types:", analysis['most_worn_types'])
    
    # Get replacement suggestions
    replacements = services.get_replacement_suggestions(user_id=1)
    print(f"\nReplacement Suggestions:")
    for suggestion in replacements:
        print(f"- Replace {suggestion['current_shoe']['brand']} {suggestion['current_shoe']['type']} (Wear: {suggestion['current_shoe']['wear_level']*100:.0f}%)")
    
    # Get seasonal recommendations
    seasonal_care = services.get_seasonal_care_recommendations(user_id=1, season='winter')
    print(f"\nWinter Care Recommendations:")
    for rec in seasonal_care[:2]:  # Show first 2
        print(f"- {rec['shoe_brand']} {rec['shoe_type']}: {rec['message']}")