import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import json
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ShoeRecommendationEngine:
    """
    Hybrid shoe recommendation system combining collaborative filtering,
    content-based filtering, and matrix factorization techniques.
    """
    
    def __init__(self, data_path: str = 'data/'):
        self.data_path = data_path
        self.users_df = None
        self.shoes_df = None
        self.interactions_df = None
        
        # Model components
        self.user_item_matrix = None
        self.item_features_matrix = None
        self.svd_model = None
        self.content_similarity_matrix = None
        self.user_similarity_matrix = None
        
        # Scalers and vectorizers
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Hyperparameters
        self.svd_components = 50
        self.min_interactions = 5
        self.similarity_threshold = 0.1
        
    def load_data(self) -> None:
        """Load user interactions, shoe catalog, and user profiles."""
        try:
            # Load interactions data
            self.interactions_df = pd.read_csv(
                os.path.join(self.data_path, 'user_interaction.csv')
            )
            
            # Load shoe catalog
            with open(os.path.join(self.data_path, 'shoe_catalog.json'), 'r') as f:
                shoes_data = json.load(f)
            self.shoes_df = pd.DataFrame(shoes_data)
            
            # Load user profiles
            self.users_df = pd.read_csv(
                os.path.join(self.data_path, 'user_profiles.csv')
            )
            
            print(f"Loaded {len(self.users_df)} users, {len(self.shoes_df)} shoes, "
                  f"{len(self.interactions_df)} interactions")
                  
        except Exception as e:
            print(f"Error loading data: {e}")
            self._generate_sample_data()
    
    def _generate_sample_data(self) -> None:
        """Generate sample data if files don't exist."""
        print("Generating sample data...")
        
        # Sample users
        users_data = []
        for i in range(1, 101):
            users_data.append({
                'user_id': i,
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F', 'Other']),
                'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'])
            })
        self.users_df = pd.DataFrame(users_data)
        
        # Sample shoes
        brands = ['Nike', 'Adidas', 'Puma', 'Reebok', 'Converse', 'Vans', 'New Balance']
        shoe_types = ['sneakers', 'running', 'casual', 'boots', 'sandals', 'dress']
        materials = ['leather', 'canvas', 'synthetic', 'mesh', 'suede']
        colors = ['black', 'white', 'brown', 'red', 'blue', 'gray']
        
        shoes_data = []
        for i in range(1, 201):
            shoes_data.append({
                'shoe_id': i,
                'brand': np.random.choice(brands),
                'model': f'Model_{i}',
                'type': np.random.choice(shoe_types),
                'material': np.random.choice(materials),
                'color': np.random.choice(colors),
                'price': round(np.random.uniform(50, 300), 2),
                'care_requirements': np.random.choice([
                    ['regular_cleaning'], ['waterproofing'], 
                    ['regular_cleaning', 'waterproofing']
                ])
            })
        self.shoes_df = pd.DataFrame(shoes_data)
        
        # Sample interactions
        interactions_data = []
        interaction_types = ['view', 'like', 'purchase', 'wishlist']
        
        for i in range(1, 2001):
            user_id = np.random.randint(1, 101)
            shoe_id = np.random.randint(1, 201)
            interaction_type = np.random.choice(interaction_types)
            rating = np.random.randint(1, 6) if interaction_type in ['purchase', 'like'] else None
            
            interactions_data.append({
                'user_id': user_id,
                'shoe_id': shoe_id,
                'interaction_type': interaction_type,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                'rating': rating
            })
        
        self.interactions_df = pd.DataFrame(interactions_data)
    
    def preprocess_data(self) -> None:
        """Preprocess data for recommendation algorithms."""
        # Create weighted ratings based on interaction types
        interaction_weights = {
            'view': 1,
            'like': 2,
            'wishlist': 3,
            'purchase': 5
        }
        
        self.interactions_df['weight'] = self.interactions_df['interaction_type'].map(interaction_weights)
        self.interactions_df['weighted_rating'] = self.interactions_df.apply(
            lambda x: x['rating'] * x['weight'] if pd.notna(x['rating']) else x['weight'], axis=1
        )
        
        # Create user-item matrix
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='shoe_id', 
            values='weighted_rating',
            aggfunc='mean',
            fill_value=0
        )
        
        # Create item features matrix
        self._create_item_features()
        
        print("Data preprocessing completed.")
    
    def _create_item_features(self) -> None:
        """Create feature matrix for content-based filtering."""
        # Combine text features
        self.shoes_df['combined_features'] = (
            self.shoes_df['brand'].astype(str) + ' ' +
            self.shoes_df['type'].astype(str) + ' ' +
            self.shoes_df['material'].astype(str) + ' ' +
            self.shoes_df['color'].astype(str)
        )
        
        # TF-IDF vectorization
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.shoes_df['combined_features'])
        
        # Price normalization
        price_features = self.scaler.fit_transform(
            self.shoes_df[['price']].fillna(self.shoes_df['price'].mean())
        )
        
        # Combine features
        self.item_features_matrix = np.hstack([tfidf_matrix.toarray(), price_features])
        
        # Calculate content similarity matrix
        self.content_similarity_matrix = cosine_similarity(self.item_features_matrix)
    
    def train_collaborative_filtering(self) -> None:
        """Train collaborative filtering models."""
        # Matrix factorization using SVD
        user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        self.svd_model = TruncatedSVD(
            n_components=min(self.svd_components, min(self.user_item_matrix.shape) - 1),
            random_state=42
        )
        self.user_factors = self.svd_model.fit_transform(user_item_sparse)
        self.item_factors = self.svd_model.components_.T
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_factors)
        
        print("Collaborative filtering model trained.")
    
    def get_user_based_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using user-based collaborative filtering."""
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_recommendations(num_recommendations)
        
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Find similar users
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        
        # Get recommendations from similar users
        recommendations = {}
        user_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        for similar_user_idx in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similarity_score = user_similarities[similar_user_idx]
            
            if similarity_score < self.similarity_threshold:
                continue
                
            similar_user_items = self.user_item_matrix.loc[similar_user_id]
            for shoe_id, rating in similar_user_items.items():
                if rating > 0 and shoe_id not in user_items:
                    if shoe_id not in recommendations:
                        recommendations[shoe_id] = 0
                    recommendations[shoe_id] += rating * similarity_score
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_recommendations]
        
        return self._format_recommendations(sorted_recommendations, 'collaborative_user_based')
    
    def get_content_based_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using content-based filtering."""
        # Get user's interaction history
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ].sort_values('weighted_rating', ascending=False)
        
        if user_interactions.empty:
            return self._get_popular_recommendations(num_recommendations)
        
        # Get top interacted shoes
        top_shoes = user_interactions['shoe_id'].head(5).tolist()
        
        # Calculate recommendations based on content similarity
        recommendations = {}
        
        for shoe_id in top_shoes:
            if shoe_id in self.shoes_df['shoe_id'].values:
                shoe_idx = list(self.shoes_df['shoe_id']).index(shoe_id)
                similarities = self.content_similarity_matrix[shoe_idx]
                
                for idx, similarity in enumerate(similarities):
                    candidate_shoe_id = self.shoes_df.iloc[idx]['shoe_id']
                    if candidate_shoe_id != shoe_id and candidate_shoe_id not in top_shoes:
                        if candidate_shoe_id not in recommendations:
                            recommendations[candidate_shoe_id] = 0
                        recommendations[candidate_shoe_id] += similarity
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_recommendations]
        
        return self._format_recommendations(sorted_recommendations, 'content_based')
    
    def get_hybrid_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get hybrid recommendations combining multiple approaches."""
        # Get recommendations from different approaches
        user_based_recs = self.get_user_based_recommendations(user_id, num_recommendations * 2)
        content_based_recs = self.get_content_based_recommendations(user_id, num_recommendations * 2)
        
        # Combine and weight recommendations
        combined_scores = {}
        
        # Weight collaborative filtering recommendations
        for rec in user_based_recs:
            shoe_id = rec['shoe_id']
            combined_scores[shoe_id] = rec['score'] * 0.6
        
        # Weight content-based recommendations
        for rec in content_based_recs:
            shoe_id = rec['shoe_id']
            if shoe_id in combined_scores:
                combined_scores[shoe_id] += rec['score'] * 0.4
            else:
                combined_scores[shoe_id] = rec['score'] * 0.4
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_recommendations]
        
        return self._format_recommendations(sorted_recommendations, 'hybrid')
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10, 
                          recommendation_type: str = 'hybrid') -> List[Dict]:
        """Main method to get recommendations."""
        if self.user_item_matrix is None:
            self.preprocess_data()
            self.train_collaborative_filtering()
        
        if recommendation_type == 'collaborative':
            return self.get_user_based_recommendations(user_id, num_recommendations)
        elif recommendation_type == 'content':
            return self.get_content_based_recommendations(user_id, num_recommendations)
        else:
            return self.get_hybrid_recommendations(user_id, num_recommendations)
    
    def _get_popular_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """Fallback to popular items for cold start users."""
        popular_items = self.interactions_df.groupby('shoe_id').agg({
            'weighted_rating': 'mean',
            'user_id': 'count'
        }).rename(columns={'user_id': 'interaction_count'})
        
        # Filter items with minimum interactions
        popular_items = popular_items[popular_items['interaction_count'] >= self.min_interactions]
        popular_items = popular_items.sort_values(['weighted_rating', 'interaction_count'], 
                                                 ascending=[False, False])
        
        top_items = popular_items.head(num_recommendations).index.tolist()
        scores = popular_items.head(num_recommendations)['weighted_rating'].tolist()
        
        recommendations = list(zip(top_items, scores))
        return self._format_recommendations(recommendations, 'popular')
    
    def _format_recommendations(self, recommendations: List[Tuple], 
                              algorithm_type: str) -> List[Dict]:
        """Format recommendations with shoe details."""
        formatted_recs = []
        
        for shoe_id, score in recommendations:
            shoe_info = self.shoes_df[self.shoes_df['shoe_id'] == shoe_id]
            
            if not shoe_info.empty:
                shoe_info = shoe_info.iloc[0]
                formatted_recs.append({
                    'shoe_id': int(shoe_id),
                    'brand': shoe_info['brand'],
                    'model': shoe_info['model'],
                    'type': shoe_info['type'],
                    'material': shoe_info['material'],
                    'color': shoe_info['color'],
                    'price': float(shoe_info['price']),
                    'score': float(score),
                    'algorithm': algorithm_type
                })
        
        return formatted_recs
    
    def get_similar_shoes(self, shoe_id: int, num_similar: int = 5) -> List[Dict]:
        """Get shoes similar to a given shoe."""
        if shoe_id not in self.shoes_df['shoe_id'].values:
            return []
        
        shoe_idx = list(self.shoes_df['shoe_id']).index(shoe_id)
        similarities = self.content_similarity_matrix[shoe_idx]
        
        # Get top similar shoes (excluding the shoe itself)
        similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]
        similar_shoes = []
        
        for idx in similar_indices:
            shoe_info = self.shoes_df.iloc[idx]
            similar_shoes.append({
                'shoe_id': int(shoe_info['shoe_id']),
                'brand': shoe_info['brand'],
                'model': shoe_info['model'],
                'type': shoe_info['type'],
                'similarity_score': float(similarities[idx]),
                'price': float(shoe_info['price'])
            })
        
        return similar_shoes
    
    def get_seasonal_recommendations(self, user_id: int, season: str, 
                                   num_recommendations: int = 10) -> List[Dict]:
        """Get season-specific recommendations."""
        season_shoe_types = {
            'spring': ['sneakers', 'casual', 'running'],
            'summer': ['sandals', 'sneakers', 'casual'],
            'fall': ['boots', 'sneakers', 'casual'],
            'winter': ['boots', 'sneakers']
        }
        
        if season not in season_shoe_types:
            return self.get_recommendations(user_id, num_recommendations)
        
        # Filter shoes by seasonal appropriateness
        seasonal_shoes = self.shoes_df[
            self.shoes_df['type'].isin(season_shoe_types[season])
        ]['shoe_id'].tolist()
        
        # Get general recommendations and filter by season
        general_recs = self.get_recommendations(user_id, num_recommendations * 2)
        seasonal_recs = [rec for rec in general_recs if rec['shoe_id'] in seasonal_shoes]
        
        # If not enough seasonal recommendations, add popular seasonal shoes
        if len(seasonal_recs) < num_recommendations:
            popular_seasonal = self._get_popular_seasonal_shoes(season, num_recommendations)
            for shoe in popular_seasonal:
                if shoe not in [rec['shoe_id'] for rec in seasonal_recs]:
                    seasonal_recs.append(shoe)
                    if len(seasonal_recs) >= num_recommendations:
                        break
        
        return seasonal_recs[:num_recommendations]
    
    def _get_popular_seasonal_shoes(self, season: str, num_shoes: int) -> List[Dict]:
        """Get popular shoes for a specific season."""
        season_shoe_types = {
            'spring': ['sneakers', 'casual', 'running'],
            'summer': ['sandals', 'sneakers', 'casual'],
            'fall': ['boots', 'sneakers', 'casual'],
            'winter': ['boots', 'sneakers']
        }
        
        seasonal_shoes = self.shoes_df[
            self.shoes_df['type'].isin(season_shoe_types[season])
        ]
        
        # Get interaction stats for seasonal shoes
        seasonal_interactions = self.interactions_df[
            self.interactions_df['shoe_id'].isin(seasonal_shoes['shoe_id'])
        ]
        
        popular_seasonal = seasonal_interactions.groupby('shoe_id').agg({
            'weighted_rating': 'mean',
            'user_id': 'count'
        }).rename(columns={'user_id': 'interaction_count'})
        
        popular_seasonal = popular_seasonal.sort_values(
            ['weighted_rating', 'interaction_count'], 
            ascending=[False, False]
        ).head(num_shoes)
        
        results = []
        for shoe_id in popular_seasonal.index:
            shoe_info = self.shoes_df[self.shoes_df['shoe_id'] == shoe_id].iloc[0]
            results.append({
                'shoe_id': int(shoe_id),
                'brand': shoe_info['brand'],
                'model': shoe_info['model'],
                'type': shoe_info['type'],
                'score': float(popular_seasonal.loc[shoe_id, 'weighted_rating']),
                'algorithm': f'popular_{season}'
            })
        
        return results
    
    def evaluate_recommendations(self, test_interactions: pd.DataFrame) -> Dict[str, float]:
        """Evaluate recommendation quality using test data."""
        precision_scores = []
        recall_scores = []
        
        for user_id in test_interactions['user_id'].unique():
            if user_id in self.user_item_matrix.index:
                # Get actual items the user interacted with in test set
                actual_items = set(test_interactions[
                    test_interactions['user_id'] == user_id
                ]['shoe_id'].tolist())
                
                # Get recommended items
                recommendations = self.get_recommendations(user_id, 10)
                recommended_items = set([rec['shoe_id'] for rec in recommendations])
                
                # Calculate precision and recall
                if recommended_items:
                    precision = len(actual_items.intersection(recommended_items)) / len(recommended_items)
                    precision_scores.append(precision)
                
                if actual_items:
                    recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
                    recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'num_users_evaluated': len(set(test_interactions['user_id']))
        }


if __name__ == "__main__":
    # Example usage
    engine = ShoeRecommendationEngine()
    engine.load_data()
    
    # Get recommendations for user 1
    recommendations = engine.get_recommendations(user_id=1, num_recommendations=5)
    print("Recommendations for User 1:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['brand']} {rec['model']} - {rec['type']} (Score: {rec['score']:.3f})")
    
    # Get similar shoes
    similar_shoes = engine.get_similar_shoes(shoe_id=1, num_similar=3)
    print(f"\nShoes similar to shoe ID 1:")
    for shoe in similar_shoes:
        print(f"- {shoe['brand']} {shoe['model']} (Similarity: {shoe['similarity_score']:.3f})")
    
    # Get seasonal recommendations
    seasonal_recs = engine.get_seasonal_recommendations(user_id=1, season='winter', num_recommendations=3)
    print(f"\nWinter recommendations for User 1:")
    for rec in seasonal_recs:
        print(f"- {rec['brand']} {rec['model']} - {rec['type']}")
        