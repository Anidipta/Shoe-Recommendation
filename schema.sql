CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER CHECK (age >= 0 AND age <= 120),
    gender VARCHAR(20),
    location VARCHAR(100),
    shoe_size_preference DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shoe_brands (
    brand_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(100) UNIQUE NOT NULL,
    brand_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shoe_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) UNIQUE NOT NULL,
    parent_category_id INTEGER REFERENCES shoe_categories(category_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shoes (
    shoe_id SERIAL PRIMARY KEY,
    brand_id INTEGER REFERENCES shoe_brands(brand_id) NOT NULL,
    category_id INTEGER REFERENCES shoe_categories(category_id) NOT NULL,
    model VARCHAR(255) NOT NULL,
    shoe_type VARCHAR(100) NOT NULL,
    material VARCHAR(100),
    primary_color VARCHAR(50),
    secondary_color VARCHAR(50),
    price DECIMAL(10,2) CHECK (price >= 0),
    available_sizes TEXT[],
    care_requirements TEXT[],
    waterproof BOOLEAN DEFAULT FALSE,
    seasonal_suitability TEXT[],
    target_gender VARCHAR(20),
    release_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    shoe_id INTEGER REFERENCES shoes(shoe_id) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL CHECK (interaction_type IN ('view', 'like', 'purchase', 'wishlist', 'cart_add', 'review')),
    interaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    purchase_price DECIMAL(10,2),
    shoe_size DECIMAL(3,1),
    session_id VARCHAR(255),
    device_type VARCHAR(50),
    INDEX idx_user_interactions_user_id (user_id),
    INDEX idx_user_interactions_shoe_id (shoe_id),
    INDEX idx_user_interactions_timestamp (interaction_timestamp),
    INDEX idx_user_interactions_type (interaction_type)
);

CREATE TABLE shoe_care_history (
    care_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    shoe_id INTEGER REFERENCES shoes(shoe_id) NOT NULL,
    care_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    care_type VARCHAR(100) NOT NULL,
    care_mode VARCHAR(50),
    duration_minutes INTEGER,
    device_model VARCHAR(100),
    care_effectiveness_rating INTEGER CHECK (care_effectiveness_rating >= 1 AND care_effectiveness_rating <= 5),
    notes TEXT,
    INDEX idx_shoe_care_user_id (user_id),
    INDEX idx_shoe_care_shoe_id (shoe_id),
    INDEX idx_shoe_care_date (care_date)
);

CREATE TABLE user_shoe_ownership (
    ownership_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    shoe_id INTEGER REFERENCES shoes(shoe_id) NOT NULL,
    purchase_date DATE,
    purchase_price DECIMAL(10,2),
    shoe_size DECIMAL(3,1),
    current_condition VARCHAR(50) DEFAULT 'new',
    estimated_wear_level INTEGER DEFAULT 0 CHECK (estimated_wear_level >= 0 AND estimated_wear_level <= 100),
    last_worn_date DATE,
    wear_frequency_per_month INTEGER DEFAULT 0,
    is_favorite BOOLEAN DEFAULT FALSE,
    retirement_date DATE,
    UNIQUE(user_id, shoe_id, purchase_date)
);

CREATE TABLE recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    shoe_id INTEGER REFERENCES shoes(shoe_id) NOT NULL,
    recommendation_type VARCHAR(100) NOT NULL,
    recommendation_score DECIMAL(5,4) CHECK (recommendation_score >= 0 AND recommendation_score <= 1),
    algorithm_used VARCHAR(100),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    presented_at TIMESTAMP,
    clicked BOOLEAN DEFAULT FALSE,
    purchased BOOLEAN DEFAULT FALSE,
    feedback_rating INTEGER CHECK (feedback_rating >= 1 AND feedback_rating <= 5),
    context_data JSONB,
    INDEX idx_recommendations_user_id (user_id),
    INDEX idx_recommendations_generated_at (generated_at),
    INDEX idx_recommendations_score (recommendation_score)
);

CREATE TABLE user_preferences (
    preference_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    preferred_brands TEXT[],
    preferred_colors TEXT[],
    preferred_materials TEXT[],
    preferred_shoe_types TEXT[],
    price_range_min DECIMAL(10,2),
    price_range_max DECIMAL(10,2),
    comfort_priority INTEGER CHECK (comfort_priority >= 1 AND comfort_priority <= 10),
    style_priority INTEGER CHECK (style_priority >= 1 AND style_priority <= 10),
    durability_priority INTEGER CHECK (durability_priority >= 1 AND durability_priority <= 10),
    eco_friendly_preference BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE environmental_data (
    env_id SERIAL PRIMARY KEY,
    user_location VARCHAR(100),
    date DATE,
    weather_condition VARCHAR(50),
    temperature_celsius INTEGER,
    humidity_percentage INTEGER,
    precipitation_mm DECIMAL(5,2),
    seasonal_category VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_env_location_date (user_location, date)
);

CREATE TABLE notification_logs (
    notification_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) NOT NULL,
    notification_type VARCHAR(100) NOT NULL,
    notification_content TEXT NOT NULL,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    action_taken BOOLEAN DEFAULT FALSE,
    related_shoe_id INTEGER REFERENCES shoes(shoe_id),
    priority_level INTEGER DEFAULT 1 CHECK (priority_level >= 1 AND priority_level <= 5)
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shoes_updated_at BEFORE UPDATE ON shoes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();