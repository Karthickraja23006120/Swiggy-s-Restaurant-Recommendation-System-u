# Swiggy Restaurant Recommendation System
**Student Project Report**

## 1. Introduction
The objective of this project is to build a recommendation system for restaurants using the provided Swiggy dataset. The system predicts and recommends the best restaurants to users based on their personal preferences such as city, cuisine, minimum rating, and maximum cost. A user-friendly Streamlit web application was developed to interface with the recommendation engine.

## 2. Approach

The project was completed in four main phases:

### Phase 1: Data Understanding and Cleaning
- **Dataset Overview**: The raw dataset (`swiggy.csv`) contained 148,541 rows and 11 columns, including both categorical (name, city, cuisine) and numerical-like (rating, rating_count, cost) data.
- **Cleaning Steps**:
  - Removed duplicate records to ensure data integrity.
  - Converted string-based numerical columns into proper numeric types. For example, `rating_count` strings like "100+ ratings" were parsed into `100`, and `cost` strings like "₹200" were parsed into `200`.
  - Dropped rows containing missing values (`NaN` or `--`) to maintain high-quality data for the recommendation model. This resulted in a clean dataset of 61,421 restaurants (`cleaned_data.csv`).

### Phase 2: Data Preprocessing
- Applied One-Hot Encoding to the categorical variables: `city` and `cuisine`. This transformation is necessary because the clustering/similarity models require numerical inputs to calculate distances.
- Saved the trained `OneHotEncoder` object as `encoder.pkl` so it can be applied to new user inputs dynamically.
- Merged the numerical features (`rating`, `rating_count`, `cost`) with the one-hot encoded categorical features to create the final preprocessed dataset (`encoded_data.csv`). To optimize memory usage, `float32` data types were used.

### Phase 3: Recommendation Methodology
- I chose **K-Nearest Neighbors (KNN)** with **Cosine Similarity** as the core algorithm.
- Cosine similarity is highly effective for recommendation systems, especially when dealing with high-dimensional sparse matrices (due to one-hot encoding).
- The `NearestNeighbors` model was trained on the `encoded_data.csv` and saved as `model.pkl`. When a user provides their preferences, the system creates a "dummy" profile, encodes it, and finds the top 10 closest matches in the dataset.

### Phase 4: Streamlit Application
- Built an interactive web application (`app.py`) using Streamlit.
- The sidebar allows users to select their `City` and `Cuisine` from dropdown menus, and use sliders to specify their desired `Minimum Rating` and `Maximum Cost`.
- The engine maps the user's input to the trained Nearest Neighbors model to quickly fetch recommendations and displays them with details like rating, cost, and a link to order.

## 3. Data Analysis & Insights
During the data cleaning and preprocessing stages, a few key insights emerged:
1. **Missing Data**: A significant portion of the raw data contained missing or insufficient ratings (e.g., "Too Few Ratings" or "--"). Removing these ensures that recommendations are only based on well-reviewed restaurants.
2. **Cost & Rating Trade-off**: The dataset features a wide distribution of costs. Scaling or standardizing these numerical features (if doing deeper distance metrics like Euclidean) would be important, but Cosine similarity naturally handles magnitude differences relatively well, focusing on the angle/direction of the feature vectors.
3. **High Dimensionality**: Because the `cuisine` column contains many unique combinations (e.g., "North Indian, South Indian, Chinese"), the one-hot encoded dataset expanded to over 2,400 columns.

## 4. Conclusion
The recommendation system successfully filters and suggests relevant restaurants tailored to the user's explicit criteria. By leveraging One-Hot Encoding and Cosine Similarity through K-Nearest Neighbors, the system handles mixed data types effectively. The integration of Streamlit provides a seamless and interactive experience, fulfilling all business use cases outlined in the project requirements.
