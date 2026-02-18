import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Config
st.set_page_config(page_title="Swiggy Recommender", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv'), pd.read_csv('encoded_data.csv')

df, encoded_df = load_data()

st.title("🍴 Swiggy Restaurant Recommendation System")

# User Inputs
with st.sidebar:
    st.header("Search Filters")
    city = st.selectbox("Select your City", df['city'].unique())
    # Extract unique cuisines for the multiselect
    all_cuisines = sorted(list(set([c for sub in df['cuisine'].apply(eval) for c in sub])))
    user_cuisines = st.multiselect("Select Cuisines", all_cuisines)
    budget = st.number_input("Maximum Budget (Cost for two)", value=500)

if st.button("Find Best Matches"):
    # 1. Filter encoded data by city to improve performance
    city_indices = df[df['city'] == city].index
    city_encoded_data = encoded_df.loc[city_indices]
    
    # 2. Build User Vector
    user_vector = np.zeros(encoded_df.shape[1])
    # Set weights for selected cuisines
    for c in user_cuisines:
        if c in encoded_df.columns:
            user_vector[encoded_df.columns.get_loc(c)] = 1
    
    # Set weights for budget and rating
    user_vector[encoded_df.columns.get_loc('cost')] = budget
    user_vector[encoded_df.columns.get_loc('rating')] = 5.0 # Preference for high ratings
    
    # 3. Calculate Similarity
    sim_scores = cosine_similarity([user_vector], city_encoded_data)[0]
    
    # 4. Get Top 5 Results
    top_idx = sim_scores.argsort()[-5:][::-1]
    results = df.iloc[city_indices[top_idx]]
    
    # Display Results
    st.subheader(f"Top 5 Recommendations in {city}")
    for i, row in results.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Rating", f"⭐ {row['rating']}")
            st.write(f"Cost: ₹{row['cost']}")
        with col2:
            st.write(f"### {row['name']}")
            st.write(f"**Cuisines:** {row['cuisine']}")
            st.write(f"📍 {row['address']}")
            st.markdown(f"[Order Now on Swiggy]({row['link']})")
        st.divider()
        