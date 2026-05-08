import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Swiggy Restaurant Recommender", layout="wide", page_icon="")

# --- LOAD DATA AND MODELS ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    return df

@st.cache_resource
def load_models():
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoder, model

def main():
    st.title(" Swiggy Restaurant Recommendation System")
    st.markdown("Find the perfect restaurant based on your preferences!")
    
    # Load data
    try:
        df = load_data()
        encoder, model = load_models()
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        st.info("Please make sure you have run data_prep.py and train.py first.")
        return

    # Extract unique values for UI
    cities = sorted(df['city'].dropna().unique().tolist())
    # We'll just take the top 100 cuisines or just unique ones
    # Since cuisines are one-hot encoded exactly as they appear in the column
    cuisines = sorted(df['cuisine'].dropna().unique().tolist())
    
    # --- SIDEBAR FOR USER INPUT ---
    st.sidebar.header("Your Preferences")
    
    selected_city = st.sidebar.selectbox("City", cities)
    selected_cuisine = st.sidebar.selectbox("Cuisine Preference", cuisines)
    
    selected_rating = st.sidebar.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    
    # For cost, let's give a slider up to the max cost in a reasonable range
    max_cost = int(df['cost'].max())
    selected_cost = st.sidebar.slider("Maximum Cost for Two (₹)", min_value=50, max_value=max_cost, value=500, step=50)
    
    # For rating_count, maybe an average or reasonable default
    avg_rating_count = df['rating_count'].mean()
    
    if st.sidebar.button("Recommend Restaurants"):
        with st.spinner("Finding the best spots for you..."):
            # Create a dataframe for the user input
            input_data = pd.DataFrame({
                'city': [selected_city],
                'cuisine': [selected_cuisine],
                'rating': [selected_rating],
                'rating_count': [avg_rating_count],
                'cost': [selected_cost]
            })
            
            # Encode categorical features
            try:
                encoded_cats = encoder.transform(input_data[['city', 'cuisine']])
                cat_columns = encoder.get_feature_names_out(['city', 'cuisine'])
                encoded_df = pd.DataFrame(encoded_cats, columns=cat_columns)
            except Exception as e:
                st.error(f"Error encoding input: {e}")
                return
            
            # Combine numerical and categorical
            num_df = input_data[['rating', 'rating_count', 'cost']]
            final_input = pd.concat([num_df, encoded_df], axis=1)
            
            # Since the model expects float32
            final_input = final_input.astype(np.float32)
            
            # Find nearest neighbors
            distances, indices = model.kneighbors(final_input, n_neighbors=10)
            
            # Map indices back to cleaned data
            # NearestNeighbors returns indices based on the encoded_df which matches cleaned_data
            recommendations = df.iloc[indices[0]].copy()
            
            # We can also filter the recommendations slightly to enforce the max cost or min rating 
            # if we wanted to, but the NN model will find the mathematically closest ones.
            # It's better to show what it found and maybe highlight them.
            
            st.subheader(f"Top Recommendations in {selected_city}")
            
            # Display results in a nice format
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {i+1}. {row['name']}")
                        st.markdown(f"**Cuisine:** {row['cuisine']}")
                        st.markdown(f"**Address:** {row['address']}")
                    with col2:
                        st.markdown(f"** {row['rating']}** ({row['rating_count']} ratings)")
                        st.markdown(f"**Cost:** ₹{row['cost']}")
                        if pd.notna(row['link']):
                            st.markdown(f"[Order Here]({row['link']})")
                    st.divider()

if __name__ == "__main__":
    main()
