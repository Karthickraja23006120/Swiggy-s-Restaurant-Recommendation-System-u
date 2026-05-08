import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import re

def clean_rating(x):
    if pd.isna(x) or x == '--':
        return np.nan
    try:
        return float(x)
    except:
        return np.nan

def clean_rating_count(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower().replace('ratings', '').replace('+', '').strip()
    if 'too few' in x:
        return 0
    if 'k' in x:
        try:
            return float(x.replace('k', '')) * 1000
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def clean_cost(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    # Extract numbers from string like '₹200', '₹ 200 for two'
    nums = re.findall(r'\d+', x)
    if nums:
        return float(nums[0])
    return np.nan

def main():
    print("Loading data...")
    df = pd.read_csv('swiggy.csv')
    
    print("Initial shape:", df.shape)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Cleaning columns
    df['rating'] = df['rating'].apply(clean_rating)
    df['rating_count'] = df['rating_count'].apply(clean_rating_count)
    df['cost'] = df['cost'].apply(clean_cost)
    
    # Drop rows with missing values
    # The instructions say "Impute or drop rows with missing values."
    # Let's drop them to keep it clean.
    df.dropna(subset=['id', 'name', 'city', 'rating', 'rating_count', 'cost', 'cuisine'], inplace=True)
    
    # Reset index and keep matching indices
    df.reset_index(drop=True, inplace=True)
    
    print("Cleaned shape:", df.shape)
    
    # Save cleaned data
    df.to_csv('cleaned_data.csv', index=False)
    print("Saved cleaned_data.csv")
    
    # Preprocessing - One-Hot Encoding
    # The categorical features are 'city' and 'cuisine'
    print("Encoding categorical features...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[['city', 'cuisine']])
    
    # Save encoder
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("Saved encoder.pkl")
    
    # Numerical features
    num_df = df[['rating', 'rating_count', 'cost']].reset_index(drop=True)
    
    # Create a DataFrame for encoded categorical features
    cat_columns = encoder.get_feature_names_out(['city', 'cuisine'])
    encoded_df = pd.DataFrame(encoded_cats, columns=cat_columns)
    
    # Combine numerical and encoded features
    final_encoded_df = pd.concat([num_df, encoded_df], axis=1)
    
    # Ensure indices match
    assert df.index.equals(final_encoded_df.index)
    
    # Save encoded data
    final_encoded_df.to_csv('encoded_data.csv', index=False)
    print("Saved encoded_data.csv")
    
if __name__ == "__main__":
    main()
