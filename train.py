import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

def main():
    print("Loading encoded data...")
    encoded_df = pd.read_csv('encoded_data.csv', dtype=np.float32)
    
    print("Training NearestNeighbors model...")
    # Using cosine similarity for recommendation
    model = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
    model.fit(encoded_df)
    
    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Done! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()
