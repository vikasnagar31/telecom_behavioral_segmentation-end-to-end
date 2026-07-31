"""
--------------------------------------train.py------------------------------------------------------------

Trains the customer segmentation model end-to-end and saves everything needed for deployment into models/.
Run using command: 
    python src/train.py  OR  python src/train.py --data data/telco.csv --k 6
"""

import argparse  # used for command line inputs that let us to pass inputs and outputs from terminal line call    
import os
import sys
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------Path Setup----------------

# Get current file location (train.py)  Note: when we use os.path then we don't have to worry about paths being different on Windows vs Mac
SRC_DIR = os.path.dirname(os.path.abspath(__file__))    

# Get project root folder  
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Add src/ folder to Python path so we can import preprocessing.py
sys.path.insert(0, SRC_DIR)

# import our preprocessing class
from preprocessing import TelcoPreprocessor

# ----------------Add default paths----------------
DEFAULT_DATA = os.path.join(PROJECT_ROOT, 'data', 'telco.csv')  # DATA_PATH = "data/telco.csv"
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')  # we can also use MODEL_DIR = "models"

# ---------------- Segment Names ----------------
SEGMENT_NAMES = {
    0: "Basic Users",
    1: "Wireless & Tech Enthusiasts",
    2: "Long-Tenure Traditional Customers",
    3: "Value-Added Service Adopters",
    4: "VIP Customers",
    5: "Heavy Users"
}

# ---------------- Main Training Function ----------------
def train(data_path=DEFAULT_DATA, k=6, models_dir=DEFAULT_MODELS_DIR):

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("Running preprocessing...")
    pre = TelcoPreprocessor()
    data_encd, telco_final = pre.fit_transform(df)
    print("Features used:", list(telco_final.columns))

    # -------- Model Training --------
    print(f"Training KMeans with k={k} ...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(telco_final)

    # -------- Model Evaluation --------
    score = silhouette_score(telco_final, km.labels_)
    print("Silhouette Score:", round(score, 4))
    print("Inertia:", round(km.inertia_, 2))

    # -------- Save Models --------
    os.makedirs(models_dir, exist_ok=True)    
    pre.save(models_dir)                             # save preprocessor.pkl
    joblib.dump(km, os.path.join(models_dir, 'kmeans_model.pkl'))        # save kmeans_model.pkl

    # Save segment names
    if k == 6:
        names_for_k = SEGMENT_NAMES
    else:
        names_for_k = {i: f"Segment {i}" for i in range(k)}
     
    joblib.dump(names_for_k, os.path.join(models_dir, 'segment_names.pkl'))   # save segment_names.pkl

    # -------- Attach results --------
    data_encd['cluster'] = km.labels_
    data_encd['segment'] = data_encd['cluster'].map(names_for_k)

    print("\nSegment Distribution:")
    print(data_encd['segment'].value_counts(normalize=True) * 100)
    print("\nModels saved in:", models_dir)

    return pre, km


# ----------------Add command line inputs that we can pass during run time if we need----------------
if __name__ == "__main__":

    # Create arguments parser (reads command line input)
    parser = argparse.ArgumentParser(description="Train segmentation model")

    # Add arguments user can pass
    parser.add_argument('--data', default=DEFAULT_DATA, help='Path to CSV file')
    parser.add_argument('--k', type=int, default=6, help='Number of clusters')
    parser.add_argument('--models_dir', default=DEFAULT_MODELS_DIR, help='Save folder')

    # Read arguments from terminal
    args = parser.parse_args()

    # Call training function with user inputs
    train(args.data, args.k, args.models_dir)