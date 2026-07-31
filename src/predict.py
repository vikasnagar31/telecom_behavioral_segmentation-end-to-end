"""
--------------------------------------predict.py---------------------------------------------------------

We use this file to make predictions on new or future customer data.
It does not train the model just apply already saved preprocessing + model  
Run using command: python src/predict.py 
"""
import argparse
import os
import sys
import joblib
import pandas as pd
from datetime import datetime 
from sklearn.metrics import silhouette_score   

# ----------------Path Setup----------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)   #Add src/ folder to Python path

from preprocessing import TelcoPreprocessor

# ----------------Add default paths ----------------

DEFAULT_DATA = os.path.join(PROJECT_ROOT, 'data', 'telco_new_cust.csv')
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'outputs', 'predictions', 'segmented_customers.csv') 

# ---------------- Main Predict Function ----------------

def predict(data_path=DEFAULT_DATA, models_dir=DEFAULT_MODELS_DIR, out_path=DEFAULT_OUT):

    print("Loading trained artifacts...")
    pre = TelcoPreprocessor.load(models_dir)
    km = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
    segment_names = joblib.load(os.path.join(models_dir, 'segment_names.pkl'))

    print("Reading new customer data...")
    df = pd.read_csv(data_path)

    print("Applying preprocessing...")
    data_encd, telco_final = pre.transform(df)    # Apply same preprocessing  

    print("Predicting segments...")
    labels = km.predict(telco_final)      # Predict cluster labels

    df_out = df.copy()
    df_out['cluster'] = labels
    df_out['segment'] = df_out['cluster'].map(segment_names)

    # Silhouette Score (only if more than 1 cluster present) 
    if len(set(labels)) > 1: 
        print(f"Silhouette Score: {silhouette_score(telco_final, labels):.4f}") 
        print("Inertia:", round(km.inertia_, 2))
    else: 
        print("Silhouette Score: Not applicable (only one cluster found)")

    # ---------------- Output Handling ----------------

    # Create folder if not exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # NEW: Add timestamp ONLY if user did not manually pass --out
    if out_path == DEFAULT_OUT:
        timestamp = datetime.now().strftime("%d-%b-%Y_%I-%M%p")   # readable format
        filename = f"pred_{timestamp}.csv"
        out_path = os.path.join(os.path.dirname(out_path), filename)

    df_out.to_csv(out_path, index=False)      # Save results

    print(f"\nSaved {len(df_out)} customers to: {out_path}")

    print("\nSegment Distribution (%):")
    print(df_out['segment'].value_counts(normalize=True).mul(100).round(1))

    return df_out

# --------------Add command line inputs that we can pass during run time if we need----------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict customer segments using trained model")   # Create argument parser

    parser.add_argument('--data', default=DEFAULT_DATA, help='Path to new customer CSV file')   # Input data file
    parser.add_argument('--models_dir', default=DEFAULT_MODELS_DIR, help='Folder where trained model is saved')    # Model folder
    parser.add_argument('--out', default=DEFAULT_OUT, help='Where to save output CSV')   # Output file

    args = parser.parse_args()   # Read inputs from terminal

    predict(args.data, args.models_dir, args.out)   # Run prediction