"""
------------------------------------generate_reports.py---------------------------------------------

This file generates business reports from the trained model.
Outputs:
1. profiling.xlsx                    -->Understand each customer segment
2. telecom_association_summary.xlsx  -->Find service combinations to cross-sell
3. segments.xlsx                     -->Final data with segment labels for business use

Run using command: python src/generate_reports.py
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime    

# ----------------Path Setup----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)   #Add src/ folder to Python path

from preprocessing import TelcoPreprocessor   

# ----------------Add default paths ----------------
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'telco_new_cust.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# k values profiled side by side 
PROFILE_K_RANGE = [3, 4, 5, 6, 7]

# variables profiled + how to describe them  for preparing the profiling table
PROFILE_VARS = [
    'tenure', 'age', 'income', 'employ', 'address', 'reside', 'ed',            # continuous
    'retire', 'gender', 'marital',                                            # binary demographics
    'region_1', 'region_2', 'region_3',                                       # region one-hot
    'custcat_1', 'custcat_2', 'custcat_3', 'custcat_4',                       # customer category one-hot
    'tollfree', 'equip', 'callcard', 'wireless', 'multline', 'voice', 'pager',
    'internet', 'callid', 'callwait', 'forward', 'confer', 'ebill',           # services
    'longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon',                   # usage
    'total_services', 'total_mon_usage', 'avg_mon_usage',
    'value_added_services', 'connectivity_services',
    'usage_per_service', 'loyalty_score'                                    # engineered
]

DESCRIPTIONS = {
    'tenure': 'Avg', 'age': 'Avg', 'income': 'Avg', 'employ': 'Avg', 'address': 'Avg',
    'reside': 'Avg', 'ed': 'Avg',
    'retire': '% of retired', 'gender': '% of males', 'marital': '% married',
    'region_1': '% of region', 'region_2': '% of region', 'region_3': '% of region',
    'custcat_1': '% of customers', 'custcat_2': '% of customers',
    'custcat_3': '% of customers', 'custcat_4': '% of customers',
    'tollfree': '% of customers', 'equip': '% of customers', 'callcard': '% of customers',
    'wireless': '% of customers', 'multline': '% of customers', 'voice': '% of customers',
    'pager': '% of customers', 'internet': '% of customers', 'callid': '% of customers',
    'callwait': '% of customers', 'forward': '% of customers', 'confer': '% of customers',
    'ebill': '% of customers',
    'longmon': 'Avg', 'tollmon': 'Avg', 'equipmon': 'Avg', 'cardmon': 'Avg', 'wiremon': 'Avg',
    'total_services': 'Avg', 'total_mon_usage': 'Avg', 'avg_mon_usage': 'Avg',
    'value_added_services': 'Avg', 'connectivity_services': 'Avg',
    'usage_per_service': 'Avg', 'loyalty_score': 'Avg',
}

# -------------------------------Profiling---------------------------------- 

def build_multi_k_profiling(data_encd, telco_final, k_range=PROFILE_K_RANGE):

    df_profile = data_encd.copy()
    km_cols = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(telco_final)
        col = f'KM{k}'
        df_profile[col] = km.labels_
        km_cols.append(col)

    rows = []

    size_row = {'variable': 'Total', 'description': 'Percentage', 'overall': 1.0}
    for kcol in km_cols:
        k = int(kcol.replace('KM', ''))
        vc = df_profile[kcol].value_counts(normalize=True).sort_index()
        for i in range(k):
            size_row[f'{kcol}_{i + 1}'] = round(vc.get(i, 0), 2)
    rows.append(size_row)

    for col in PROFILE_VARS:
        if col not in df_profile.columns:
            continue
        row = {'variable': col, 'description': DESCRIPTIONS.get(col, ''),
               'overall': round(df_profile[col].mean(), 2)}
        for kcol in km_cols:
            k = int(kcol.replace('KM', ''))
            for i in range(k):
                row[f'{kcol}_{i + 1}'] = round(df_profile.loc[df_profile[kcol] == i, col].mean(), 2)
        rows.append(row)

    return pd.DataFrame(rows)

# -------------------Telecom business summary from association rules--------------------------------

def build_telecom_association_summary(data_encd, service_cols,min_support=0.10, lift_threshold=1.7):

    from mlxtend.frequent_patterns import apriori, association_rules

    basket = data_encd[service_cols].astype(bool)
    freq_items = apriori(basket, min_support=min_support, max_len=5, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(columns=['Recommended Service', 'Strong Affinities', 'Best Lift', 'Confidence'])

    rules = association_rules(freq_items, metric='lift', min_threshold=lift_threshold)
    if rules.empty:
        return pd.DataFrame(columns=['Recommended Service', 'Strong Affinities', 'Best Lift', 'Confidence'])

    rules = rules[(rules['confidence'] >= 0.70) & (rules['support'] >= min_support)].copy()
    rules = rules[(rules['antecedents'].apply(len) == 2) & (rules['consequents'].apply(len) == 1)]
    if rules.empty:
        return pd.DataFrame(columns=['Recommended Service', 'Strong Affinities', 'Best Lift', 'Confidence'])

    rules['Antecedent'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['Consequent'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
    rules = rules[['Antecedent', 'Consequent', 'support', 'confidence', 'lift']]
    rules = rules.round({'support': 5, 'confidence': 5, 'lift': 2})
    rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=False, ignore_index=True)

    summary = (rules.sort_values('lift', ascending=False).groupby('Consequent')
    .agg({'Antecedent': lambda x: ' | '.join(x.head(5)), 'lift': 'max', 'confidence': 'max'}).reset_index())
    summary.columns = ['Recommended Service', 'Strong Affinities', 'Best Lift', 'Confidence']
    summary = summary.sort_values(by='Best Lift', ascending=False, ignore_index=True)

    return summary

# ------------------------------- Main Function ----------------------------------------------

def main():
    
    print("Loading trained artifacts + training data...")
    pre = TelcoPreprocessor.load(MODELS_DIR)
    km = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    segment_names = joblib.load(os.path.join(MODELS_DIR, 'segment_names.pkl'))

    df_raw = pd.read_csv(DATA_PATH)
    data_encd, telco_final = pre.transform(df_raw)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # create reports folder with timestamp
    timestamp = datetime.now().strftime("%d-%b-%Y_%I-%M%p")
    report_dir = os.path.join(OUTPUTS_DIR, "reports", f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # ---- 1) Profiling for k = 3..7 --------------------------------------
    print(f"Building profiling table for k = {PROFILE_K_RANGE} ...")
    profiling = build_multi_k_profiling(data_encd, telco_final)
    profiling_path = os.path.join(report_dir, 'profiling.xlsx')    
    profiling.to_excel(profiling_path, index=False)
    print(f"  -> saved {profiling_path}")

    # ---- 2) Telecom business summary (affinities / lift / confidence) ---
    print("Building telecom business summary (service affinities)...")
    try:
        summary = build_telecom_association_summary(data_encd, pre.SERVICE_COLS)
        summary_path = os.path.join(report_dir, 'telecom_association_summary.xlsx')    
        summary.to_excel(summary_path, index=False)
        print(f"  -> saved {summary_path} ({len(summary)} recommended services)")
    except ImportError:
        print("  -> mlxtend not installed, skipping (pip install mlxtend)")

    # ------------------Segments = original data + cluster + segment name --------------------
    print("Saving segmented customers (original columns + cluster + segment)...")
    labels = km.predict(telco_final)
    df_segments = df_raw.copy()
    df_segments['cluster'] = labels
    df_segments['segment'] = df_segments['cluster'].map(segment_names)
    segments_path = os.path.join(report_dir, 'segments.xlsx')   # CHANGED
    df_segments.to_excel(segments_path, index=False)
    print(f"saved {segments_path}")

if __name__ == "__main__":
    main()