"""
---------------------------------------app/app.py-------------------------------------------------------

Streamlit app for the Telecom Customer Segmentation project.
Run using command: streamlit run app/app.py
What it does:
1. Loads the model saved by src/train.py.
2. Lets a user upload a CSV of customers and scores each one into a segment.
3. Shows KPI numbers, a few segment-wise charts, and the full result table.
4. Lets the user download the scored CSV, and (in the sidebar) the latest
   profiling / association-summary reports created by src/generate_reports.py.
"""

import os
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

#----------------------path setup----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from preprocessing import TelcoPreprocessor  

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports')
SAMPLE_FILE = os.path.join(PROJECT_ROOT, 'data', 'telco_new_cust.csv')

# Segments treated as "high value" for the KPI below. Matches the segment
# names train.py assigns when k=6 (see SEGMENT_NAMES in src/train.py).
HIGH_VALUE_SEGMENTS = ["VIP Customers", "Heavy Users"]

st.set_page_config(page_title="Telecom Customer Segmentation", page_icon="📡", layout="wide")

col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/128/18619/18619563.png", width=100)
with col2:
    st.title("Telecom Customer Segmentation")

st.write("Upload a CSV of customers (same columns as the training data) and this "
         "app assigns each customer to a behavioural segment, then summarizes the "
         "results with a few key numbers and charts.")

# ---------------------------------------------------------------- artifacts
@st.cache_resource       # Cache the model and preprocessor so they reload fast
def load_artifacts():
    pre = TelcoPreprocessor.load(MODELS_DIR)
    km = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    segment_names = joblib.load(os.path.join(MODELS_DIR, 'segment_names.pkl'))
    return pre, km, segment_names


if not os.path.exists(os.path.join(MODELS_DIR, 'kmeans_model.pkl')):
    st.error(
        "No trained model found in `models/`. Train one first:\n\n"
        "```\npython src/train.py\n```"
    )
    st.stop()

pre, km, segment_names = load_artifacts()


# ------------------------------------------------------------------ sidebar
def find_latest_report_dir():
    """outputs/reports/ holds one timestamped folder per generate_reports.py
    run. Return the most recent one, or None if it doesn't exist yet."""
    if not os.path.isdir(REPORTS_DIR):
        return None
    subdirs = [os.path.join(REPORTS_DIR, d) for d in os.listdir(REPORTS_DIR)
               if os.path.isdir(os.path.join(REPORTS_DIR, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


with st.sidebar:
    st.header("About the segments")
    for cluster_id, name in segment_names.items():
        st.markdown(f"**{cluster_id} — {name}**")
    st.caption(
        f"Model: K-Means (k={km.n_clusters}). Features: monthly usage, "
        "value-added & connectivity service counts, usage-per-service ratio, "
        "and a tenure * income loyalty score."
    )

    st.markdown("---")
    st.header("📥 Download reports")
    latest_report_dir = find_latest_report_dir()
    if latest_report_dir:
        for filename, label in [
            ('profiling.xlsx', 'Profiling (k = 3 to 7)'),
            ('telecom_association_summary.xlsx', 'Service affinities'),
        ]:
            filepath = os.path.join(latest_report_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    st.download_button(f"⬇️ {label}", f.read(), filename)
        st.caption(f"From: {os.path.basename(latest_report_dir)}")
    else:
        st.caption("No reports yet. Run `python src/generate_reports.py` to create them.")


# ------------------------------------------------------------------- upload
uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

use_sample = False
if uploaded is None and os.path.exists(SAMPLE_FILE):
    use_sample = st.checkbox("No file? Try data/telco_new_cust.csv instead")

if uploaded is None and not use_sample:
    st.info("Waiting for a CSV upload, or check the sample-data box above.")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(SAMPLE_FILE)

st.subheader("Preview of input data")
st.dataframe(df_raw.head(), use_container_width=True)

# ------------------------------------------------------------------- score
try:
    with st.spinner("Scoring customers..."):
        data_encd, telco_final = pre.transform(df_raw)   # engineered features (for charts)
        labels = km.predict(telco_final)

        data_encd['segment'] = pd.Series(labels, index=data_encd.index).map(segment_names)

        df_out = df_raw.copy()                            # original columns (for the table/download)
        df_out['cluster'] = labels
        df_out['segment'] = df_out['cluster'].map(segment_names)
except ValueError as e:
    st.error(f"Could not process this file: {e}")
    st.stop()


# --------------------------------------------------------------------- KPIs
st.subheader("Key numbers")

total_customers = len(df_out)
avg_income = df_raw['income'].mean()
avg_tenure = df_raw['tenure'].mean()
avg_monthly_spend = data_encd['total_mon_usage'].mean()   # sum of the 5 monthly usage columns
avg_services = data_encd['total_services'].mean()          # count of services adopted
pct_high_value = df_out['segment'].isin(HIGH_VALUE_SEGMENTS).mean() * 100

kpi_cols = st.columns(6)
kpi_cols[0].metric("Total Customers", f"{total_customers:,}")
kpi_cols[1].metric("Avg Monthly Usage", f"{avg_monthly_spend:,.0f}")
kpi_cols[2].metric("Avg Income", f"${avg_income:,.0f}")
kpi_cols[3].metric("Avg Tenure", f"{avg_tenure:,.0f} mo")
kpi_cols[4].metric("Avg Services Used", f"{avg_services:,.1f}")
kpi_cols[5].metric("High-Value Customers", f"{pct_high_value:,.0f}%")


# ---------------------------------------------------------- segment charts

st.markdown("**Service Adoption Rate by Segment**")
st.caption("Share of customers in each segment who use each service (0 = none, 1 = all).")
adoption = data_encd.groupby('segment')[pre.SERVICE_COLS].mean()

fig, ax = plt.subplots(figsize=(11, 0.5 * len(adoption) + 1.5))
im = ax.imshow(adoption.values, cmap='Greens', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(pre.SERVICE_COLS)))
ax.set_xticklabels(pre.SERVICE_COLS, rotation=45, ha='right')
ax.set_yticks(range(len(adoption.index)))
ax.set_yticklabels(adoption.index)
for i in range(adoption.shape[0]):
    for j in range(adoption.shape[1]):
        ax.text(j, i, f"{adoption.values[i, j]:.2f}", ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=ax, label='Adoption rate')
fig.tight_layout()
st.pyplot(fig)

st.subheader("Segment breakdown")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Customers per Segment**")
    counts = df_out['segment'].value_counts()
    st.bar_chart(counts)

with col2:
    st.markdown("**Avg Monthly Revenue per Segment ($)**")
    revenue_by_segment = data_encd.groupby('segment')['total_mon_usage'].mean().round(0)
    st.bar_chart(revenue_by_segment)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Avg Services Used per Segment**")
    usage_by_segment = data_encd.groupby('segment')['total_services'].mean().round(1)
    st.bar_chart(usage_by_segment)

with col4:
    st.markdown("**Avg Tenure per Segment**")
    tenure_by_segment = data_encd.groupby('segment')['tenure'].mean().round(0)
    st.bar_chart(tenure_by_segment)

# --------------------------------------------------------------- full table
st.subheader("Segment assignments")
st.dataframe(df_out, use_container_width=True)

csv_bytes = df_out.to_csv(index=False).encode('utf-8')
st.download_button("Download results as CSV", csv_bytes, "segmented_customers.csv", "text/csv")
