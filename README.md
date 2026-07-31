# Telecom Customer Segmentation

Turns the exploratory Jupyter notebook into a small, reusable deployment
project: a shared preprocessing pipeline, a train script, a predict script,
an optional reporting script, and a Streamlit app.

## Folder structure

```
telecom_segmentation/
├── README.md
├── requirements.txt
├── business_problem/
│   └── business_problem_and_notebook_snapshots.pdf   # original problem statement + notebook
├── data/
│   ├── telco.csv                # historical/training data (you provide)
│   └── telco_new_cust.csv       # new customers to score  (you provide)
├── src/
│   ├── preprocessing.py         # TelcoPreprocessor: clean, impute, clip, encode, engineer, scale
│   ├── train.py                 # fits everything on data/telco.csv -> saves models/
│   ├── predict.py               # scores data/telco_new_cust.csv -> saves outputs/
│   └── generate_reports.py      # optional: profiling.xlsx + association_rules_summary.xlsx
├── app/
│   └── app.py                   # Streamlit UI: upload a CSV, get segments
├── models/                      # created by train.py
│   ├── preprocessor.pkl         # fitted imputer + clip bounds + one-hot encoder + scaler
│   ├── kmeans_model.pkl         # trained clustering model
│   └── segment_names.pkl        # cluster id -> business segment name
└── outputs/                     # created by predict.py / generate_reports.py
    ├── segmented_customers.csv        # NEW customers (from predict.py) with their assigned segment
    ├── profiling.xlsx                 # full profiling table for k = 3, 4, 5, 6, 7 side by side
    ├── telecom_association_summary.xlsx  # per service: strongest affinities, best lift, confidence
    └── segments.xlsx                  # ALL of data/telco.csv, original columns, + cluster + segment
```

## What the original notebook did (recap)

1. **Business problem:** a telecom company wanted to understand customer
   behaviour to improve marketing, retention, and cross-selling (see
   `business_problem/`).
2. **Data prep:** cleaned column names, median-imputed missing values,
   clipped outliers at the 1st/99th percentile, one-hot encoded the two
   nominal-but-numeric columns (`region`, `custcat`).
3. **Feature engineering:** built 7 behaviour-level features — `longmon`,
   `tollmon`, `wiremon`, `value_added_services`, `connectivity_services`,
   `usage_per_service`, `loyalty_score` — chosen after checking correlation
   and VIF to avoid multicollinearity.
4. **Model selection:** compared PCA vs. raw features, and KMeans vs.
   Agglomerative Clustering vs. DBSCAN. KMeans (k=6, elbow method +
   silhouette score) gave the most balanced, interpretable segments.
5. **Profiling:** described and named each of the 6 clusters (Basic Users,
   Wireless & Tech Enthusiasts, Long-Tenure Traditional Customers,
   Value-Added Service Adopters, VIP Customers, Heavy Users).
6. **Market basket analysis:** Apriori + association rules (`mlxtend`) to
   find which services get adopted together, for bundling decisions.
7. **Business output:** a segment → marketing strategy table.

## How each file maps back to the notebook

| File                      | Replaces notebook cells | Purpose |
|---------------------------|--------------------------|---------|
| `src/preprocessing.py`    | [7]–[24]                 | Reusable `fit_transform` (train) / `transform` (new data) pipeline |
| `src/train.py`            | [30]–[37], [65]          | Fits the pipeline + KMeans(k=6), saves `models/` |
| `src/predict.py`          | new                       | Scores any new CSV using the saved `models/` |
| `src/generate_reports.py` | [50]–[63]                 | Optional: profiling table (k=3..7) + service-affinity business summary + segmented customer list, saved as Excel to `outputs/` |
| `app/app.py`              | new                       | Streamlit UI wrapping `predict.py` |

Why split it this way: a notebook can't be safely re-run on next month's
data, because re-fitting the imputer/scaler/clip-bounds on new data (instead
of reusing what was learned on training data) causes data leakage and
inconsistent results. `preprocessing.py` fixes that — it's fit **once** in
`train.py` and only ever *reused* in `predict.py` / `app.py`.

## How to run it

```bash
pip install -r requirements.txt

# put your real data here first:
#   data/telco.csv
#   data/telco_new_cust.csv

# 1) Train once on historical data
python src/train.py
# -> models/preprocessor.pkl, kmeans_model.pkl, segment_names.pkl

# 2) Score new customers any time
python src/predict.py
# -> outputs/segmented_customers.csv

# 3) (optional) regenerate the business reports
python src/generate_reports.py
# -> outputs/profiling.xlsx (k=3..7 side by side)
# -> outputs/telecom_association_summary.xlsx (service affinities, lift, confidence)
# -> outputs/segments.xlsx (original telco.csv columns + cluster + segment)

# 4) Or launch the interactive app
streamlit run app/app.py
```

All scripts default to the paths above but accept `--data`, `--models_dir`,
`--out` if you want to point them elsewhere.

## Notes for the resume / interview

- **Train/predict split**: fit once, score repeatedly — the core pattern of
  any real ML system, and the difference between a notebook and a
  deployable pipeline.
- **No data leakage**: clip bounds, the imputer, the encoder and the scaler
  are fit only on `data/telco.csv` and reused as-is on new data.
- **Retraining**: just re-run `src/train.py` on the latest data; it
  overwrites `models/` with fresh artifacts.
- **Productionizing further**: swap the local `models/` folder for a model
  registry, add monitoring for cluster drift over time, and wrap
  `src/predict.py` as a FastAPI/Flask endpoint alongside (or instead of) the
  Streamlit app.
