"""  
-----------------------------------preprocessing.py-------------------------------------------------------- 

Reusable data preprocessing pipeline that can be: 
1.Fit on the original training data that we use inside train.py 
2.Reuse on brand new customer data that we use used inside predict.py / app.py 
"""

import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class TelcoPreprocessor:
    # ------------------- Column Groups -------------------
    VALUE_ADDED_COLS = ['voice', 'pager', 'internet', 'callid', 'callwait', 'forward', 'confer', 'ebill']
    CONNECTIVITY_COLS = ['tollfree', 'equip', 'callcard', 'wireless', 'multline']
    SERVICE_COLS = ['tollfree', 'equip', 'callcard', 'wireless', 'multline',
                    'voice', 'pager', 'internet', 'callid', 'callwait', 'forward', 'confer', 'ebill']
    USAGE_COLS = ['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']
    OHE_COLS = ['region', 'custcat']

    # Final features used in model
    CLUSTER_FEATURES = ['longmon', 'tollmon', 'wiremon','value_added_services', 'connectivity_services',
                        'usage_per_service', 'loyalty_score']

    def __init__(self):
        self.num_imputer = None
        self.clip_cutoffs = {}
        self.ohe = None
        self.scaler = None
        self.num_cols = None
        self.is_fitted = False

    # ---------------Clean column names (lowercase, remove spaces)---------------------
    def _clean_column_names(self, df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        return df

    # ----------------Feature Engineering (creating new columns)----------------------------
    def _engineer_features(self, df):

        df = df.copy()
        df['total_services'] = df[self.SERVICE_COLS].sum(axis=1)
        df['total_mon_usage'] = df[self.USAGE_COLS].sum(axis=1)
        df['avg_mon_usage'] = df[self.USAGE_COLS].mean(axis=1)

        df['value_added_services'] = df[self.VALUE_ADDED_COLS].sum(axis=1)
        df['connectivity_services'] = df[self.CONNECTIVITY_COLS].sum(axis=1)

        df['usage_per_service'] = df['total_mon_usage'] / (df['total_services'] + 1)

        df['loyalty_score'] = df['tenure'] * df['income'] / 1000

        return df

    # --------------Training---------------------------------------
    def fit_transform(self, df):

        # Clean and Save column names
        df = self._clean_column_names(df)  
        self.num_cols = df.columns.tolist() 

        # Fill missing values using median
        self.num_imputer = SimpleImputer(strategy='median')
        self.num_imputer.fit(df[self.num_cols])
        data_imp = df.copy()
        data_imp[self.num_cols] = self.num_imputer.transform(df[self.num_cols])

        # Outlier handling (clip values)
        data_clipped = data_imp.copy()
        for col in self.num_cols:
            lower = data_imp[col].quantile(0.01)
            upper = data_imp[col].quantile(0.99)
            self.clip_cutoffs[col] = (lower, upper)
            data_clipped[col] = data_imp[col].clip(lower=lower, upper=upper)

        ## One Hot Encoding
        data_encd = data_clipped.copy()
        data_encd[self.OHE_COLS] = data_encd[self.OHE_COLS].astype(int)
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ohe.fit(data_encd[self.OHE_COLS])
        ohe_names = self.ohe.get_feature_names_out(self.OHE_COLS)
        ohe_data = pd.DataFrame(self.ohe.transform(data_encd[self.OHE_COLS]),columns=ohe_names,index=data_encd.index)
        data_encd = pd.concat([data_encd.drop(columns=self.OHE_COLS), ohe_data], axis=1)

        # Feature Engineering
        data_encd = self._engineer_features(data_encd)

        # Scaling
        cols_to_scale = [c for c in data_encd.columns if data_encd[c].nunique() > 2]
        self.scaler = StandardScaler()
        self.scaler.fit(data_encd[cols_to_scale])
        data_scaled = data_encd.copy()
        data_scaled[cols_to_scale] = self.scaler.transform(data_encd[cols_to_scale])

        # Final features for model
        telco_final = data_scaled[self.CLUSTER_FEATURES]
        self.is_fitted = True

        return data_encd, telco_final

    # ----------------Transform-------------------------------------
    def transform(self, df):

        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet!")

        df = self._clean_column_names(df)

        # Check missing columns
        missing = set(self.num_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Apply same steps as training
        data_imp = df.copy()
        data_imp[self.num_cols] = self.num_imputer.transform(df[self.num_cols])

        data_clipped = data_imp.copy()
        for col, (lower, upper) in self.clip_cutoffs.items():
            data_clipped[col] = data_imp[col].clip(lower=lower, upper=upper)

        data_encd = data_clipped.copy()
        data_encd[self.OHE_COLS] = data_encd[self.OHE_COLS].astype(int)
        ohe_names = self.ohe.get_feature_names_out(self.OHE_COLS)
        ohe_data = pd.DataFrame(self.ohe.transform(data_encd[self.OHE_COLS]),columns=ohe_names,index=data_encd.index)
        data_encd = pd.concat([data_encd.drop(columns=self.OHE_COLS), ohe_data], axis=1)
        data_encd = self._engineer_features(data_encd)

        cols_to_scale = list(self.scaler.feature_names_in_)
        data_scaled = data_encd.copy()
        data_scaled[cols_to_scale] = self.scaler.transform(data_encd[cols_to_scale])

        telco_final = data_scaled[self.CLUSTER_FEATURES]

        return data_encd, telco_final

    # ------------Save preprocessor-----------------------------------------

    def save(self, folder='artifacts'):

        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, 'preprocessor.pkl')
        joblib.dump(self, file_path)

    # ------------ Load preprocessor-----------------------------------------
    @staticmethod    # use when we didn't want to use self, but still want to call the method on the class itself
    def load(folder='artifacts'):

        file_path = os.path.join(folder, 'preprocessor.pkl')

        return joblib.load(file_path)