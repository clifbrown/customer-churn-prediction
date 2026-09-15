"""Validate raw customer profiles and learn preprocessing only inside each fit."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = 'Churn'
ID_COL = 'customerID'
NUMERIC_RAW = ['tenure', 'MonthlyCharges', 'TotalCharges']
ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
          'StreamingTV', 'StreamingMovies']
CATEGORIES = {
    'gender': ['Female', 'Male'], 'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'], 'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'], 'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{c: ['Yes', 'No', 'No internet service'] for c in ADDONS},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                      'Credit card (automatic)'],
}
CATEGORICAL_FEATURES = list(CATEGORIES)
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL_FEATURES
NUMERIC_FEATURES = NUMERIC_RAW + ['AvgMonthlySpend']


def validate_features(frame):
    """Return normalised raw features; reject malformed/inconsistent profiles.

    Missing numeric values are allowed and imputed using training-fold medians.
    Categorical blanks become missing and use training-fold modes. Unknown
    nonblank categories fail explicitly rather than silently becoming all-zero.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Provide a non-empty table of customer profiles.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not supported.')
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    x = frame[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        if x[col].dtype == object:
            x[col] = x[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        x[col] = x[col].mask(x[col].eq(''), np.nan)
    for col in NUMERIC_RAW + ['SeniorCitizen']:
        original = x[col]
        values = pd.to_numeric(original, errors='coerce')
        if (original.notna() & values.isna()).any():
            raise ValueError(f'{col} contains non-numeric values.')
        if np.isinf(values.to_numpy(dtype=float)).any() or (values.dropna() < 0).any():
            raise ValueError(f'{col} must contain finite, non-negative values.')
        x[col] = values
    if (x.tenure.dropna() % 1 != 0).any():
        raise ValueError('tenure must be a whole number of months.')
    for col, allowed in CATEGORIES.items():
        bad = x[col].notna() & ~x[col].isin(allowed)
        if bad.any():
            raise ValueError(f'Unsupported value in {col}: {x.loc[bad, col].iloc[0]}')
    phone_known = x.PhoneService.notna() & x.MultipleLines.notna()
    if (phone_known & (x.PhoneService.eq('No') != x.MultipleLines.eq('No phone service'))).any():
        raise ValueError('MultipleLines must agree with PhoneService.')
    for col in ADDONS:
        known = x.InternetService.notna() & x[col].notna()
        if (known & (x.InternetService.eq('No') != x[col].eq('No internet service'))).any():
            raise ValueError(f'{col} must agree with InternetService.')
    return x


def clean(frame):
    """Stateless cleaning and feature engineering, also used at prediction time."""
    x = validate_features(frame)
    # The supplied dataset has blank cumulative bills at tenure zero. Those
    # represent no completed billing history; other missing bills stay missing.
    x.loc[x.tenure.eq(0) & x.TotalCharges.isna(), 'TotalCharges'] = 0.0
    denom = x.tenure.where(x.tenure > 0)
    x['AvgMonthlySpend'] = (x.TotalCharges / denom).where(x.tenure != 0, x.MonthlyCharges)
    return x


def get_feature_target(frame):
    x = validate_features(frame)
    if TARGET not in frame:
        raise ValueError('Training data requires Churn.')
    y = frame[TARGET].map(lambda v: {'Yes': 1, 'No': 0}.get(v, v))
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Churn must be Yes/No or 0/1 with no missing labels.')
    return x, y.astype(int)


class CustomerCleaner(TransformerMixin, BaseEstimator):
    """A serialisable sklearn step ensuring raw-data train/serve consistency."""
    def fit(self, X, y=None):
        clean(X)
        self.n_features_in_ = len(RAW_FEATURES)
        self.feature_names_in_ = np.array(RAW_FEATURES, dtype=object)
        return self

    def transform(self, X):
        return clean(X)


def build_preprocessor(include_spend=True):
    numeric = NUMERIC_FEATURES if include_spend else NUMERIC_RAW
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scaler', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ])


def build_pipeline(model, include_spend=True):
    return Pipeline([('cleaner', CustomerCleaner()),
                     ('preprocessor', build_preprocessor(include_spend)), ('model', model)])
