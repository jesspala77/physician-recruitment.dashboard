"""Data loading and preprocessing utilities for recruitment data."""
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Load recruitment dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    return pd.read_csv(filepath)


def preprocess_for_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare data for ML model by encoding categorical features."""
    df_processed = df.copy()
    categorical_cols = ['physician_specialty']

    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            encoder = LabelEncoder()
            df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = encoder

    return df_processed, label_encoders


def get_feature_columns() -> list[str]:
    """Define feature columns for model training."""
    return [
        'physician_specialty',
        'patient_volume',
        'eligible_patients',
        'research_interest',
        'distance_to_site',
        'active_trials',
        'coordinator_load',
        'screen_failure_rate',
        'historical_enrollment',
        'site_experience',
        'visit_burden',
        'eligibility_strictness',
        'specialty_match',
        'geographic_score',
        'site_burden',
        'capacity_score',
        'patient_fit',
    ]


def get_target_column(df: pd.DataFrame | None = None) -> str:
    """Return the preferred classification target column for the dataset."""
    if df is not None and 'match_label' in df.columns:
        return 'match_label'
    return 'predicted_match_label'
