"""Train ML model for physician-site match prediction."""
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_feature_columns, get_target_column, load_data


TARGET_LABEL_MAP = {
    'strong match': 1,
    'weak match': 0,
    '1': 1,
    '0': 0,
}


def normalize_target(series: pd.Series) -> pd.Series:
    """Normalize supported target formats to binary labels."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().astype(int).unique())
        if unique_values <= {0, 1}:
            return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower().map(TARGET_LABEL_MAP)
    if normalized.isna().any():
        invalid = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(f"Unsupported target values: {invalid}")
    return normalized.astype(int)


def train_model(data_filepath: str, model_save_path: str, verbose: bool = True):
    """Train a RandomForest model on recruitment data."""
    if verbose:
        print("Loading data...")
    df = load_data(data_filepath)

    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    feature_cols = get_feature_columns()
    target_col = get_target_column(df)

    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df_model = df[feature_cols + [target_col]].copy()

    label_encoders: dict[str, LabelEncoder] = {}
    if 'physician_specialty' in df_model.columns:
        encoder = LabelEncoder()
        df_model['physician_specialty'] = encoder.fit_transform(df_model['physician_specialty'].astype(str))
        label_encoders['physician_specialty'] = encoder

    if verbose:
        print(f"\nProcessed data shape: {df_model.shape}")
        print(f"Data types:\n{df_model.dtypes}")

    X = df_model[feature_cols]
    y_binary = normalize_target(df_model[target_col])

    if verbose:
        print(f"\nTraining target: {target_col}")
        print(f"Binary target distribution:\n{y_binary.value_counts()}")

    test_size = 0.2
    n_test = max(int(len(y_binary) * test_size), 1)

    stratify = y_binary
    if y_binary.nunique() <= 1 or y_binary.value_counts().min() < 2 or n_test < y_binary.nunique():
        stratify = None
    if verbose and stratify is None:
        print("\nWARNING: Not enough samples for stratified split; using random split instead.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    if verbose:
        print(f"\nTrain set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print("\nTraining RandomForest classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if verbose:
        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("=" * 50)

    feature_importance = pd.DataFrame(
        {
            'feature': feature_cols,
            'importance': model.feature_importances_,
        }
    ).sort_values('importance', ascending=False)

    if verbose:
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'label_encoders': label_encoders,
        'target_column': target_col,
    }
    joblib.dump(model_data, model_save_path)

    if verbose:
        print(f"\nModel saved to: {model_save_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_importance': feature_importance,
        'model_path': model_save_path,
        'target_column': target_col,
    }


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    data_path = os.path.join(repo_dir, 'crssnp_recruitment_data.csv')
    model_path = os.path.join(script_dir, 'model.pkl')

    train_model(data_path, model_path)
