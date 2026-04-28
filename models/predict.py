"""Load trained model and make predictions."""
import os
from typing import Any

import joblib
import pandas as pd


class MatchPredictor:
    """Wrapper for loading and using a trained match classifier."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_columns = self.model_data['feature_columns']
        self.label_encoders = self.model_data['label_encoders']
        self.target_column = self.model_data.get('target_column')

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, encoder in self.label_encoders.items():
            if col not in df.columns:
                continue

            values = df[col].astype(str)
            unknown_values = sorted(set(values) - set(encoder.classes_))
            if unknown_values:
                raise ValueError(
                    f"Unsupported value for {col}: {unknown_values[0]}. "
                    f"Expected one of: {', '.join(map(str, encoder.classes_))}"
                )
            df[col] = encoder.transform(values)
        return df

    def predict(self, input_data: dict[str, Any]) -> tuple[str, float]:
        """Make a prediction on a single observation."""
        df = self._encode_categorical(pd.DataFrame([input_data]))

        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")

        X = df[self.feature_columns]
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        label = 'Strong Match' if prediction == 1 else 'Weak Match'
        return label, confidence

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on a batch of observations."""
        df_pred = self._encode_categorical(df.copy())
        X = df_pred[self.feature_columns]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        df_pred['predicted_label'] = ['Strong Match' if p == 1 else 'Weak Match' for p in predictions]
        df_pred['confidence'] = probabilities.max(axis=1)
        return df_pred


def load_model(model_path: str) -> MatchPredictor:
    """Convenience function to load a trained predictor."""
    return MatchPredictor(model_path)
