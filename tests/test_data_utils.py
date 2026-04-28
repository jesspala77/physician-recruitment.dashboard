import pandas as pd
from pathlib import Path

from utils.data import get_feature_columns, get_target_column, load_data, preprocess_for_model


def test_load_data(tmp_path: Path):
    data = pd.DataFrame({
        'physician_specialty': ['ENT', 'Allergy'],
        'study_id': ['A', 'B'],
        'patient_volume': [100, 200],
        'eligible_patients': [10, 20],
        'research_interest': [5, 7],
        'distance_to_site': [10, 20],
        'active_trials': [2, 3],
        'coordinator_load': [5, 6],
        'screen_failure_rate': [0.1, 0.2],
        'historical_enrollment': [0.4, 0.5],
        'site_experience': [3, 4],
        'visit_burden': [8, 9],
        'eligibility_strictness': [2, 3],
        'specialty_match': [1.0, 0.5],
        'geographic_score': [0.7, 0.8],
        'site_burden': [0.2, 0.3],
        'capacity_score': [0.5, 0.6],
        'patient_fit': [0.6, 0.7],
        'predicted_match_label': ['Strong Match', 'Weak Match'],
    })
    file_path = tmp_path / 'sample.csv'
    data.to_csv(file_path, index=False)
    loaded = load_data(str(file_path))
    assert len(loaded) == 2
    assert 'physician_specialty' in loaded.columns


def test_preprocess_for_model_encodes_supported_categories():
    df = pd.DataFrame({
        'physician_specialty': ['ENT', 'Allergy', 'ENT'],
        'study_id': ['S1', 'S2', 'S1'],
    })
    processed, encoders = preprocess_for_model(df)
    assert 'physician_specialty' in processed.columns
    assert processed['physician_specialty'].dtype.kind in 'iu'
    assert 'physician_specialty' in encoders
    assert 'study_id' not in encoders
    assert processed['study_id'].tolist() == ['S1', 'S2', 'S1']


def test_get_feature_columns_returns_list():
    features = get_feature_columns()
    assert isinstance(features, list)
    assert 'physician_specialty' in features
    assert 'capacity_score' in features


def test_get_target_column_prefers_ground_truth_label():
    assert get_target_column(pd.DataFrame({'match_label': [1, 0]})) == 'match_label'
    assert get_target_column(pd.DataFrame({'predicted_match_label': ['Strong Match']})) == 'predicted_match_label'
