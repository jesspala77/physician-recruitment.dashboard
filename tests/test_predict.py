import pandas as pd
from pathlib import Path
from models.train import train_model
from models.predict import load_model
from utils.data import get_feature_columns


def create_sample_training_data(tmp_path: Path) -> Path:
    feature_columns = get_feature_columns()
    data = {
        'physician_specialty': ['ENT', 'Allergy', 'Cardiology', 'Oncology', 'ENT', 'Allergy', 'Cardiology', 'Oncology', 'ENT', 'Allergy'],
        'patient_volume': [100, 120, 90, 110, 130, 95, 105, 115, 125, 85],
        'eligible_patients': [10, 12, 8, 11, 14, 9, 13, 15, 11, 7],
        'research_interest': [5, 6, 7, 4, 6, 5, 8, 7, 6, 4],
        'distance_to_site': [10, 15, 8, 20, 12, 18, 9, 17, 14, 11],
        'active_trials': [2, 3, 1, 4, 3, 2, 4, 3, 2, 1],
        'coordinator_load': [5, 6, 4, 3, 5, 6, 4, 3, 6, 5],
        'screen_failure_rate': [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.11, 0.19, 0.14, 0.16],
        'historical_enrollment': [0.4, 0.45, 0.35, 0.5, 0.47, 0.38, 0.49, 0.44, 0.46, 0.36],
        'site_experience': [3, 4, 2, 5, 4, 3, 5, 4, 4, 2],
        'visit_burden': [8, 9, 7, 6, 8, 9, 7, 6, 8, 7],
        'eligibility_strictness': [2, 3, 4, 3, 2, 4, 3, 2, 3, 4],
        'specialty_match': [1.0, 0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.65, 0.9, 0.7],
        'geographic_score': [0.7, 0.6, 0.8, 0.5, 0.65, 0.55, 0.85, 0.45, 0.6, 0.5],
        'site_burden': [0.2, 0.3, 0.25, 0.4, 0.22, 0.35, 0.28, 0.38, 0.24, 0.34],
        'capacity_score': [0.5, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.35, 0.6, 0.4],
        'patient_fit': [0.6, 0.7, 0.5, 0.8, 0.7, 0.55, 0.75, 0.5, 0.68, 0.48],
        'match_score': [0.8, 0.85, 0.4, 0.45, 0.85, 0.5, 0.9, 0.55, 0.84, 0.42],
        'match_label': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        'predicted_match_label': ['Strong Match', 'Strong Match', 'Weak Match', 'Weak Match', 'Strong Match', 'Weak Match', 'Strong Match', 'Weak Match', 'Strong Match', 'Weak Match'],
        'study_id': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S1', 'S2']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / 'sample_training.csv'
    df.to_csv(file_path, index=False)
    return file_path


def test_predictor_predict_and_batch_predict(tmp_path: Path):
    data_path = create_sample_training_data(tmp_path)
    model_path = tmp_path / 'model.pkl'

    train_model(str(data_path), str(model_path), verbose=False)
    predictor = load_model(str(model_path))

    sample_input = {
        'physician_specialty': 'ENT',
        'patient_volume': 110,
        'eligible_patients': 11,
        'research_interest': 6,
        'distance_to_site': 12,
        'active_trials': 3,
        'coordinator_load': 5,
        'screen_failure_rate': 0.15,
        'historical_enrollment': 0.45,
        'site_experience': 4,
        'visit_burden': 8,
        'eligibility_strictness': 3,
        'specialty_match': 0.9,
        'geographic_score': 0.6,
        'site_burden': 0.25,
        'capacity_score': 0.55,
        'patient_fit': 0.65
    }

    label, confidence = predictor.predict(sample_input)
    assert label in {'Strong Match', 'Weak Match'}
    assert 0.0 <= confidence <= 1.0

    batch_df = pd.DataFrame([sample_input, {**sample_input, 'physician_specialty': 'Allergy', 'study_id': 'S2'}])
    batch_result = predictor.batch_predict(batch_df)
    assert 'predicted_label' in batch_result.columns
    assert 'confidence' in batch_result.columns
    assert len(batch_result) == 2
