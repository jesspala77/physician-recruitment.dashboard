"""
Recruitment Dashboard - Streamlit Application
Multi-page app for exploring recruitment data and predicting physician-site matches.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data import load_data, get_feature_columns
from models.predict import load_model
from models.train import train_model
from utils.clinicaltrials_api import get_api_version, locations_to_dataframe, search_studies, studies_to_dataframe

# Constants
MATCH_COLOR_MAP = {'Strong Match': '#2ecc71', 'Weak Match': '#e74c3c'}
FEATURE_IMPORTANCE_LIMIT = 10
DATA_FILE = project_root / 'crssnp_recruitment_data.csv'
MODEL_FILE = project_root / 'models' / 'model.pkl'


def format_percentage(value: float) -> str:
    return f"{value*100:.1f}%"


def build_scatter_by_match(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str = None,
    size_column: str | None = None,
    size_scale: float = 10.0
) -> go.Figure:
    fig = go.Figure()
    for label, color in MATCH_COLOR_MAP.items():
        subset = df[df['predicted_match_label'] == label]
        if subset.empty:
            continue
        marker = dict(color=color, opacity=0.6)
        if size_column is not None and size_column in subset.columns:
            marker['size'] = subset[size_column].fillna(1).clip(lower=1).tolist()
            marker['sizemode'] = 'area'
            marker['sizeref'] = max(subset[size_column].max(), 1) / (size_scale ** 2)
        else:
            marker['size'] = 8

        fig.add_trace(
            go.Scatter(
                x=subset[x],
                y=subset[y],
                mode='markers',
                marker=marker,
                name=label,
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Match: {label}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title or f"{y_label} by {x_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title_text="Match Label",
        height=400,
    )
    return fig


def build_box_by_match(df: pd.DataFrame, x: str, y: str, x_label: str, y_label: str, title: str = None) -> go.Figure:
    fig = go.Figure()
    for label, color in MATCH_COLOR_MAP.items():
        subset = df[df['predicted_match_label'] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                x=subset[x],
                y=subset[y],
                name=label,
                marker_color=color,
                boxmean='sd',
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Match: {label}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title or f"{y_label} by {x_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title_text="Match Label",
        height=400,
        boxmode='group'
    )
    return fig


def build_histogram_by_match(df: pd.DataFrame, x: str, x_label: str, title: str = None) -> go.Figure:
    fig = go.Figure()
    for label, color in MATCH_COLOR_MAP.items():
        subset = df[df['predicted_match_label'] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=subset[x],
                name=label,
                marker_color=color,
                opacity=0.65,
                autobinx=True,
            )
        )

    fig.update_layout(
        title=title or f"Distribution of {x_label}",
        xaxis_title=x_label,
        yaxis_title="Count",
        legend_title_text="Match Label",
        height=400,
        barmode='overlay'
    )
    return fig


def build_pie_by_counts(values, names, title: str = None) -> go.Figure:
    colors = [MATCH_COLOR_MAP.get(name, '#636EFA') for name in names]
    fig = go.Figure(
        go.Pie(
            labels=names,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            sort=False,
        )
    )
    fig.update_layout(
        title=title or "",
        height=400,
        legend_title_text="Match Label"
    )
    return fig

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Recruitment Dashboard",
    page_icon="RD",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Physician-Site Recruitment Match Dashboard")

# ============================================================================
# CACHE AND DATA LOADING
# ============================================================================

@st.cache_data
def load_dashboard_data():
    """Load and cache the recruitment dataset."""
    df = load_data(str(DATA_FILE))
    return df

@st.cache_resource
def load_predictor_model():
    """Load and cache the trained model."""
    if not MODEL_FILE.exists():
        return None
    return load_model(str(MODEL_FILE))


def reload_predictor_model():
    """Clear cached model and reload the latest trained model."""
    load_predictor_model.clear()
    return load_predictor_model()


@st.cache_data(ttl=6 * 60 * 60)
def load_trial_api_metadata():
    """Load ClinicalTrials.gov API metadata with caching."""
    return get_api_version()


@st.cache_data(ttl=30 * 60)
def load_live_trials(condition: str, location: str, overall_status: str, phase: str, page_size: int):
    """Load live study search results from ClinicalTrials.gov."""
    return search_studies(
        condition=condition,
        location=location,
        overall_status=overall_status,
        phase=phase,
        page_size=page_size,
    )

# Try to load data and model
try:
    df = load_dashboard_data()
    predictor = load_predictor_model()
    model_available = predictor is not None
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    df = None
    model_available = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "Search", "Match Analysis", "Model Insights", "Train Model", "Predictions", "Live Trial Finder"],
    help="Select a page to navigate"
)

if df is not None:
    st.sidebar.divider()
    st.sidebar.markdown("**Dataset Summary**")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Physicians", df['physician_id'].nunique())
    st.sidebar.metric("Sites", df['site_id'].nunique())
    st.sidebar.metric("Specialties", df['physician_specialty'].nunique())

    st.sidebar.divider()
    st.sidebar.markdown("**Model Status**")
    st.sidebar.metric("Model Loaded", "Yes" if model_available else "No")
    if model_available:
        st.sidebar.info("Model available for predictions and insights.")
    else:
        st.sidebar.info("Train the model with `python models/train.py` to enable prediction features.")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    if df is None:
        st.error("Data could not be loaded")
    else:
        st.subheader("Dashboard Overview")

        strong_predicted = (df['predicted_match_label'] == 'Strong Match').sum()
        avg_match_score = df['match_score'].mean()
        avg_enrollment = df['historical_enrollment'].mean()
        agreement_rate = None

        if 'match_label' in df.columns:
            actual_labels = df['match_label'].map({1: 'Strong Match', 0: 'Weak Match'})
            agreement_rate = (actual_labels == df['predicted_match_label']).mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Predicted Strong Matches", strong_predicted, format_percentage(strong_predicted / len(df)))
        with col3:
            st.metric("Avg Match Score", f"{avg_match_score:.3f}")
        with col4:
            st.metric("Avg Historical Enrollment", f"{avg_enrollment:.2f}")

        if agreement_rate is not None:
            st.metric("Prediction Agreement", format_percentage(agreement_rate))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Match Distribution")
            match_counts = df['predicted_match_label'].value_counts()
            fig_match = build_pie_by_counts(
                values=match_counts.values,
                names=match_counts.index,
                title='Predicted Match Distribution'
            )
            st.plotly_chart(fig_match, use_container_width=True)
        with col2:
            if 'match_label' in df.columns:
                st.subheader("Actual Match Distribution")
                actual_counts = df['match_label'].map({1: 'Strong Match', 0: 'Weak Match'}).value_counts()
                fig_actual = build_pie_by_counts(
                    values=actual_counts.values,
                    names=actual_counts.index,
                    title='Actual Match Distribution'
                )
                st.plotly_chart(fig_actual, use_container_width=True)
            else:
                st.subheader("Actual Match Distribution")
                st.info("Actual match labels are not available in this dataset.")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matches by Specialty")
            specialty_matches = df.groupby('physician_specialty')['predicted_match_label'].apply(
                lambda x: (x == 'Strong Match').sum()
            ).sort_values(ascending=False)
            fig_specialty = px.bar(
                x=specialty_matches.index,
                y=specialty_matches.values,
                labels={'x': 'Specialty', 'y': 'Strong Matches'},
                color_discrete_sequence=['#3498db']
            )
            fig_specialty.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_specialty, use_container_width=True)
        with col2:
            st.subheader("Capacity vs Enrollment")
            fig_capacity = build_scatter_by_match(
                df,
                x='capacity_score',
                y='historical_enrollment',
                x_label='Capacity Score',
                y_label='Historical Enrollment',
                title='Capacity vs Enrollment'
            )
            st.plotly_chart(fig_capacity, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distance to Site vs Match Score")
            fig_distance = build_scatter_by_match(
                df,
                x='distance_to_site',
                y='match_score',
                x_label='Distance (miles)',
                y_label='Match Score',
                title='Distance to Site vs Match Score'
            )
            st.plotly_chart(fig_distance, use_container_width=True)
        with col2:
            st.subheader("Enrollment vs Match Score")
            fig_enrollment = build_scatter_by_match(
                df,
                x='historical_enrollment',
                y='match_score',
                x_label='Historical Enrollment',
                y_label='Match Score',
                title='Enrollment vs Match Score'
            )
            st.plotly_chart(fig_enrollment, use_container_width=True)

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================

elif page == "Data Explorer":
    if df is None:
        st.error("Data could not be loaded")
    else:
        st.subheader("Interactive Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_specialty = st.multiselect(
                "Specialty",
                sorted(df['physician_specialty'].unique()),
                default=sorted(df['physician_specialty'].unique()),
                help="Filter by physician specialty"
            )
        
        with col2:
            selected_match = st.multiselect(
                "Match Label",
                sorted(df['predicted_match_label'].unique()),
                default=sorted(df['predicted_match_label'].unique()),
                help="Filter by match prediction"
            )
        
        with col3:
            study_id = st.selectbox(
                "Study ID",
                ['All'] + sorted(df['study_id'].unique()),
                help="Filter by study ID"
            )
        
        # Apply filters
        filtered_df = df[
            (df['physician_specialty'].isin(selected_specialty)) &
            (df['predicted_match_label'].isin(selected_match))
        ]
        
        if study_id != 'All':
            filtered_df = filtered_df[filtered_df['study_id'] == study_id]
        
        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_column = st.selectbox(
                "Sort by",
                ['match_score', 'distance_to_site', 'capacity_score', 'historical_enrollment'],
                help="Select column to sort by"
            )
        
        with col2:
            sort_ascending = st.checkbox("Ascending Order", value=False)
        
        filtered_df = filtered_df.sort_values(sort_column, ascending=sort_ascending)
        
        # Display table
        st.write(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        # Select columns to display
        display_cols = [
            'physician_id', 'site_id', 'physician_specialty', 'study_id',
            'match_score', 'predicted_match_label', 'match_label',
            'distance_to_site', 'capacity_score', 'historical_enrollment'
        ]
        
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=500
        )
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="recruitment_data_filtered.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: SEARCH
# ============================================================================

elif page == "Search":
    if df is None:
        st.error("Data could not be loaded")
    else:
        st.subheader("Search Physician or Site")
        search_type = st.radio("Search by:", ["Physician ID", "Site ID"], horizontal=True)
        search_query = st.text_input("Enter full or partial ID to search", value="")

        if not search_query:
            st.info("Type a physician or site ID fragment and press Enter to see results.")
        else:
            if search_type == "Physician ID":
                results = df[df['physician_id'].astype(str).str.contains(search_query, case=False, na=False)]
            else:
                results = df[df['site_id'].astype(str).str.contains(search_query, case=False, na=False)]

            if len(results) > 0:
                st.success(f"Found {len(results)} records matching '{search_query}'")
                st.divider()

                if search_type == "Physician ID":
                    display_cols = [
                        'site_id', 'study_id', 'match_score', 'predicted_match_label',
                        'distance_to_site', 'capacity_score', 'historical_enrollment'
                    ]
                    st.write("**Site Matches:**")
                    st.dataframe(results.sort_values('match_score', ascending=False)[display_cols], use_container_width=True)
                else:
                    display_cols = [
                        'physician_id', 'physician_specialty', 'match_score', 'predicted_match_label',
                        'patient_volume', 'research_interest', 'distance_to_site'
                    ]
                    st.write("**Physician Matches:**")
                    st.dataframe(results.sort_values('match_score', ascending=False)[display_cols], use_container_width=True)

                first_result = results.iloc[0]
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    if search_type == "Physician ID":
                        st.metric("Specialty", first_result['physician_specialty'])
                    else:
                        st.metric("Study ID", first_result['study_id'])
                with col2:
                    st.metric("Match Score", f"{first_result['match_score']:.3f}")
                with col3:
                    st.metric("Predicted Label", first_result['predicted_match_label'])
            else:
                st.warning(f"No records found for '{search_query}'")

# ============================================================================
# PAGE: MATCH ANALYSIS
# ============================================================================

elif page == "Match Analysis":
    if df is None:
        st.error("Data could not be loaded")
    else:
        st.subheader("Detailed Match Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Match Score Distribution by Specialty**")
            fig_box = build_box_by_match(
                df,
                x='physician_specialty',
                y='match_score',
                x_label='Specialty',
                y_label='Match Score',
                title='Match Score Distribution by Specialty'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.write("**Screen Failure Rate vs Match Score**")
            fig_scatter = build_scatter_by_match(
                df,
                x='screen_failure_rate',
                y='match_score',
                x_label='Screen Failure Rate',
                y_label='Match Score',
                title='Screen Failure Rate vs Match Score'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Coordinator Load Distribution**")
            fig_hist = build_histogram_by_match(
                df,
                x='coordinator_load',
                x_label='Coordinator Load',
                title='Coordinator Load Distribution'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.write("**Eligibility Strictness vs Match Score**")
            fig_scatter2 = build_scatter_by_match(
                df,
                x='eligibility_strictness',
                y='match_score',
                x_label='Eligibility Strictness',
                y_label='Match Score',
                title='Eligibility Strictness vs Match Score',
                size_column='patient_volume',
                size_scale=15.0
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
        
        # Statistics by specialty
        st.divider()
        st.write("**Specialty Statistics**")
        
        specialty_stats = df.groupby('physician_specialty').agg({
            'match_score': ['mean', 'std', 'min', 'max', 'count'],
            'predicted_match_label': lambda x: (x == 'Strong Match').sum()
        }).round(3)
        
        specialty_stats.columns = ['Avg Score', 'Std Dev', 'Min Score', 'Max Score', 'Total', 'Strong Matches']
        st.dataframe(specialty_stats, use_container_width=True)

# ============================================================================
# PAGE: MODEL INSIGHTS
# ============================================================================

elif page == "Model Insights":
    if not model_available:
        st.error("Model is not available. Please train the model first by running:\n\n`python models/train.py`")
    else:
        st.subheader("Model Insights")
        st.info("Review trained model importance and how predictions align with dataset labels.")

        feature_importance = pd.DataFrame({
            'Feature': get_feature_columns(),
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(FEATURE_IMPORTANCE_LIMIT)

        st.write("**Top Feature Importances**")
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color_discrete_sequence=['#3498db']
        )
        fig_importance.update_layout(height=420, showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)

        if 'match_label' in df.columns:
            predictions = predictor.batch_predict(df)
            actual_labels = df['match_label'].map({1: 'Strong Match', 0: 'Weak Match'})
            predicted_labels = predictions['predicted_label']
            agreement_rate = (actual_labels == predicted_labels).mean()

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Strong Matches", int((actual_labels == 'Strong Match').sum()))
            with col2:
                st.metric("Predicted Strong Matches", int((predicted_labels == 'Strong Match').sum()))
            with col3:
                st.metric("Accuracy vs Actual", format_percentage(agreement_rate))

            st.write("**Confusion Matrix (Actual vs Predicted)**")
            confusion_matrix = pd.crosstab(actual_labels, predicted_labels, normalize='index')
            st.dataframe(confusion_matrix.round(3), use_container_width=True)
        else:
            st.warning("Actual match labels are not available in this dataset, so agreement metrics cannot be computed.")

# ============================================================================
# PAGE: TRAIN MODEL
# ============================================================================

elif page == "Train Model":
    st.subheader("Train Model")
    st.write("Retrain the physician-site match model using the current dataset.")
    st.info("Training will overwrite the existing saved model and refresh prediction capabilities.")

    if st.button("Train Model", type="primary"):
        try:
            with st.spinner("Training model, please wait..."):
                result = train_model(str(DATA_FILE), str(MODEL_FILE), verbose=False)
                predictor = reload_predictor_model()
                model_available = predictor is not None

            st.success("Model training completed successfully.")
            st.metric("Accuracy", f"{result['accuracy']:.3f}")
            st.metric("Precision", f"{result['precision']:.3f}")
            st.metric("Recall", f"{result['recall']:.3f}")
            st.metric("F1 Score", f"{result['f1']:.3f}")

            st.divider()
            st.write("**Top Feature Importances**")
            feature_importance = result['feature_importance'].head(FEATURE_IMPORTANCE_LIMIT)
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                color_discrete_sequence=['#3498db']
            )
            fig_importance.update_layout(height=420, showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

        except Exception as e:
            st.error(f"Model training failed: {e}")

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

elif page == "Predictions":
    if not model_available:
        st.error("Model is not available. Please train the model first by running:\n\n`python models/train.py`")
    else:
        st.subheader("Predict Physician-Site Match")
        st.info("Enter physician and site characteristics to predict match probability")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            specialty = st.selectbox("Physician Specialty", df['physician_specialty'].unique())
            patient_volume = st.number_input("Patient Volume", min_value=0, max_value=2000, value=500)
            eligible_patients = st.number_input("Eligible Patients", min_value=0, max_value=500, value=100)
            research_interest = st.slider("Research Interest (1-10)", 1, 10, 5)
            distance_to_site = st.slider("Distance to Site (miles)", 0, 50, 25)
        
        with col2:
            active_trials = st.slider("Active Trials", 0, 10, 5)
            coordinator_load = st.slider("Coordinator Load", 0, 50, 20)
            screen_failure_rate = st.slider("Screen Failure Rate (0-1)", 0.0, 1.0, 0.3)
            historical_enrollment = st.slider("Historical Enrollment (0-1)", 0.0, 1.0, 0.5)
            site_experience = st.slider("Site Experience (years)", 0, 10, 5)
        
        with col3:
            visit_burden = st.slider("Visit Burden", 0, 20, 10)
            eligibility_strictness = st.slider("Eligibility Strictness", 0, 10, 5)
            specialty_match = st.slider("Specialty Match (0-1)", 0.0, 1.0, 0.7)
            geographic_score = st.slider("Geographic Score (0-1)", 0.0, 1.0, 0.5)
            site_burden = st.slider("Site Burden (0-1)", 0.0, 1.0, 0.4)
            capacity_score = st.slider("Capacity Score (0-1)", 0.0, 1.0, 0.6)
            patient_fit = st.slider("Patient Fit (0-1)", 0.0, 1.0, 0.5)
        
        # Make prediction
        if st.button("Predict Match", use_container_width=True):
            input_data = {
                'physician_specialty': specialty,
                'patient_volume': patient_volume,
                'eligible_patients': eligible_patients,
                'research_interest': research_interest,
                'distance_to_site': distance_to_site,
                'active_trials': active_trials,
                'coordinator_load': coordinator_load,
                'screen_failure_rate': screen_failure_rate,
                'historical_enrollment': historical_enrollment,
                'site_experience': site_experience,
                'visit_burden': visit_burden,
                'eligibility_strictness': eligibility_strictness,
                'specialty_match': specialty_match,
                'geographic_score': geographic_score,
                'site_burden': site_burden,
                'capacity_score': capacity_score,
                'patient_fit': patient_fit
            }
            
            try:
                prediction_label, confidence = predictor.predict(input_data)
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction_label == 'Strong Match':
                        st.success(f"### Prediction: {prediction_label}")
                    else:
                        st.warning(f"### Prediction: {prediction_label}")
                
                with col2:
                    st.info(f"### Confidence: {confidence*100:.1f}%")
                
                # Show confidence breakdown
                st.divider()
                st.write("**Prediction Confidence Breakdown:**")
                
                feature_importance = pd.DataFrame({
                    'Feature': get_feature_columns(),
                    'Importance': predictor.model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color_discrete_sequence=['#3498db']
                )
                fig_importance.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")


# ============================================================================
# PAGE: LIVE TRIAL FINDER
# ============================================================================

elif page == "Live Trial Finder":
    st.subheader("Live Trial Finder")
    st.info("Search current ClinicalTrials.gov studies to identify active and recruiting trials by condition, location, and phase.")

    try:
        metadata = load_trial_api_metadata()
        st.caption(
            f"ClinicalTrials.gov API v{metadata['api_version']} | Data timestamp: {metadata['data_timestamp']}"
        )
    except Exception as e:
        st.warning(f"Could not load API metadata: {e}")

    with st.form("live_trial_search"):
        col1, col2 = st.columns(2)
        with col1:
            live_condition = st.text_input("Condition or disease", value="chronic rhinosinusitis")
            live_location = st.text_input("Location", value="United States")
        with col2:
            live_status = st.selectbox(
                "Overall status",
                ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "All"],
                index=0,
            )
            live_phase = st.selectbox(
                "Phase",
                ["All", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
                index=0,
            )

        live_page_size = st.slider("Max studies", min_value=5, max_value=50, value=15, step=5)
        live_submit = st.form_submit_button("Search live studies", type="primary")

    if live_submit:
        try:
            with st.spinner("Loading current studies from ClinicalTrials.gov..."):
                live_studies = load_live_trials(
                    live_condition,
                    live_location,
                    live_status,
                    live_phase,
                    live_page_size,
                )

            if not live_studies:
                st.warning("No studies matched the current search filters.")
            else:
                studies_df = studies_to_dataframe(live_studies)
                locations_df = locations_to_dataframe(live_studies)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Studies returned", len(studies_df))
                with col2:
                    st.metric("Total study sites", int(studies_df['locations'].sum()))
                with col3:
                    st.metric("Recruiting sites", int(studies_df['recruiting_locations'].sum()))

                st.divider()
                st.write("**Study summary**")
                st.dataframe(
                    studies_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'study_url': st.column_config.LinkColumn('Study record'),
                        'has_results': st.column_config.CheckboxColumn('Results posted'),
                    },
                )

                st.divider()
                st.write("**Recruiting site details**")
                recruiting_locations_df = locations_df[locations_df['status'] == 'RECRUITING'].copy()
                if recruiting_locations_df.empty:
                    st.info("No recruiting locations were listed in the returned studies.")
                else:
                    st.dataframe(
                        recruiting_locations_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as e:
            st.error(f"Live trial search failed: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Recruitment Dashboard | Built with Streamlit and scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)


