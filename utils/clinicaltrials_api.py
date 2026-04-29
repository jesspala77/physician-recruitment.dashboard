"""ClinicalTrials.gov API helpers for live study discovery."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

BASE_URL = "https://clinicaltrials.gov/api/v2"
STUDY_URL_PREFIX = "https://clinicaltrials.gov/study/"


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def get_api_version() -> dict[str, str]:
    """Return ClinicalTrials.gov API and data freshness metadata."""
    payload = _fetch_json(f"{BASE_URL}/version")
    return {
        'api_version': payload.get('apiVersion', 'unknown'),
        'data_timestamp': payload.get('dataTimestamp', 'unknown'),
    }


def _build_studies_url(
    condition: str,
    location: str = "",
    overall_status: str = "",
    page_size: int = 20,
) -> str:
    params = {
        'format': 'json',
        'query.cond': condition.strip(),
        'pageSize': max(1, min(page_size, 100)),
    }
    if location.strip():
        params['query.locn'] = location.strip()
    if overall_status.strip() and overall_status != 'All':
        params['filter.overallStatus'] = overall_status.strip().upper()
    return f"{BASE_URL}/studies?{urlencode(params)}"


def _extract_phase(study: dict[str, Any]) -> str:
    phases = (
        study.get('protocolSection', {})
        .get('designModule', {})
        .get('phases', [])
    )
    return ', '.join(phases) if phases else 'Not specified'


def search_studies(
    condition: str,
    location: str = "",
    overall_status: str = "RECRUITING",
    phase: str = "All",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Search ClinicalTrials.gov studies using the v2 API."""
    if not condition or not condition.strip():
        raise ValueError('Condition is required for live study search.')

    payload = _fetch_json(_build_studies_url(condition, location, overall_status, page_size))
    studies = payload.get('studies', [])

    if phase and phase != 'All':
        phase_upper = phase.upper()
        studies = [study for study in studies if phase_upper in _extract_phase(study).upper()]

    return studies


def studies_to_dataframe(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten study responses into a dashboard-friendly table."""
    records: list[dict[str, Any]] = []

    for study in studies:
        protocol = study.get('protocolSection', {})
        ident = protocol.get('identificationModule', {})
        status = protocol.get('statusModule', {})
        design = protocol.get('designModule', {})
        conditions = protocol.get('conditionsModule', {})
        contacts = protocol.get('contactsLocationsModule', {})
        locations = contacts.get('locations', [])

        records.append({
            'nct_id': ident.get('nctId', ''),
            'title': ident.get('briefTitle', 'Untitled study'),
            'sponsor': ident.get('organization', {}).get('fullName', 'Unknown sponsor'),
            'status': status.get('overallStatus', 'Unknown'),
            'phase': _extract_phase(study),
            'study_type': design.get('studyType', 'Unknown'),
            'conditions': ', '.join(conditions.get('conditions', [])),
            'locations': len(locations),
            'recruiting_locations': sum(1 for location in locations if location.get('status') == 'RECRUITING'),
            'last_update': status.get('lastUpdatePostDateStruct', {}).get('date', 'Unknown'),
            'has_results': bool(study.get('hasResults', False)),
            'study_url': f"{STUDY_URL_PREFIX}{ident.get('nctId', '')}" if ident.get('nctId') else '',
        })

    return pd.DataFrame(records)


def locations_to_dataframe(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested study locations into site rows."""
    rows: list[dict[str, Any]] = []

    for study in studies:
        protocol = study.get('protocolSection', {})
        ident = protocol.get('identificationModule', {})
        contacts = protocol.get('contactsLocationsModule', {})

        for location in contacts.get('locations', []):
            rows.append({
                'nct_id': ident.get('nctId', ''),
                'title': ident.get('briefTitle', 'Untitled study'),
                'facility': location.get('facility', 'Unknown facility'),
                'status': location.get('status', 'Unknown'),
                'city': location.get('city', ''),
                'state': location.get('state', ''),
                'country': location.get('country', ''),
                'contact_name': ', '.join(contact.get('name', '') for contact in location.get('contacts', []) if contact.get('name')),
                'contact_phone': ', '.join(contact.get('phone', '') for contact in location.get('contacts', []) if contact.get('phone')),
                'contact_email': ', '.join(contact.get('email', '') for contact in location.get('contacts', []) if contact.get('email')),
            })

    return pd.DataFrame(rows)
