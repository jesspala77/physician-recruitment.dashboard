import json
from urllib.parse import urlparse, parse_qs

from utils import clinicaltrials_api as ctg


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode('utf-8')


SAMPLE_STUDY_PAYLOAD = {
    'studies': [
        {
            'protocolSection': {
                'identificationModule': {
                    'nctId': 'NCT00000001',
                    'briefTitle': 'Study A',
                    'organization': {'fullName': 'Example Sponsor'},
                },
                'statusModule': {
                    'overallStatus': 'RECRUITING',
                    'lastUpdatePostDateStruct': {'date': '2026-04-01'},
                },
                'designModule': {
                    'studyType': 'INTERVENTIONAL',
                    'phases': ['PHASE2'],
                },
                'conditionsModule': {
                    'conditions': ['Asthma'],
                },
                'contactsLocationsModule': {
                    'locations': [
                        {
                            'facility': 'Boston Medical Center',
                            'status': 'RECRUITING',
                            'city': 'Boston',
                            'state': 'Massachusetts',
                            'country': 'United States',
                            'contacts': [
                                {
                                    'name': 'Pat Lee',
                                    'phone': '555-0100',
                                    'email': 'pat@example.org',
                                }
                            ],
                        }
                    ]
                },
            },
            'hasResults': False,
        }
    ]
}


def test_search_studies_builds_expected_query(monkeypatch):
    def fake_urlopen(url, timeout=20):
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert params['query.cond'] == ['asthma']
        assert params['query.locn'] == ['Boston, MA']
        assert params['filter.overallStatus'] == ['RECRUITING']
        assert params['pageSize'] == ['10']
        return DummyResponse(SAMPLE_STUDY_PAYLOAD)

    monkeypatch.setattr(ctg, 'urlopen', fake_urlopen)
    studies = ctg.search_studies('asthma', location='Boston, MA', overall_status='RECRUITING', page_size=10)
    assert len(studies) == 1


def test_search_studies_filters_by_phase(monkeypatch):
    monkeypatch.setattr(ctg, 'urlopen', lambda url, timeout=20: DummyResponse(SAMPLE_STUDY_PAYLOAD))
    studies = ctg.search_studies('asthma', phase='PHASE3')
    assert studies == []


def test_studies_to_dataframe_flattens_live_response():
    df = ctg.studies_to_dataframe(SAMPLE_STUDY_PAYLOAD['studies'])
    assert list(df['nct_id']) == ['NCT00000001']
    assert list(df['locations']) == [1]
    assert list(df['recruiting_locations']) == [1]
    assert df.loc[0, 'study_url'].endswith('NCT00000001')


def test_locations_to_dataframe_flattens_sites():
    df = ctg.locations_to_dataframe(SAMPLE_STUDY_PAYLOAD['studies'])
    assert list(df['facility']) == ['Boston Medical Center']
    assert list(df['contact_email']) == ['pat@example.org']


def test_get_api_version_returns_metadata(monkeypatch):
    monkeypatch.setattr(
        ctg,
        'urlopen',
        lambda url, timeout=20: DummyResponse({'apiVersion': '2.0.5', 'dataTimestamp': '2026-04-28T09:00:05'})
    )
    metadata = ctg.get_api_version()
    assert metadata['api_version'] == '2.0.5'
    assert metadata['data_timestamp'] == '2026-04-28T09:00:05'
