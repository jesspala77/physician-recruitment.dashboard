# Contributing

Thank you for contributing to the CRSsNP Physician-Site Recruitment Dashboard.

## Local setup

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Run the app

```powershell
python -m streamlit run .\streamlit_app.py
```

## Run tests

```powershell
pytest
```

## Guidelines

- Keep changes focused and easy to review.
- Preserve existing behavior unless the change explicitly updates it.
- Add or update tests when behavior changes.
- Update documentation when setup or usage changes.
