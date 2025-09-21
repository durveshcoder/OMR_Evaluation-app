# OMR Evaluation System

This repository contains a Streamlit-based Automated OMR Evaluation System.

Files kept:
- `omr_app.py` - Main Streamlit application
- `omr_processor.py` - OMR processing helpers
- `requirements.txt` - Python dependencies

Removed files:
- `new_omr.py`, `run_cli.py`, `omr_results.db`, and `__pycache__` were removed as cleanup.

Quick start

1. Create a virtual environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run omr_app.py
```

Using camera capture

- Open the "OMR Evaluation" page and click "📷 Capture from Camera" to take a snapshot from the default webcam. Allow camera permissions if prompted.

Notes & troubleshooting

- If the camera does not open, ensure no other application is using it and that the correct drivers are installed.
- For best detection results, use high-resolution, well-lit, flat OMR sheet images.

If you need further cleanup or to restore any deleted files, tell me which files to restore.
