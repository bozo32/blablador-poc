# Blablador‑assisted Citation‑Support Checker

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python application.py --csv data/source.csv --api_key sk-...
```

* The backend index is built from `/source/<Author>.pdf(.tei.xml)`.
* Open <http://localhost:8501> to start segmenting sentences and trigger NLI checks.
* Intermediate results stream into `output.json`.
