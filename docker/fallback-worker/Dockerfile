FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     tesseract-ocr \
     tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/app.lock.txt /app/requirements/app.lock.txt
RUN python -m pip install --no-cache-dir -r /app/requirements/app.lock.txt

COPY backend /app/backend
COPY application.py /app/application.py

EXPOSE 8010

CMD ["uvicorn", "backend.ocr_worker_app:app", "--host", "0.0.0.0", "--port", "8010"]
