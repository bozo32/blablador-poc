FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install pinned runtime deps first for better layer caching.
COPY requirements/app.lock.txt /app/requirements/app.lock.txt
RUN python -m pip install --no-cache-dir -r /app/requirements/app.lock.txt

# Copy application code.
COPY backend /app/backend
COPY frontend /app/frontend
COPY application.py /app/application.py

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
