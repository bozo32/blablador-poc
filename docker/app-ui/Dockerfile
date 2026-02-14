FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Optional: Graphviz binary for UI rendering.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install pinned runtime deps first for better layer caching.
COPY requirements/app.lock.txt /app/requirements/app.lock.txt
RUN python -m pip install --no-cache-dir -r /app/requirements/app.lock.txt

# Copy application code.
COPY backend /app/backend
COPY frontend /app/frontend
COPY application.py /app/application.py

EXPOSE 8501

CMD ["streamlit", "run", "frontend/ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
