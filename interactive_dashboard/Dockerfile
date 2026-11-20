# Explicit image instead of buildpacks: the current Google buildpack builder
# (google-24) only ships Python 3.13+, and the pinned numpy/scipy/pandas here
# have no 3.13 wheels. `gcloud run deploy --source` uses this file when present.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Shell form so $PORT (set by Cloud Run) is expanded. Mirrors the old Procfile.
CMD exec gunicorn main:application --bind :$PORT --timeout 120
