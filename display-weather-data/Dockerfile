# Use a slim Python base
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. fonts for matplotlib)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         libffi-dev libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects to listen on $PORT or 8080 if unset
ENV PORT 8080
EXPOSE 8080

# Use the Functions Framework as the entrypoint
CMD ["functions-framework", "--target", "display_weather_data", "--port", "8080"]
