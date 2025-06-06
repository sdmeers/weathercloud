# ---- Build image ----
FROM python:3.11-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy your Streamlit app
COPY . .

# 3. Tell Cloud Run which port we’ll listen on
ENV PORT 8080
EXPOSE 8080

# 4. Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true"]