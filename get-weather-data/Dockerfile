# ---- Build image ----
FROM python:3.11-slim

# 1. Install deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy code
COPY . .

# 3. Expose the default Cloud Run port
ENV PORT=8080
EXPOSE 8080

# 4. Start the MCP server
CMD ["python", "server.py"]