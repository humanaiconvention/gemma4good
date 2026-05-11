FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAESTRO_LAUNCH_MODE=test \
    PORT=8000

WORKDIR /app

# System deps kept minimal — fastapi/uvicorn/pydantic only.
COPY maestro_gateway/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Copy the gateway plus its only Python dependency: utils/merkle.py
COPY utils ./utils
COPY maestro_gateway ./maestro_gateway

EXPOSE 8000

# Healthcheck against /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn maestro_gateway.app:app --host 0.0.0.0 --port ${PORT}"]
