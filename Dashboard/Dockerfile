FROM python:3.13-slim

WORKDIR /app

COPY Dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Dashboard/app.py .
COPY Dashboard/static ./static

# The page polls several endpoints every 2 s, so access logs are just noise
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
