FROM python:3.13-slim

WORKDIR /app

COPY LocalDashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY LocalDashboard/dashboard.py .
COPY LocalDashboard/assets ./assets

# Listen on all interfaces so the published port reaches it
CMD ["python", "-u", "dashboard.py", "--host", "0.0.0.0"]
