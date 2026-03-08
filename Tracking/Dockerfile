FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Root-level requirements (opencv, numpy, ffmpeg-python)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kafka producer helper (lives in Kafka/ at the repo root)
COPY Kafka/producer.py ./Kafka/producer.py

# Main tracking script
COPY Tracking/monitor_traffic.py .

CMD ["python", "monitor_traffic.py"]
