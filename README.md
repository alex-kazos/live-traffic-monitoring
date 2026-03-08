# Traffic Monitoring Pipeline

A scalable traffic monitoring system built with **Apache Kafka**, **Apache Spark Structured Streaming**, and **OpenCV / YOLOv8**. Processes video streams from traffic cameras to detect vehicles, measure speeds, count violations, and emit real-time alerts.

---

## Architecture

```
[ Source Video ]
       │
       ▼
  ┌─────────┐     (2-min segments)    ┌─────────────┐
  │  Split  │ ──────────────────────▶ │ /data/      │
  │ Service │                         │  segments/  │
  └─────────┘                         └──────┬──────┘
                                             │
             ┌───────────────────────────────┘
             │  (one container per segment — horizontally scalable)
             ▼
  ┌──────────────────┐
  │ Tracking Service │   opencv   vehicle detection + speed calculation
  │  (×N replicas)   │ ─────────────────────────────────────────────▶
  └────────┬─────────┘
           │                   ┌──────────────────┐
           │ traffic-stats ──▶ │  Spark Analytics │  Q1-Q3, Q5, Q6
           │                   └──────────────────┘
           │ traffic-alerts ─▶ ┌──────────────────┐
           │                   │  Alerts Consumer │  Q4 (>130 km/h)
           │                   └──────────────────┘
           ▼
       [ Kafka ]  (Zookeeper)
```

---

## Services

| Service           | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `zookeeper`       | Kafka coordination                                             |
| `kafka`           | Message broker — two topics: `traffic-stats`, `traffic-alerts` |
| `kafka-init`      | One-shot topic creator                                         |
| `split`           | Splits the 32-min source video into 2-min segments             |
| `tracking`        | Detects & tracks vehicles per segment, publishes to Kafka      |
| `spark-master`    | Spark standalone master                                        |
| `spark-worker`    | Spark worker node                                              |
| `spark-analytics` | Spark Structured Streaming — answers Q1, Q2, Q3, Q5, Q6        |
| `spark-alerts`    | Lightweight consumer — answers Q4 (real-time >130 km/h log)    |

---

## Assignment Queries

| #     | Query                                        | Implementation                                        |
| ----- | -------------------------------------------- | ----------------------------------------------------- |
| Q1    | Speed of every vehicle                       | `spark-analytics` — streamed to console per event     |
| Q2    | Vehicle count per lane                       | `spark-analytics` — `countDistinct` over full video   |
| Q3    | Speed violations (cars >90, trucks >80)      | `spark-analytics` — filtered aggregate                |
| Q4    | Real-time alert if >130 km/h                 | `spark-alerts` — Kafka `traffic-alerts` consumer      |
| Q5    | Vehicle count per lane per 5 min             | `spark-analytics` — 5-min tumbling window             |
| Q6    | Avg speed per lane per 5 min                 | `spark-analytics` — log: `(inbound, 1st 5min, 60kmh)` |
| Bonus | Perspective-corrected speed (25 m reference) | `monitor_traffic.py` — BEV homography                 |

---

## Quick Start

### Prerequisites

- Docker Desktop with Compose V2 enabled
- The 32-minute source video placed at `Downloads/traffic_video.mp4`
  (or mount it into the `video-data` volume)

### 1 — Launch the full pipeline

```cmd
docker compose up
```

Spark Web UI → [http://localhost:8080](http://localhost:8080)

### 2 — Scale the Tracking service (horizontal scaling)

```cmd
docker compose up --scale tracking=4
```

Each replica picks up a different video segment from the shared `/data/segments` volume.

### 3 — Standalone (no Kafka)

The tracking scripts fall back to local CSV output when Kafka is unreachable:

```bash
# Traditional CV (fast)
python Tracking/monitor_traffic.py

# YOLOv8 + ByteTrack (accurate)
python Other/monitor_traffic_supervision.py
```

---

## Installation (local dev)

```bash
pip install -r requirements.txt
```

Major dependencies: `opencv-python`, `numpy`, `ffmpeg-python`, `kafka-python`

For the YOLOv8 version also install:

```bash
pip install -r Other/requirements.txt  # ultralytics, supervision
```

---

## Generated Artifacts

| File                                   | Description                                     |
| -------------------------------------- | ----------------------------------------------- |
| `traffic_speed_output.mp4`             | Annotated video with bounding boxes & speed HUD |
| `vehicle_speeds.csv`                   | Per-vehicle speed report                        |
| `traffic_speed_output_supervision.mp4` | YOLOv8-annotated video                          |
| `vehicle_speeds_supervision.csv`       | YOLOv8 speed report                             |

---

## Project Structure

```
traffic-monitoring/
├── docker-compose.yml          ← Full pipeline orchestration
├── start_docker.bat            ← Build helper + usage hints
├── requirements.txt
│
├── Split/
│   ├── Dockerfile
│   └── video_split.py          ← Splits source video into 2-min clips
│
├── Tracking/
│   ├── Dockerfile
│   └── monitor_traffic.py      ← CV detection + Kafka producer
│
├── Kafka/
│   └── producer.py             ← Shared Kafka publishing helper
│
├── Spark/
│   ├── Dockerfile
│   ├── analytics.py            ← Spark Structured Streaming (Q1-Q3, Q5-Q6)
│   └── alerts_consumer.py      ← Real-time alert consumer (Q4)
│
└── Other/
    ├── monitor_traffic_supervision.py   ← YOLOv8 + ByteTrack variant
    ├── split_video.py                   ← Google Drive downloader + splitter
    └── yolov8x.pt                       ← YOLO model weights
```
