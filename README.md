# Traffic Monitoring Pipeline

A real-time traffic monitoring system built with **Apache Kafka**, **Apache Spark Structured Streaming**, and **OpenCV**. Processes live HLS streams from the **Attiki Odos** highway cameras (naodos.gr) to detect vehicles, measure speeds, persist data to SQLite, and emit real-time alerts for speeds exceeding 130 km/h.

---

## Monitored Cameras

| Camera | ipcamlive Alias | Description |
|---|---|---|
| **I/C Papagou (Imittos Ring Road)** | `cam231` | Direction Katechaki / Mesogeia-Vrilissia |
| **Roupaki Toll Station** | `cam88` | Direction Elefsina / Athens Airport |

Stream URLs are resolved dynamically at runtime via the ipcamlive API.

---

## Pipeline overview

The default `docker-compose.yml` stack runs without Zookeeper and follows this flow:

1. Live HLS streams from `cam231` and `cam88` are consumed by the two tracking containers.
2. The tracking services publish vehicle events to Kafka topics named `traffic-stats` and `traffic-alerts`.
3. `spark-analytics` reads `traffic-stats` and answers Q1, Q2, Q3, Q5, and Q6.
4. `spark-alerts` reads `traffic-alerts`, prints real-time alerts, and stores them in SQLite.

In short: streams -> tracking -> Kafka -> Spark consumers -> SQLite.

---

## Services

| Service | Description |
|---|---|
| `kafka` | Kafka broker in KRaft mode; no Zookeeper is used |
| `kafka-init` | One-shot topic creator for `traffic-stats` and `traffic-alerts` |
| `tracking-papagou` | Live vehicle tracking from cam231 (I/C Papagou) |
| `tracking-roupaki` | Live vehicle tracking from cam88 (Roupaki Toll Station) |
| `spark-analytics` | Spark Structured Streaming for Q1, Q2, Q3, Q5, and Q6 |
| `spark-alerts` | Kafka consumer for Q4 alerts and SQLite persistence |

---

## SQLite Database

The shared SQLite database at `/data/db/traffic.db` (Docker volume `db-data`) is created by the tracking service and reused by the alerts consumer.

### `vehicle_events` table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Autoincrement primary key |
| `camera_name` | TEXT | `IC_Papagou` or `Roupaki_Toll_Station` |
| `vehicle_id` | TEXT | Track ID such as `L3` or `R12` |
| `vehicle_type` | TEXT | `car` or `truck` |
| `speed_kmh` | REAL | Measured speed in km/h |
| `direction` | TEXT | Road direction label, such as `to_Elefsina` |
| `movement` | TEXT | Detected movement (`left`, `right`, or `unknown`) |
| `speed_source` | TEXT | `tripwire` or `displacement` |
| `recorded_at` | TEXT | ISO-8601 UTC timestamp |

### `speed_alerts` table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Autoincrement primary key |
| `camera_name` | TEXT | Source camera |
| `vehicle_id` | TEXT | Track ID |
| `vehicle_type` | TEXT | `car` or `truck` |
| `speed_kmh` | REAL | Speed that triggered the alert |
| `direction` | TEXT | Road direction label |
| `alerted_at` | TEXT | ISO-8601 UTC timestamp |

---

## Assignment Queries

| # | Query | Implementation |
|---|---|---|
| Q1 | Speed of every vehicle | `spark-analytics` — streamed per event |
| Q2 | Vehicle count per lane | `spark-analytics` — `countDistinct` |
| Q3 | Speed violations (cars >90, trucks >80) | `spark-analytics` — filtered aggregate |
| Q4 | Real-time alert if >130 km/h | `spark-alerts` — Kafka consumer + SQLite |
| Q5 | Vehicle count per lane per 5 min | `spark-analytics` — 5-min tumbling window |
| Q6 | Avg speed per lane per 5 min | `spark-analytics` — windowed avg |
| Bonus | Perspective-corrected speed | `monitor_traffic.py` — BEV homography |

---

## Quick Start

### Prerequisites

- Docker Desktop with Compose V2 enabled
- Internet access (to resolve ipcamlive HLS streams)

### Launch the full pipeline

```powershell
docker compose up --build
```

The SQLite database is created automatically in the `db-data` Docker volume.

### Inspect the database from the host

```powershell
docker run --rm -v traffic-monitoring_db-data:/db alpine sh -c "apk add --no-cache sqlite && sqlite3 /db/traffic.db '.tables'"
```

---

## Installation (local dev)

```powershell
pip install -r requirements.txt
```

Base dependencies: `opencv-python`, `numpy`, `ffmpeg-python`, `kafka-python-ng`.

If you want the optional YOLOv8 and ByteTrack experiment in `Other/monitor_traffic_supervision.py`, install the extra package set as well:

```powershell
pip install -r Other/requirements.txt
```

---

## Project structure

```
traffic-monitoring/
├── docker-compose.yml
├── requirements.txt
├── Kafka/
│   └── producer.py
├── Tracking/
│   ├── Dockerfile
│   └── monitor_traffic.py
├── Spark/
│   ├── Dockerfile
│   ├── analytics.py
│   └── alerts_consumer.py
├── Other/
│   ├── monitor_traffic_supervision.py   # optional YOLOv8 + ByteTrack variant
│   ├── get-model.bat                    # downloads the optional model weight
│   └── requirements.txt                 # optional experiment dependencies
├── Split/                               # legacy split-video utilities; not used by docker compose
└── start_docker.bat                     # legacy launcher for the older split-based workflow
```
