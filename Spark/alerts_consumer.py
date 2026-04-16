"""
alerts_consumer.py — Real-time speed alert consumer.

Subscribes to the `traffic-alerts` Kafka topic, logs every alert to stdout,
and persists each alert to the shared SQLite database at DB_PATH.

Schema consumed:
  {
    "vehicle_id"   : str,
    "vehicle_type" : str,
    "speed_kmh"    : float,
    "carriageway"  : str,
    "segment"      : str,   ← used as camera_name
    "timestamp_s"  : float,
    "speed_source" : str
  }

Environment variables:
  KAFKA_BOOTSTRAP  default: kafka:29092
  KAFKA_TOPIC_ALERTS default: traffic-alerts
  ALERT_SPEED_KMH  default: 130
  DB_PATH          default: /data/db/traffic.db
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timezone

from kafka import KafkaConsumer

# ── Configuration ──────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP",    "kafka:29092")
KAFKA_TOPIC      = os.getenv("KAFKA_TOPIC_ALERTS", "traffic-alerts")
GROUP_ID         = "traffic-alerts-group"
ALERT_THRESHOLD  = float(os.getenv("ALERT_SPEED_KMH", "130"))
DB_PATH          = os.getenv("DB_PATH", "/data/db/traffic.db")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("alerts_consumer")


# ── SQLite ────────────────────────────────────────────────────────────────────

def open_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the shared SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS speed_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name   TEXT    NOT NULL,
            vehicle_id    TEXT    NOT NULL,
            vehicle_type  TEXT    NOT NULL,
            speed_kmh     REAL    NOT NULL,
            carriageway   TEXT    NOT NULL,
            alerted_at    TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alert_cam ON speed_alerts (camera_name);
    """)
    conn.commit()
    log.info("SQLite ready at %s", db_path)
    return conn


def save_alert(conn: sqlite3.Connection, camera: str, vehicle_id: str,
               vehicle_type: str, speed_kmh: float, carriageway: str) -> None:
    """Insert alert record and commit."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO speed_alerts
               (camera_name, vehicle_id, vehicle_type, speed_kmh, carriageway, alerted_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (camera, vehicle_id, vehicle_type, round(speed_kmh, 2), carriageway, now),
    )
    conn.commit()


# ── Consumer loop ─────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Speed Alert Consumer starting ===")
    log.info("Kafka broker  : %s", KAFKA_BOOTSTRAP)
    log.info("Topic         : %s", KAFKA_TOPIC)
    log.info("Threshold     : %.0f km/h", ALERT_THRESHOLD)
    log.info("DB path       : %s", DB_PATH)
    log.info("-" * 60)

    conn = open_db(DB_PATH)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )

    alert_count = 0

    for message in consumer:
        try:
            ev      = message.value
            speed   = float(ev.get("speed_kmh", 0))
            vid     = ev.get("vehicle_id",   "UNKNOWN")
            vtype   = ev.get("vehicle_type", "vehicle")
            lane    = ev.get("carriageway",  "?")
            camera  = ev.get("segment",      "?")  # segment field carries camera_name

            # The tracking service already filters; double-check here
            if speed < ALERT_THRESHOLD:
                continue

            alert_count += 1

            # ── Log the alert ─────────────────────────────────────────
            log.warning(
                "⚠  SPEED ALERT #%d  |  cam=%-15s  %s %s  |  %.1f km/h  |  lane: %s",
                alert_count, camera, vtype.upper(), vid, speed, lane,
            )

            # ── Persist to SQLite ─────────────────────────────────────
            save_alert(conn, camera, vid, vtype, speed, lane)

        except (KeyError, ValueError, TypeError) as exc:
            log.error("Malformed alert message: %s  raw=%r", exc, message.value)

    log.info("Consumer loop ended. Total alerts logged: %d", alert_count)
    conn.close()


if __name__ == "__main__":
    main()
