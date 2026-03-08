"""
alerts_consumer.py — Real-time alert consumer (Query 4).

Subscribes to the `traffic-alerts` Kafka topic and logs a human-readable
alert to stdout whenever a vehicle is detected travelling at more than
130 km/h.

This is intentionally a lightweight pure-Python consumer (kafka-python)
rather than a Spark job so that it starts instantly with minimal overhead
and truly delivers *real-time* notifications without batching latency.

Each Kafka message on `traffic-alerts` is a JSON object with the schema:

  {
    "vehicle_id"   : str,
    "vehicle_type" : str,
    "speed_kmh"    : float,
    "carriageway"  : str,
    "segment"      : str,
    "timestamp_s"  : float
  }
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone

from kafka import KafkaConsumer

# ── Configuration ──────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_ALERTS", "traffic-alerts")
GROUP_ID = "traffic-alerts-group"
ALERT_THRESHOLD = float(os.getenv("ALERT_SPEED_KMH", "130"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("alerts")


# ── Helper: convert video-offset seconds → human time label ──────────────────
def _video_time(seconds: float) -> str:
    """Convert a seconds offset into a MM:SS label (e.g. 305.2 → '05:05')."""
    td = timedelta(seconds=int(seconds))
    mins, secs = divmod(int(td.total_seconds()), 60)
    return f"{mins:02d}:{secs:02d}"


# ── Consumer loop ─────────────────────────────────────────────────────────────
def main():
    log.info("Starting real-time alert consumer.")
    log.info("Kafka broker : %s", KAFKA_BOOTSTRAP)
    log.info("Topic        : %s", KAFKA_TOPIC)
    log.info("Threshold    : %.0f km/h", ALERT_THRESHOLD)
    log.info("-" * 60)

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
            event = message.value
            speed = float(event.get("speed_kmh", 0))
            vid = event.get("vehicle_id", "UNKNOWN")
            vtype = event.get("vehicle_type", "vehicle").upper()
            carriageway = event.get("carriageway", "?")
            segment = event.get("segment", "?")
            ts_s = float(event.get("timestamp_s", 0))

            # The tracking service already filters; double-check here
            if speed < ALERT_THRESHOLD:
                continue

            alert_count += 1
            video_pos = _video_time(ts_s)

            log.warning(
                "⚠  SPEED ALERT #%d  |  %s %s  |  %.1f km/h  |  "
                "lane: %s  |  video offset: %s  |  segment: %s",
                alert_count,
                vtype,
                vid,
                speed,
                carriageway,
                video_pos,
                segment,
            )

        except (KeyError, ValueError, TypeError) as exc:
            log.error("Malformed alert message: %s  raw=%r", exc, message.value)

    log.info("Consumer loop ended. Total alerts logged: %d", alert_count)


if __name__ == "__main__":
    main()
