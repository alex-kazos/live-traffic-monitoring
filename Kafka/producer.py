"""
producer.py — Kafka producer helper for the Traffic Monitoring pipeline.

This module is imported by monitor_traffic.py (the Tracking service) to
publish per-vehicle detection events to two Kafka topics:

  traffic-stats   — every vehicle with a measured speed (used by Spark analytics)
  traffic-alerts  — vehicles exceeding ALERT_SPEED_KMH (used by alerts consumer)

Each message is a JSON-serialised dict with the schema:

  {
    "vehicle_id"   : str,    # e.g. "L3" or "R7"
    "vehicle_type" : str,    # "car" | "truck"
    "speed_kmh"    : float,
    "carriageway"  : str,    # "inbound" (left side) | "outbound" (right side)
    "segment"      : str,    # source video filename (for traceability)
    "timestamp_s"  : float,  # seconds offset in the FULL 32-min video
    "speed_source" : str     # "tripwire" | "bev_avg"
  }

The producer is created lazily on first use and is shared for the lifetime
of the tracking process.  If Kafka is unavailable the module degrades
gracefully: events are silently dropped so that the CV pipeline continues.
"""

import os
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Configuration (read from environment, with sensible defaults) ──────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC_STATS = os.getenv("KAFKA_TOPIC_STATS", "traffic-stats")
TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "traffic-alerts")
ALERT_SPEED_KMH = float(os.getenv("ALERT_SPEED_KMH", "130"))

# ── Lazy producer singleton ───────────────────────────────────────────────────
_producer = None
_kafka_available = True  # set to False after the first connection failure


def _get_producer():
    """Return the shared KafkaProducer, creating it on first call."""
    global _producer, _kafka_available
    if not _kafka_available:
        return None
    if _producer is not None:
        return _producer
    try:
        from kafka import KafkaProducer  # imported lazily to avoid hard dep

        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,  # wait for leader acknowledgment
            retries=3,
            linger_ms=50,  # micro-batch for throughput
            batch_size=16384,
        )
        log.info("[producer] Connected to Kafka at %s", KAFKA_BOOTSTRAP)
        return _producer
    except Exception as exc:
        log.warning(
            "[producer] Kafka unavailable (%s). "
            "Events will be skipped – CV pipeline continues.",
            exc,
        )
        _kafka_available = False
        return None


# ── Public API ─────────────────────────────────────────────────────────────────


def publish_vehicle_event(
    *,
    vehicle_id: str,
    vehicle_type: str,
    speed_kmh: float,
    carriageway: str,
    segment: str,
    timestamp_s: float,
    speed_source: str,
) -> None:
    """
    Publish a vehicle speed event to the appropriate Kafka topic(s).

    Args:
        vehicle_id   : Unique track ID string (e.g. "L3").
        vehicle_type : Classification – "car" or "truck".
        speed_kmh    : Measured speed in km/h.
        carriageway  : "inbound" (left carriageway) or "outbound" (right).
        segment      : Source video filename for traceability.
        timestamp_s  : Seconds offset within the full 32-minute video.
        speed_source : How the speed was measured ("tripwire" | "bev_avg").
    """
    producer = _get_producer()
    if producer is None:
        return

    payload = {
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle_type,
        "speed_kmh": round(float(speed_kmh), 2),
        "carriageway": carriageway,
        "segment": segment,
        "timestamp_s": round(float(timestamp_s), 2),
        "speed_source": speed_source,
    }

    try:
        # Always publish to traffic-stats
        producer.send(TOPIC_STATS, value=payload)

        # Additionally publish to traffic-alerts if speed exceeds threshold
        if speed_kmh > ALERT_SPEED_KMH:
            producer.send(TOPIC_ALERTS, value=payload)
            log.warning(
                "[ALERT] Vehicle %s (%s) doing %.1f km/h on %s lane — "
                "alert published to %s",
                vehicle_id,
                vehicle_type,
                speed_kmh,
                carriageway,
                TOPIC_ALERTS,
            )
    except Exception as exc:
        log.error("[producer] Failed to publish event for %s: %s", vehicle_id, exc)


def flush() -> None:
    """Flush any buffered messages to Kafka (call at end of segment processing)."""
    if _producer is not None:
        try:
            _producer.flush(timeout=10)
        except Exception as exc:
            log.error("[producer] Flush failed: %s", exc)


def close() -> None:
    """Gracefully close the Kafka producer."""
    global _producer
    if _producer is not None:
        try:
            _producer.close(timeout=10)
        except Exception:
            pass
        _producer = None
