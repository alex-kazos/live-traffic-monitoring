"""
analytics.py — Spark Structured Streaming analytics for the traffic monitoring pipeline.

Consumes the `traffic-stats` Kafka topic and answers the following assignment queries:

  Q1. Speed of every vehicle (printed per event as they arrive).
  Q2. Total vehicle count per lane (inbound / outbound) over the whole video.
  Q3. Total speed-violation count (cars > 90 km/h, trucks > 80 km/h).
  Q5. Vehicle count per lane per 5-minute window.
  Q6. Average speed per lane per 5-minute window
       → log format: (inbound, 1st 5min, 60kmh)

Each Kafka message on `traffic-stats` is a JSON object with the schema:

  {
    "vehicle_id"   : str,      # unique track ID (e.g. "L3", "R7")
    "vehicle_type" : str,      # "car" | "truck" | "motorcycle" | ...
    "speed_kmh"    : float,    # measured speed
    "carriageway"  : str,      # "inbound" | "outbound"
    "segment"      : str,      # source video segment filename
    "timestamp_s"  : float,    # seconds-offset within the *full* 32-min video
    "speed_source" : str       # "tripwire" | "bev_avg"
  }
"""

import os
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)

# ── Configuration ──────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_STATS", "traffic-stats")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

CHECKPOINT_BASE = "/tmp/spark_checkpoints"
FIVE_MIN_SECS = 5 * 60  # 300 seconds – window size for Q5 / Q6


def _wait_for_kafka_topic(
    bootstrap: str, topic: str, retries: int = 30, delay: int = 5
) -> None:
    """
    Block until the Kafka topic exists, polling every `delay` seconds.
    Raises RuntimeError if the topic is still absent after `retries` attempts.
    """
    from kafka import KafkaAdminClient
    from kafka.errors import UnknownTopicOrPartitionError, NoBrokersAvailable

    print(f"[analytics] Waiting for Kafka topic '{topic}' on {bootstrap} ...")
    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap, request_timeout_ms=5000
            )
            topics = admin.list_topics()
            admin.close()
            if topic in topics:
                print(f"[analytics] Topic '{topic}' is ready (attempt {attempt}).")
                return
            print(
                f"[analytics] Topic '{topic}' not found yet (attempt {attempt}/{retries}). Retrying in {delay}s..."
            )
        except NoBrokersAvailable:
            print(
                f"[analytics] Kafka not reachable yet (attempt {attempt}/{retries}). Retrying in {delay}s..."
            )
        except Exception as exc:
            print(
                f"[analytics] Probe failed: {exc} (attempt {attempt}/{retries}). Retrying in {delay}s..."
            )
        time.sleep(delay)
    raise RuntimeError(f"Kafka topic '{topic}' did not appear after {retries * delay}s")


# ── Wait for the topic to be ready before starting Spark ───────────────────────
_wait_for_kafka_topic(KAFKA_BOOTSTRAP, KAFKA_TOPIC)

# ── Spark Session ──────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder.appName("TrafficAnalytics").master(SPARK_MASTER).getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"[analytics] Spark session started. Reading from Kafka topic '{KAFKA_TOPIC}'")

# ── Schema for the JSON payload ───────────────────────────────────────────────
event_schema = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("vehicle_type", StringType(), True),
        StructField("speed_kmh", FloatType(), True),
        StructField("carriageway", StringType(), True),
        StructField("segment", StringType(), True),
        StructField("timestamp_s", FloatType(), True),
        StructField("speed_source", StringType(), True),
    ]
)

# ── Read stream from Kafka ─────────────────────────────────────────────────────
raw_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

# Parse the JSON value and cast timestamp_s to a proper event-time column
events = (
    raw_stream.select(
        F.from_json(F.col("value").cast("string"), event_schema).alias("d")
    ).select("d.*")
    # Convert the float seconds-offset into a real timestamp for windowing.
    # We use 1970-01-01 00:00:00 as the epoch anchor so that 0 s → 00:00,
    # 300 s → 00:05, etc. – which lets Spark build clean 5-minute windows.
    .withColumn(
        "event_time",
        (
            F.to_timestamp(F.lit("1970-01-01 00:00:00"))
            + F.expr("INTERVAL 1 SECOND") * F.col("timestamp_s").cast("long")
        ),
    )
    # Drop rows with null speed (no measurement was possible)
    .filter(F.col("speed_kmh").isNotNull())
)

# ═══════════════════════════════════════════════════════════════════════════════
# Q1 – Speed of every vehicle (streaming print as events arrive)
# ═══════════════════════════════════════════════════════════════════════════════
q1 = (
    events.select(
        "vehicle_id",
        "carriageway",
        "vehicle_type",
        "speed_kmh",
        "speed_source",
        "segment",
    )
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/q1")
    .queryName("Q1_Vehicle_Speeds")
    .start()
)

# ═══════════════════════════════════════════════════════════════════════════════
# Q2 – Total vehicle count per lane (complete mode – updated as new data arrives)
# ═══════════════════════════════════════════════════════════════════════════════
q2 = (
    events.groupBy("carriageway")
    .agg(F.approx_count_distinct("vehicle_id").alias("vehicle_count"))
    .writeStream.outputMode("complete")
    .format("console")
    .option("truncate", False)
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/q2")
    .queryName("Q2_Vehicle_Count_Per_Lane")
    .start()
)

# ═══════════════════════════════════════════════════════════════════════════════
# Q3 – Speed violation count
#       cars  > 90 km/h
#       trucks > 80 km/h  (bus / motorcycle use the car threshold for simplicity)
# ═══════════════════════════════════════════════════════════════════════════════
violations = events.filter(
    ((F.col("vehicle_type") == "truck") & (F.col("speed_kmh") > 80))
    | ((F.col("vehicle_type") != "truck") & (F.col("speed_kmh") > 90))
)

q3 = (
    violations.groupBy("carriageway", "vehicle_type")
    .agg(F.count("*").alias("violation_count"))
    .writeStream.outputMode("complete")
    .format("console")
    .option("truncate", False)
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/q3")
    .queryName("Q3_Speed_Violations")
    .start()
)

# ═══════════════════════════════════════════════════════════════════════════════
# Q5 – Vehicle count per lane per 5-minute window
# ═══════════════════════════════════════════════════════════════════════════════
q5 = (
    events.withWatermark("event_time", "5 minutes")
    .groupBy(F.window("event_time", "5 minutes"), "carriageway")
    .agg(F.approx_count_distinct("vehicle_id").alias("vehicle_count"))
    # Format the window start as an ordinal label: "1st 5min", "2nd 5min", etc.
    .select(
        "carriageway",
        "vehicle_count",
        F.col("window.start").alias("window_start"),
        # Window index (1-based)
        ((F.unix_timestamp("window.start") / FIVE_MIN_SECS).cast("int") + 1).alias(
            "window_index"
        ),
    )
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/q5")
    .queryName("Q5_Vehicles_Per_Lane_Per_5min")
    .start()
)

# ═══════════════════════════════════════════════════════════════════════════════
# Q6 – Average speed per lane per 5-minute window
#       Log format: (inbound, 1st 5min, 60kmh)
# ═══════════════════════════════════════════════════════════════════════════════


def _ordinal(n: int) -> str:
    """Return English ordinal string for integer n (1→'1st', 2→'2nd', …)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return {1: f"{n}st", 2: f"{n}nd", 3: f"{n}rd"}.get(n % 10, f"{n}th")


# UDF to convert a window index to an ordinal label
ordinal_udf = F.udf(lambda n: _ordinal(int(n)), StringType())

q6_base = (
    events.withWatermark("event_time", "5 minutes")
    .groupBy(F.window("event_time", "5 minutes"), "carriageway")
    .agg(F.avg("speed_kmh").alias("avg_speed_kmh"))
    .select(
        "carriageway",
        F.round("avg_speed_kmh", 0).cast(IntegerType()).alias("avg_speed_kmh"),
        ((F.unix_timestamp("window.start") / FIVE_MIN_SECS).cast("int") + 1).alias(
            "window_index"
        ),
    )
)

# Build the formatted log string  (inbound, 3rd 5min, 72kmh)
q6 = (
    q6_base.withColumn(
        "log_entry",
        F.concat_ws(
            ", ",
            F.col("carriageway"),
            F.concat(ordinal_udf(F.col("window_index")), F.lit(" 5min")),
            F.concat(F.col("avg_speed_kmh").cast("string"), F.lit("kmh")),
        ),
    )
    # Wrap in parentheses as per the spec
    .withColumn("log_entry", F.concat(F.lit("("), F.col("log_entry"), F.lit(")")))
    .select("log_entry", "carriageway", "window_index", "avg_speed_kmh")
    .writeStream.outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/q6")
    .queryName("Q6_Avg_Speed_Per_Lane_Per_5min")
    .start()
)

# ── Wait for all queries ───────────────────────────────────────────────────────
print("[analytics] All streaming queries started. Waiting for data...")
spark.streams.awaitAnyTermination()
