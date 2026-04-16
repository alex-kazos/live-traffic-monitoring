"""
monitor_traffic.py — Real-time traffic tracking from ipcamlive livestreams.

Designed for SIDE-MOUNTED highway cameras (naodos.gr / Attiki Odos) where
vehicles move LEFT ↔ RIGHT across the frame:

  • I/C Papagou (cam231) : left = to Katechaki   | right = to Mesogeia-Vrilissia
  • Roupaki Toll (cam88)  : left = to Elefsina    | right = to Athens Airport

Speed is measured with VERTICAL TRIPWIRES (X-based crossing detection).
Displacement-based fallback speed uses pixel-to-metre calibration.

All per-camera parameters are environment-variable driven so the same image
can run two independent containers with no code changes.

Configuration env vars:
  INPUT_VIDEO        ipcamlive alias, e.g. "cam231"      (required)
  CAMERA_NAME        human label, e.g. "IC_Papagou"      (optional)
  KAFKA_BOOTSTRAP    default: kafka:29092
  KAFKA_TOPIC_STATS  default: traffic-stats
  KAFKA_TOPIC_ALERTS default: traffic-alerts
  ALERT_SPEED_KMH    km/h threshold for alerts           (default: 130)
  DB_PATH            SQLite file path                    (default: /data/db/traffic.db)

  --- Geometry / Calibration ---
  SPLIT_COORD        X pixel that divides left / right carriageway  (default: 640)
  LINE_L1_COORD      Left carriage first  tripwire X (car enters)   (default: 580)
  LINE_L2_COORD      Left carriage second tripwire X (car exits)    (default: 100)
  LINE_R1_COORD      Right carriage first  tripwire X               (default: 700)
  LINE_R2_COORD      Right carriage second tripwire X               (default: 1180)
  DIST_REAL_M        Real-world distance (m) between the two lines  (default: 28.0)
  PIXELS_PER_METER   Pixel-to-metre ratio for fallback speed        (default: 16.0)

  --- Lane Labels ---
  LANE_LEFT_LABEL    e.g. "to_Elefsina"          (default: left_lane)
  LANE_RIGHT_LABEL   e.g. "to_Athens_Airport"    (default: right_lane)
"""

import os
import sys
import re
import cv2
import time
import sqlite3
import logging
import numpy as np
import urllib.request
import json
from collections import defaultdict, deque
from datetime import datetime, timezone

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("monitor")

# ---------------------------------------------------------------------------
# Kafka producer (imported lazily – silently disabled if unavailable)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kafka"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kafka"))
    import producer as kafka_producer
    _KAFKA_ENABLED = True
    log.info("Kafka producer module loaded.")
except ImportError:
    _KAFKA_ENABLED = False
    log.warning("Kafka producer not found — events persisted to SQLite only.")

# ---------------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------------
_ENV_VIDEO  = os.getenv("INPUT_VIDEO", "")
CAMERA_NAME = os.getenv("CAMERA_NAME", _ENV_VIDEO or "unknown_cam")
DB_PATH     = os.getenv("DB_PATH", "/data/db/traffic.db")
ALERT_SPEED = float(os.getenv("ALERT_SPEED_KMH", "130"))

if not _ENV_VIDEO:
    log.error("INPUT_VIDEO environment variable is not set. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Camera geometry & calibration — all tunable via env vars
# ---------------------------------------------------------------------------
# X-coordinate that splits the two carriageways (always X-based on position)
SPLIT_COORD = int(os.getenv("SPLIT_COORD", "640"))

# Vertical tripwires: X positions where vehicles cross
# Left carriageway:  cars move RIGHT→LEFT, so LINE_L1 > LINE_L2
LINE_L1 = int(os.getenv("LINE_L1_COORD", "580"))   # left-moving: first cross (higher X)
LINE_L2 = int(os.getenv("LINE_L2_COORD", "100"))   # left-moving: second cross (lower X)
# Right carriageway: cars move LEFT→RIGHT, so LINE_R1 < LINE_R2
LINE_R1 = int(os.getenv("LINE_R1_COORD", "700"))   # right-moving: first cross (lower X)
LINE_R2 = int(os.getenv("LINE_R2_COORD", "1180"))  # right-moving: second cross (higher X)

# Real-world distance between the two tripwires (metres).
# Tune this value for each camera by measuring a known ground feature.
DIST_M = float(os.getenv("DIST_REAL_M", "28.0"))

# Pixel-to-metre scaling for displacement fallback speed.
# Estimate: (visible road length in m) / (visible pixel width per carriageway)
# e.g. 40 m road visible across 640 px → 640/40 = 16.0
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", "16.0"))

# Human-readable direction labels for each carriageway side
LANE_LEFT_LABEL  = os.getenv("LANE_LEFT_LABEL",  "left_lane")
LANE_RIGHT_LABEL = os.getenv("LANE_RIGHT_LABEL", "right_lane")

# ---------------------------------------------------------------------------
# Tracking constants
# ---------------------------------------------------------------------------
FPS            = 25.0
MIN_AREA       = 1_500     # min contour area (px²) to be considered a vehicle
TRUCK_MIN_AREA = 9_000     # bounding-box area threshold for truck vs car
MAX_LOST       = 12        # frames before a track is dropped
MAX_DIST       = 180       # max pixel distance for track matching

DISP_WINDOW    = 10        # frames of x-position history for displacement speed
DISP_MIN_PTS   = 8         # min history points before computing displacement speed
DISP_SPD_MIN   = 5.0       # km/h  — ignore slower readings (noise)
DISP_SPD_MAX   = 200.0     # km/h  — cap unreasonably high values

VEL_THRESHOLD_PX = 10      # min |Δx| over the history window to call a direction

LOG_EVERY_N_FRAMES = 125   # ~5 s at 25 fps

# ---------------------------------------------------------------------------
# ipcamlive API
# ---------------------------------------------------------------------------
_IPCAM_SECRET = "65586c9ba88ef"
_IPCAM_API    = "https://ipcamlive.com/api/v2"


def get_stream_url(alias: str) -> str:
    """Fetch dynamic m3u8 URL from ipcamlive API."""
    url = (f"{_IPCAM_API}/getstreamhlsurl"
           f"?apisecret={_IPCAM_SECRET}&alias={alias}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("result") == "ok" and "url" in data.get("data", {}):
                return data["data"]["url"].replace("http://", "https://")
    except Exception as exc:
        log.error("Failed to fetch HLS URL for alias '%s': %s", alias, exc)
    return ""


# ---------------------------------------------------------------------------
# SQLite — schema & helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Create / open the SQLite DB and ensure the schema exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vehicle_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name   TEXT    NOT NULL,
            vehicle_id    TEXT    NOT NULL,
            vehicle_type  TEXT    NOT NULL,
            speed_kmh     REAL,
            direction     TEXT    NOT NULL,   -- named road direction (e.g. "to_Elefsina")
            movement      TEXT,               -- detected movement: "left"|"right"|"unknown"
            speed_source  TEXT    NOT NULL,   -- "tripwire"|"displacement"
            recorded_at   TEXT    NOT NULL    -- ISO-8601 UTC
        );

        CREATE TABLE IF NOT EXISTS speed_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name   TEXT    NOT NULL,
            vehicle_id    TEXT    NOT NULL,
            vehicle_type  TEXT    NOT NULL,
            speed_kmh     REAL    NOT NULL,
            direction     TEXT    NOT NULL,
            alerted_at    TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ve_cam    ON vehicle_events (camera_name);
        CREATE INDEX IF NOT EXISTS idx_ve_speed  ON vehicle_events (speed_kmh);
        CREATE INDEX IF NOT EXISTS idx_ve_dir    ON vehicle_events (direction);
        CREATE INDEX IF NOT EXISTS idx_alert_cam ON speed_alerts   (camera_name);
        CREATE INDEX IF NOT EXISTS idx_alert_dir ON speed_alerts   (direction);
    """)
    conn.commit()
    log.info("SQLite ready at %s", db_path)
    return conn


def insert_event(conn: sqlite3.Connection, camera: str, vehicle_id: str,
                 vehicle_type: str, speed_kmh: float | None, direction: str,
                 movement: str, speed_source: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO vehicle_events
               (camera_name, vehicle_id, vehicle_type, speed_kmh,
                direction, movement, speed_source, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (camera, vehicle_id, vehicle_type,
         round(speed_kmh, 2) if speed_kmh is not None else None,
         direction, movement, speed_source, now),
    )
    conn.commit()


def insert_alert(conn: sqlite3.Connection, camera: str, vehicle_id: str,
                 vehicle_type: str, speed_kmh: float, direction: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO speed_alerts
               (camera_name, vehicle_id, vehicle_type, speed_kmh, direction, alerted_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (camera, vehicle_id, vehicle_type, round(speed_kmh, 2), direction, now),
    )
    conn.commit()
    log.warning(
        "⚠  SPEED ALERT  |  cam=%-20s  %s %s  |  %.1f km/h  |  %s",
        camera, vehicle_type.upper(), vehicle_id, speed_kmh, direction,
    )


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def init_tracker() -> dict:
    return {"next_id": 0, "centroids": {}, "lost": defaultdict(int)}


def update_tracker(tracker: dict, boxes: list) -> dict:
    """
    Match bounding boxes to existing tracks (nearest-centroid).
    Uses centre-of-box as tracking point (better for side-view cameras).
    """
    centroids = tracker["centroids"]
    lost      = tracker["lost"]
    # Centre point of each detection
    new_pts = [(int(x + w // 2), int(y + h // 2)) for x, y, w, h in boxes]

    if not boxes:
        for tid in list(centroids):
            lost[tid] += 1
            if lost[tid] > MAX_LOST:
                del centroids[tid]
                del lost[tid]
        return dict(centroids)

    if not centroids:
        for pt in new_pts:
            centroids[tracker["next_id"]] = pt
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1
        return dict(centroids)

    existing_ids = list(centroids)
    existing_pts = [centroids[i] for i in existing_ids]
    dist = np.linalg.norm(
        np.array(existing_pts)[:, None] - np.array(new_pts)[None, :], axis=-1
    )
    pairs = sorted(
        [(dist[i, j], i, j)
         for i in range(len(existing_pts))
         for j in range(len(new_pts))],
        key=lambda x: x[0],
    )
    used_ex, used_nw = set(), set()
    for d, ei, ni in pairs:
        if ei in used_ex or ni in used_nw:
            continue
        if d > MAX_DIST:
            break
        tid = existing_ids[ei]
        centroids[tid] = new_pts[ni]
        lost[tid] = 0
        used_ex.add(ei)
        used_nw.add(ni)

    for ei, tid in enumerate(existing_ids):
        if ei not in used_ex:
            lost[tid] += 1
            if lost[tid] > MAX_LOST:
                del centroids[tid]
                del lost[tid]

    for ni, pt in enumerate(new_pts):
        if ni not in used_nw:
            centroids[tracker["next_id"]] = pt
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1

    return dict(centroids)


# ---------------------------------------------------------------------------
# VERTICAL TRIPWIRE helpers (X-based, for horizontal vehicle motion)
# ---------------------------------------------------------------------------

def init_tripwire(line_first: int, line_second: int, dist_m: float) -> dict:
    """
    line_first / line_second are X pixel positions (vertical lines).
    Vehicles are expected to cross line_first before line_second.
      Left-moving  → line_first > line_second  (high X first, then low X)
      Right-moving → line_first < line_second  (low X first, then high X)
    """
    return {
        "prev_x":        {},   # previous X position per track id
        "crossed_first": {},   # frame number when line_first was crossed
        "speed":         {},   # measured tripwire speed (km/h)
        "line_first":  line_first,
        "line_second": line_second,
        "dist_m":      dist_m,
    }


def _x_crossed(prev_x: float, cur_x: float, line_x: int) -> bool:
    """True if a vertical line at line_x was crossed between the two frames."""
    return (prev_x < line_x <= cur_x) or (cur_x < line_x <= prev_x)


def update_tripwire(tw: dict, tracked: dict, frame_id: int,
                    cam: str = "", direction_label: str = "") -> None:
    """Detect vertical-line crossings and compute speed when both lines are hit."""
    for tid, (cx, _cy) in tracked.items():
        prev = tw["prev_x"].get(tid)
        tw["prev_x"][tid] = cx
        if prev is None:
            continue

        if tid not in tw["crossed_first"]:
            if _x_crossed(prev, cx, tw["line_first"]):
                tw["crossed_first"][tid] = frame_id
                log.debug(
                    "[%s] id=%s%d crossed line-1 @ x=%d (frame=%d)",
                    cam, "L" if direction_label == LANE_LEFT_LABEL else "R",
                    tid, tw["line_first"], frame_id,
                )
        elif tid not in tw["speed"]:
            if _x_crossed(prev, cx, tw["line_second"]):
                elapsed = (frame_id - tw["crossed_first"][tid]) / FPS
                if elapsed > 0:
                    spd = (tw["dist_m"] / elapsed) * 3.6
                    tw["speed"][tid] = spd
                    log.info(
                        "[%s] 📍 TRIPWIRE  vehicle=%s%d  speed=%.1f km/h"
                        "  dir=%-25s  elapsed=%.2fs",
                        cam,
                        "L" if direction_label == LANE_LEFT_LABEL else "R",
                        tid, spd, direction_label, elapsed,
                    )


# ---------------------------------------------------------------------------
# Displacement-based fallback speed (X-axis pixel tracking)
# ---------------------------------------------------------------------------

def compute_displacement_speed(x_positions: list[float]) -> float | None:
    """
    Estimate speed from the X-pixel displacement over DISP_WINDOW frames.
    Returns km/h or None if the window is insufficient or speed is out of range.
    """
    if len(x_positions) < DISP_MIN_PTS:
        return None
    dx = abs(x_positions[-1] - x_positions[0])
    frames = len(x_positions) - 1
    px_per_frame = dx / frames
    spd = (px_per_frame / PIXELS_PER_METER) * FPS * 3.6
    return spd if DISP_SPD_MIN < spd < DISP_SPD_MAX else None


def detect_movement(x_positions: list[float]) -> str:
    """
    Determine if a vehicle is moving left, right, or can't be determined.

    Returns:
        "left"    – clearly moving toward lower X values
        "right"   – clearly moving toward higher X values
        "unknown" – insufficient data or near-stationary
    """
    if len(x_positions) < 4:
        return "unknown"
    dx = x_positions[-1] - x_positions[0]
    if dx > VEL_THRESHOLD_PX:
        return "right"
    if dx < -VEL_THRESHOLD_PX:
        return "left"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    alias      = _ENV_VIDEO
    video_path = alias

    # Resolve ipcamlive alias → live HLS URL
    if re.match(r"^cam\d+$", alias):
        log.info("Resolving ipcamlive alias '%s' …", alias)
        url = get_stream_url(alias)
        if url:
            log.info("HLS URL obtained: %s", url)
            video_path = url
        else:
            log.warning("Could not resolve alias '%s' — trying directly.", alias)

    # ── Print startup config ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Camera     : %s", CAMERA_NAME)
    log.info("Stream     : %s", video_path)
    log.info("Split X    : %d px  (left < %d = %s, right >= %d = %s)",
             SPLIT_COORD, SPLIT_COORD, LANE_LEFT_LABEL, SPLIT_COORD, LANE_RIGHT_LABEL)
    log.info("Tripwires (LEFT  %-25s): X=%d → X=%d  dist=%.1f m",
             LANE_LEFT_LABEL, LINE_L1, LINE_L2, DIST_M)
    log.info("Tripwires (RIGHT %-25s): X=%d → X=%d  dist=%.1f m",
             LANE_RIGHT_LABEL, LINE_R1, LINE_R2, DIST_M)
    log.info("px/m       : %.1f  (displacement fallback calibration)", PIXELS_PER_METER)
    log.info("Alert ≥    : %.0f km/h", ALERT_SPEED)
    log.info("DB         : %s", DB_PATH)
    log.info("=" * 70)

    conn = init_db(DB_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open stream: %s", video_path)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Stream opened: %dx%d", frame_w, frame_h)

    # Background subtractor + morphology kernels
    bg       = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)
    k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_large  = cv2.getStructuringElement(cv2.MORPH_RECT,    (15, 15))

    # Trackers — one per carriageway
    tr_left  = init_tracker()
    tr_right = init_tracker()

    # Vertical tripwires
    # Left-side  vehicles move  RIGHT→LEFT : LINE_L1 (higher X) crossed first
    # Right-side vehicles move  LEFT→RIGHT : LINE_R1 (lower  X) crossed first
    tw_left  = init_tripwire(LINE_L1, LINE_L2, DIST_M)
    tw_right = init_tripwire(LINE_R1, LINE_R2, DIST_M)

    seen_left:  set = set()
    seen_right: set = set()

    vehicle_type: dict = {}                                    # key → "car"|"truck"
    x_hist       = defaultdict(lambda: deque(maxlen=DISP_WINDOW))  # key → deque of X positions
    disp_speed:  dict = {}                                     # key → float km/h (fallback)

    saved_ids: set = set()   # (camera, vid) pairs already persisted this session
    frame_id  = 0

    log.info("Processing stream — press Ctrl+C to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("Stream read failed — reconnecting in 5 s …")
                cap.release()
                time.sleep(5)
                # Re-resolve URL (ipcamlive tokens expire)
                if re.match(r"^cam\d+$", alias):
                    new_url = get_stream_url(alias)
                    if new_url:
                        video_path = new_url
                        log.info("Re-resolved URL: %s", video_path)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    log.error("Reconnect failed. Waiting 30 s …")
                    time.sleep(30)
                continue

            frame_id += 1

            # ── Background subtraction ──────────────────────────────────────
            fg = bg.apply(frame)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_small)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_medium)
            fg = cv2.dilate(fg, k_large)

            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= MIN_AREA]

            # Split detections by carriageway (centre X vs SPLIT_COORD)
            rects_l = [r for r in all_rects if (r[0] + r[2] // 2) <  SPLIT_COORD]
            rects_r = [r for r in all_rects if (r[0] + r[2] // 2) >= SPLIT_COORD]

            # ── Update trackers ─────────────────────────────────────────────
            active_l = update_tracker(tr_left,  rects_l)
            active_r = update_tracker(tr_right, rects_r)

            # Log first appearance of each vehicle
            for tid in active_l:
                if tid not in seen_left:
                    seen_left.add(tid)
                    log.info("[%s] 🔵 NEW vehicle  id=L%d  side=%s  frame=%d",
                             CAMERA_NAME, tid, LANE_LEFT_LABEL, frame_id)
            for tid in active_r:
                if tid not in seen_right:
                    seen_right.add(tid)
                    log.info("[%s] 🔵 NEW vehicle  id=R%d  side=%s  frame=%d",
                             CAMERA_NAME, tid, LANE_RIGHT_LABEL, frame_id)

            # ── Tripwire speed detection ─────────────────────────────────────
            update_tripwire(tw_left,  active_l, frame_id,
                            cam=CAMERA_NAME, direction_label=LANE_LEFT_LABEL)
            update_tripwire(tw_right, active_r, frame_id,
                            cam=CAMERA_NAME, direction_label=LANE_RIGHT_LABEL)

            # ── X-displacement speed (fallback) + vehicle size classification
            for side, active, rects, side_label in [
                ("L", active_l, rects_l, LANE_LEFT_LABEL),
                ("R", active_r, rects_r, LANE_RIGHT_LABEL),
            ]:
                for tid, (cx, cy) in active.items():
                    key = (side, tid)
                    x_hist[key].append(float(cx))
                    pts = list(x_hist[key])

                    spd = compute_displacement_speed(pts)
                    if spd is not None:
                        prev = disp_speed.get(key)
                        disp_speed[key] = spd
                        if prev is None:  # log the first estimate only
                            log.info(
                                "[%s] 📡 DISP speed   vehicle=%s%d  speed=%.1f km/h  dir=%s",
                                CAMERA_NAME, side, tid, spd, side_label,
                            )

                    # Classify vehicle by bounding-box area
                    if rects:
                        rect = min(
                            rects,
                            key=lambda r: abs(cx - (r[0] + r[2] // 2))
                                        + abs(cy - (r[1] + r[3] // 2)),
                        )
                        vehicle_type[key] = (
                            "truck" if rect[2] * rect[3] >= TRUCK_MIN_AREA else "car"
                        )

            # ── Persist newly resolved speeds ────────────────────────────────
            for side, tw, side_label in [
                ("L", tw_left,  LANE_LEFT_LABEL),
                ("R", tw_right, LANE_RIGHT_LABEL),
            ]:
                seen_tids = (
                    set(tw["speed"])
                    | {tid for (s, tid) in vehicle_type if s == side}
                    | {tid for (s, tid) in disp_speed   if s == side}
                )
                for tid in seen_tids:
                    key    = (side, tid)
                    vid    = f"{side}{tid}"
                    db_key = (CAMERA_NAME, vid)
                    if db_key in saved_ids:
                        continue   # already saved this session

                    spd_tw   = tw["speed"].get(tid)
                    spd_disp = disp_speed.get(key)
                    if spd_tw is not None:
                        speed, source = spd_tw, "tripwire"
                    elif spd_disp is not None:
                        speed, source = spd_disp, "displacement"
                    else:
                        continue   # no speed estimate yet

                    vtype    = vehicle_type.get(key, "unknown")
                    movement = detect_movement(list(x_hist.get(key, [])))

                    # ── SQLite ─────────────────────────────────────────────
                    insert_event(conn, CAMERA_NAME, vid, vtype,
                                 speed, side_label, movement, source)
                    saved_ids.add(db_key)

                    log.info(
                        "[%s] ✅ SAVED  id=%-4s  type=%-5s  speed=%6.1f km/h"
                        "  dir=%-25s  movement=%-7s  src=%s",
                        CAMERA_NAME, vid, vtype, speed,
                        side_label, movement, source,
                    )

                    # ── Speed alert (> threshold) ───────────────────────────
                    if speed > ALERT_SPEED:
                        insert_alert(conn, CAMERA_NAME, vid, vtype,
                                     speed, side_label)

                    # ── Kafka ───────────────────────────────────────────────
                    if _KAFKA_ENABLED:
                        entry_frame = (tw["crossed_first"].get(tid, 0)
                                       if source == "tripwire" else 0)
                        try:
                            kafka_producer.publish_vehicle_event(
                                vehicle_id=vid,
                                vehicle_type=vtype,
                                speed_kmh=speed,
                                carriageway=side_label,
                                segment=CAMERA_NAME,
                                timestamp_s=entry_frame / FPS,
                                speed_source=source,
                            )
                        except Exception as exc:
                            log.error("Kafka publish error: %s", exc)

            # ── Periodic active-vehicle snapshot (every ~5 s) ──────────────
            if frame_id % LOG_EVERY_N_FRAMES == 0:
                n_active = len(active_l) + len(active_r)
                n_saved  = len(saved_ids)
                spd_l = "  ".join(
                    f"L{t}={v:.0f}" for t, v in tw_left["speed"].items()
                ) or "—"
                spd_r = "  ".join(
                    f"R{t}={v:.0f}" for t, v in tw_right["speed"].items()
                ) or "—"
                log.info(
                    "[%s] ⏱  frame=%d  active_l=%d/%s  active_r=%d/%s"
                    "  saved=%d  tripwire=[%s | %s]",
                    CAMERA_NAME, frame_id,
                    len(active_l), LANE_LEFT_LABEL[:12],
                    len(active_r), LANE_RIGHT_LABEL[:12],
                    n_saved, spd_l, spd_r,
                )

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        if _KAFKA_ENABLED:
            try:
                kafka_producer.flush()
                kafka_producer.close()
            except Exception:
                pass
        conn.close()
        log.info(
            "Session ended. Camera: %s | Frames processed: %d | Events saved: %d",
            CAMERA_NAME, frame_id, len(saved_ids),
        )


if __name__ == "__main__":
    main()
