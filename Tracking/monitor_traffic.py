"""
monitor_traffic.py — Real-time traffic tracking from ipcamlive livestreams.

The naodos.gr / Attiki Odos cameras are mounted above the motorway and look
ALONG it, so vehicles move toward or away from the camera:

  • I/C Papagou (cam231) : left carriageway = to Katechaki | right = to Mesogeia-Vrilissia
  • Roupaki Toll (cam88)  : left carriageway = to Elefsina  | right = to Athens Airport

Each carriageway has a measuring zone (an image polygon) and a ground-plane
calibration that maps image pixels to road metres: X across the road, Y along
it, increasing away from the camera.  Speed comes from two TRIPWIRES at known
Y positions — crossing times are interpolated between frames.  Vehicles that
miss a tripwire fall back to a least-squares fit of their road position over
time.

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
  RETENTION_DAYS     delete this camera's records older than this (default: 7, 0 = keep all)

  --- Carriageway geometry (_L = left carriageway in the image, _R = right) ---
  ROI_L / ROI_R              measuring zone polygon "x,y;x,y;..." in pixels   (required)
  CALIB_L / CALIB_R          4 ground points "x,y=X,Y;..." pixels → metres    (required)
  TRIPWIRES_L / TRIPWIRES_R  two road positions "Y1,Y2" in metres            (required)
  CALIB_WIDTH                frame width the pixel coordinates refer to      (default: 1920)
  PROCESS_WIDTH              frames are downscaled to this width for detection (default: 960)

  --- Lane Labels ---
  LANE_LEFT_LABEL    e.g. "to_Elefsina"          (default: left_lane)
  LANE_RIGHT_LABEL   e.g. "to_Athens_Airport"    (default: right_lane)

  --- Live Preview (read by the dashboard service) ---
  FRAME_DIR          directory for the annotated preview JPEG   (default: disabled)
  PREVIEW_FPS        max preview frames written per second      (default: 5)
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
from datetime import datetime, timedelta, timezone

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

# Records older than RETENTION_DAYS are deleted every PRUNE_EVERY_S; 0 keeps all
RETENTION_DAYS = float(os.getenv("RETENTION_DAYS", "7"))
PRUNE_EVERY_S  = 3600
PRUNE_BATCH    = 5000     # rows per delete, so the write lock stays short

if not _ENV_VIDEO:
    log.error("INPUT_VIDEO environment variable is not set. Exiting.")
    sys.exit(1)

# Human-readable direction labels for each carriageway
LANE_LEFT_LABEL  = os.getenv("LANE_LEFT_LABEL",  "left_lane")
LANE_RIGHT_LABEL = os.getenv("LANE_RIGHT_LABEL", "right_lane")

# Calibration pixels are measured on a CALIB_WIDTH-wide frame; detection runs
# on frames downscaled to PROCESS_WIDTH (≈4× cheaper than full HD)
CALIB_WIDTH   = int(os.getenv("CALIB_WIDTH", "1920"))
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", "960"))
SCALE         = PROCESS_WIDTH / CALIB_WIDTH

# ---------------------------------------------------------------------------
# Live preview — annotated JPEG overwritten a few times per second
# ---------------------------------------------------------------------------
FRAME_DIR   = os.getenv("FRAME_DIR", "")                # empty = preview disabled
PREVIEW_FPS = float(os.getenv("PREVIEW_FPS", "5"))
# The dashboard derives the same file name from CAMERA_NAME
FRAME_PATH  = (os.path.join(FRAME_DIR, re.sub(r"[^\w.-]", "_", CAMERA_NAME) + ".jpg")
               if FRAME_DIR and PREVIEW_FPS > 0 else "")

# ---------------------------------------------------------------------------
# Tracking constants
# ---------------------------------------------------------------------------
FPS               = 25.0     # used only if the stream has no timestamps
MIN_AREA_M2       = 5.0      # apparent ground area (m²) below which a blob is noise
# A vehicle's apparent area depends on the view angle, its distance and road spray,
# so sizes are judged against the running median of the same zone (mostly cars):
FRAGMENT_FACTOR   = 0.4      # below 0.4 × median: part of a vehicle / spray — not saved
TRUCK_FACTOR      = 1.8      # above 1.8 × median: truck
SIZE_REF_LEN      = 200      # vehicles in the running median
MAX_LOST          = 12       # frames before a track is dropped
MAX_DIST          = 180 * PROCESS_WIDTH / 1280   # max pixel distance for track matching
EDGE_MARGIN_PX    = 2        # boxes this close to the frame edge are cut off — skip them

HIST_LEN        = 50         # road positions kept per track (~2 s)
DISP_MIN_PTS    = 8          # fallback speed needs at least this many positions…
DISP_MIN_SPAN_S = 0.4        # …spanning at least this many seconds…
DISP_MIN_TRAVEL = 10.0       # …and at least this many metres travelled
DISP_MAX_RESID  = 1.5        # m — a noisier fit means the ground point jumped (blob merge)
TRIPWIRE_TOL    = 0.35       # reject a tripwire speed this far (±35 %) from the fitted speed
SPD_MIN         = 5.0        # km/h — ignore slower readings (noise)
SPD_MAX         = 200.0      # km/h — discard unreasonably high values
MOVE_MIN_M      = 1.0        # min |ΔY| (m) to call a direction of travel

LOG_EVERY_N_FRAMES = 125     # ~5 s at 25 fps

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
# Carriageway geometry — measuring zone, pixel → road-metre mapping, tripwires
# ---------------------------------------------------------------------------

def _points(spec: str) -> np.ndarray:
    """'x,y;x,y;...' → float32 array of shape (n, 2)."""
    return np.float32([[float(v) for v in p.split(",")] for p in spec.split(";") if p.strip()])


class Carriageway:
    """One side of the road.  All pixel coordinates are in processing resolution."""

    def __init__(self, side: str, label: str, roi: str, calib: str, tripwires: str):
        self.side  = side
        self.label = label
        self.roi   = _points(roi) * SCALE

        pairs = [p.split("=") for p in calib.split(";") if p.strip()]
        if len(pairs) != 4 or any(len(p) != 2 for p in pairs):
            raise ValueError("CALIB needs exactly 4 'x,y=X,Y' pairs")
        img_pts  = _points(";".join(p[0] for p in pairs)) * SCALE
        road_pts = _points(";".join(p[1] for p in pairs))
        self.H     = cv2.getPerspectiveTransform(img_pts, road_pts)
        self.H_inv = np.linalg.inv(self.H)

        self.wires = tuple(float(v) for v in tripwires.split(","))
        if len(self.wires) != 2:
            raise ValueError("TRIPWIRES needs two values 'Y1,Y2'")
        # Tripwires drawn across the calibrated width of the road
        x0, x1 = float(road_pts[:, 0].min()), float(road_pts[:, 0].max())
        self.wire_segments = [self.to_image([(x0, y), (x1, y)]) for y in self.wires]

        self.sizes = deque(maxlen=SIZE_REF_LEN)   # apparent areas of recent vehicles

    def contains(self, pt: tuple) -> bool:
        return cv2.pointPolygonTest(self.roi, pt, False) >= 0

    def to_road(self, pts) -> np.ndarray:
        """Image points (n, 2) → road metres (n, 2)."""
        return cv2.perspectiveTransform(np.float64(pts).reshape(1, -1, 2), self.H)[0]

    def to_image(self, pts) -> np.ndarray:
        """Road metres (n, 2) → image points (n, 2)."""
        return cv2.perspectiveTransform(np.float64(pts).reshape(1, -1, 2), self.H_inv)[0]

    def area_m2(self, x: int, y: int, w: int, h: int) -> float:
        """Apparent ground area of a bounding box, using the local scale at its base."""
        bx, by = x + w / 2, y + h
        p0, px, py = self.to_road([(bx, by), (bx + 1, by), (bx, by + 1)])
        (ax, ay), (cx, cy) = px - p0, py - p0
        return w * h * abs(ax * cy - ay * cx)


def _load_carriageway(side: str, label: str) -> Carriageway:
    names  = [f"ROI_{side}", f"CALIB_{side}", f"TRIPWIRES_{side}"]
    values = [os.getenv(n, "") for n in names]
    missing = [n for n, v in zip(names, values) if not v]
    if missing:
        log.error("Calibration env vars not set: %s. Exiting.", ", ".join(missing))
        sys.exit(1)
    try:
        return Carriageway(side, label, *values)
    except (ValueError, cv2.error) as exc:
        log.error("Invalid calibration for carriageway %s: %s. Exiting.", side, exc)
        sys.exit(1)


CARRIAGEWAYS = [_load_carriageway("L", LANE_LEFT_LABEL),
                _load_carriageway("R", LANE_RIGHT_LABEL)]


# ---------------------------------------------------------------------------
# SQLite — schema & helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Create / open the SQLite DB and ensure the schema exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    # speed_alerts belongs to the alerts consumer (Spark/alerts_consumer.py)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vehicle_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name   TEXT    NOT NULL,
            vehicle_id    TEXT    NOT NULL,
            vehicle_type  TEXT    NOT NULL,
            speed_kmh     REAL,
            direction     TEXT    NOT NULL,   -- named road direction (e.g. "to_Elefsina")
            movement      TEXT,               -- detected movement: "away"|"toward"|"unknown"
            speed_source  TEXT    NOT NULL,   -- "tripwire"|"displacement"
            recorded_at   TEXT    NOT NULL    -- ISO-8601 UTC
        );

        CREATE INDEX IF NOT EXISTS idx_ve_cam    ON vehicle_events (camera_name);
        CREATE INDEX IF NOT EXISTS idx_ve_speed  ON vehicle_events (speed_kmh);
        CREATE INDEX IF NOT EXISTS idx_ve_dir    ON vehicle_events (direction);
        CREATE INDEX IF NOT EXISTS idx_ve_time   ON vehicle_events (camera_name, recorded_at);
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


def prune_events(conn: sqlite3.Connection, camera: str, days: float) -> int:
    """Delete this camera's records older than `days`; returns the number removed."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    removed = 0
    while True:
        # Batches keep each write short, so the other tracker isn't locked out
        n = conn.execute(
            """DELETE FROM vehicle_events WHERE id IN (
                   SELECT id FROM vehicle_events
                   WHERE camera_name = ? AND recorded_at < ?
                   LIMIT ?)""",
            (camera, cutoff, PRUNE_BATCH),
        ).rowcount
        conn.commit()
        removed += n
        if n < PRUNE_BATCH:
            return removed


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def init_tracker() -> dict:
    return {"next_id": 0, "centroids": {}, "lost": defaultdict(int)}


def update_tracker(tracker: dict, new_pts: list) -> dict:
    """
    Match detection points to existing tracks (nearest point first).
    Tracks unmatched for more than MAX_LOST frames are dropped; the returned
    dict still holds tracks that were missed this frame (at their last point).
    """
    centroids = tracker["centroids"]
    lost      = tracker["lost"]

    if not new_pts:
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


def new_track() -> dict:
    return {
        "hist":   deque(maxlen=HIST_LEN),   # (t, X, Y) road positions
        "areas":  deque(maxlen=HIST_LEN),   # apparent ground areas (m²)
        "cross":  {},                       # tripwire index → crossing time (s)
        "speed":  None,                     # tripwire speed (km/h)
        "saved":  False,
    }


# ---------------------------------------------------------------------------
# Speed estimation (road metres)
# ---------------------------------------------------------------------------

def _crossed(prev: float, cur: float, line: float) -> bool:
    """True if the road position `line` was passed between the two samples."""
    return (prev < line <= cur) or (cur < line <= prev)


def update_crossings(track: dict, cw: Carriageway, t: float, y: float) -> None:
    """
    Record interpolated tripwire crossing times; set the speed once both are
    crossed.  The result is kept only if a clean fit of the same track agrees
    with it — blob merges and ID switches make the ground point jump.
    """
    if not track["hist"]:
        return
    t0, _, y0 = track["hist"][-1]
    for i, line in enumerate(cw.wires):
        if i not in track["cross"] and _crossed(y0, y, line):
            track["cross"][i] = t0 + (line - y0) / (y - y0) * (t - t0)
    if len(track["cross"]) == 2 and track["speed"] is None:
        elapsed = abs(track["cross"][1] - track["cross"][0])
        if elapsed > 0:
            spd = abs(cw.wires[1] - cw.wires[0]) / elapsed * 3.6
            fitted = fallback_speed(track["hist"], min_travel=0)
            if fitted is not None and abs(spd - fitted) <= TRIPWIRE_TOL * fitted:
                track["speed"] = spd


def fallback_speed(hist, min_travel: float = DISP_MIN_TRAVEL) -> float | None:
    """Least-squares speed over the track's recent road positions (km/h)."""
    if len(hist) < DISP_MIN_PTS:
        return None
    t, x, y = np.array(hist).T
    if t[-1] - t[0] < DISP_MIN_SPAN_S or abs(y[-1] - y[0]) < min_travel:
        return None
    vy, y0 = np.polyfit(t, y, 1)
    if np.sqrt(np.mean((y - (vy * t + y0)) ** 2)) > DISP_MAX_RESID:
        return None
    vx = np.polyfit(t, x, 1)[0]
    spd = float(np.hypot(vx, vy)) * 3.6
    return spd if SPD_MIN < spd < SPD_MAX else None


def detect_movement(hist) -> str:
    """'away' / 'toward' the camera, or 'unknown' if the track barely moved."""
    if len(hist) < 2:
        return "unknown"
    dy = hist[-1][2] - hist[0][2]
    if dy > MOVE_MIN_M:
        return "away"
    if dy < -MOVE_MIN_M:
        return "toward"
    return "unknown"


# ---------------------------------------------------------------------------
# Live preview helpers
# ---------------------------------------------------------------------------
PREVIEW_MAX_W = 1280      # previews are downscaled to this width

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_GREEN  = (80, 220, 80)     # BGR
_RED    = (60, 60, 255)
_YELLOW = (0, 210, 255)
_GREY   = (170, 170, 170)
_WHITE  = (255, 255, 255)
_ZONE   = {"L": (255, 200, 0), "R": (255, 120, 255)}


def _text(frame: np.ndarray, text: str, org: tuple, color: tuple, k: float) -> None:
    """Text sized for the frame (k = width / 1280) with a dark outline for contrast."""
    scale, thick = 0.7 * k, max(1, round(2 * k))
    org = (int(org[0]), int(org[1]))
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, rects: list, tracks: list,
                 n_saved: int, frame_id: int, fps: float) -> None:
    """
    Draw the measuring zones, tripwires, detections and tracked vehicles onto
    the full-resolution frame in place.  rects / tracks are in processing pixels;
    tracks: (label, x, y, speed_kmh | None).
    """
    h, w = frame.shape[:2]
    k = w / 1280                     # keeps the overlay the same size at any resolution
    f = w / PROCESS_WIDTH            # processing pixels → frame pixels
    line_w = max(1, round(2 * k))

    for cw in CARRIAGEWAYS:
        roi = np.int32(cw.roi * f)
        cv2.polylines(frame, [roi], True, _ZONE[cw.side], max(1, round(k)), cv2.LINE_AA)
        for seg in cw.wire_segments:
            a, b = np.int32(seg * f)
            cv2.line(frame, tuple(a), tuple(b), _YELLOW, line_w, cv2.LINE_AA)
        far = roi[np.argmin(roi[:, 1])]           # label the zone at its far end
        _text(frame, cw.label, (far[0], far[1] - 8 * k), _ZONE[cw.side], k)

    for x, y, bw, bh in rects:
        x, y, bw, bh = (round(v * f) for v in (x, y, bw, bh))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), _GREY, max(1, round(k)))

    for label, px, py, speed in tracks:
        color = _RED if speed is not None and speed > ALERT_SPEED else _GREEN
        if speed is not None:
            label = f"{label} {speed:.0f} km/h"
        cx, cy = round(px * f), round(py * f)
        cv2.circle(frame, (cx, cy), round(5 * k), color, -1)
        _text(frame, label, (cx + 8 * k, cy - 8 * k), color, k)

    cv2.rectangle(frame, (0, 0), (w, round(34 * k)), (0, 0, 0), -1)
    _text(frame,
          f"{CAMERA_NAME}   active {len(tracks)}   saved {n_saved}   frame {frame_id}"
          f"   {fps:.0f} fps   {datetime.now(timezone.utc):%H:%M:%S} UTC",
          (10 * k, 24 * k), _WHITE, k)


def write_preview(frame: np.ndarray, path: str) -> None:
    """Replace the preview JPEG atomically so readers never see a partial file."""
    h, w = frame.shape[:2]
    if w > PREVIEW_MAX_W:
        frame = cv2.resize(frame, (PREVIEW_MAX_W, round(h * PREVIEW_MAX_W / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Persistence of a finished measurement
# ---------------------------------------------------------------------------

def save_vehicle(conn: sqlite3.Connection, cw: Carriageway, tid: int, track: dict,
                 speed: float, source: str) -> bool:
    """Store one vehicle in SQLite and publish it to Kafka; False if it was a fragment."""
    vid  = f"{cw.side}{tid}"
    # Smallest readings are taken closest to the camera, with the least height distortion
    area = float(np.percentile(track["areas"], 25))
    cw.sizes.append(area)
    typical = float(np.median(cw.sizes))
    track["saved"] = True
    if area < FRAGMENT_FACTOR * typical:
        log.info("[%s] discarded %s: %.0fm² vs typical %.0fm² — fragment, not a vehicle",
                 CAMERA_NAME, vid, area, typical)
        return False

    vtype    = "truck" if area > TRUCK_FACTOR * typical else "car"
    movement = detect_movement(track["hist"])

    insert_event(conn, CAMERA_NAME, vid, vtype, speed, cw.label, movement, source)
    log.info(
        "[%s] ✅ SAVED  id=%-4s  type=%-5s  speed=%6.1f km/h"
        "  dir=%-25s  movement=%-7s  src=%-12s  area=%.0fm²",
        CAMERA_NAME, vid, vtype, speed, cw.label, movement, source, area,
    )
    if speed > ALERT_SPEED:
        log.warning(
            "[%s] ⚠  SPEED ALERT  %s %s  |  %.1f km/h  |  %s",
            CAMERA_NAME, vtype.upper(), vid, speed, cw.label,
        )

    if _KAFKA_ENABLED:
        # Stream time when the vehicle reached its first tripwire (or was first seen)
        entry_t = min(track["cross"].values()) if source == "tripwire" else track["hist"][0][0]
        try:
            kafka_producer.publish_vehicle_event(
                vehicle_id=vid,
                vehicle_type=vtype,
                speed_kmh=speed,
                carriageway=cw.label,
                segment=CAMERA_NAME,
                timestamp_s=entry_t,
                speed_source=source,
            )
        except Exception as exc:
            log.error("Kafka publish error: %s", exc)
    return True


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
    for cw in CARRIAGEWAYS:
        log.info("Zone %s     : %-25s tripwires at %.0f m and %.0f m (%.0f m apart)",
                 cw.side, cw.label, cw.wires[0], cw.wires[1],
                 abs(cw.wires[1] - cw.wires[0]))
    log.info("Processing : %d px wide (calibration measured at %d px)",
             PROCESS_WIDTH, CALIB_WIDTH)
    log.info("Alert ≥    : %.0f km/h", ALERT_SPEED)
    log.info("DB         : %s", DB_PATH)
    log.info("Retention  : %s", f"{RETENTION_DAYS:g} days" if RETENTION_DAYS > 0 else "keep all")
    log.info("Preview    : %s", f"{FRAME_PATH} ({PREVIEW_FPS:g} fps)" if FRAME_PATH else "disabled")
    log.info("=" * 70)

    conn = init_db(DB_PATH)
    if FRAME_PATH:
        os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open stream: %s", video_path)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    log.info("Stream opened: %dx%d @ %.1f fps", frame_w, frame_h, stream_fps)

    # Background subtractor + morphology kernels (sizes tuned at 1280 px width)
    def kernel(shape: int, size: int) -> np.ndarray:
        n = max(3, round(size * PROCESS_WIDTH / 1280) | 1)
        return cv2.getStructuringElement(shape, (n, n))

    bg       = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)
    k_small  = kernel(cv2.MORPH_ELLIPSE, 3)
    k_medium = kernel(cv2.MORPH_ELLIPSE, 11)
    k_large  = kernel(cv2.MORPH_RECT,    15)

    trackers = {cw.side: init_tracker() for cw in CARRIAGEWAYS}
    tracks: dict = {}          # (side, tid) → track state
    n_saved  = 0
    frame_id = 0

    preview_path = FRAME_PATH
    last_preview = 0.0
    proc_fps     = 0.0
    fps_t0, fps_f0 = time.monotonic(), 0
    last_prune   = float("-inf")   # prune once at startup

    log.info("Processing stream — press Ctrl+C to stop.\n")

    try:
        while True:
            # ── Retention: drop this camera's old records (also during outages)
            if RETENTION_DAYS > 0 and time.monotonic() - last_prune >= PRUNE_EVERY_S:
                last_prune = time.monotonic()
                removed = prune_events(conn, CAMERA_NAME, RETENTION_DAYS)
                if removed:
                    log.info("Retention: removed %d records older than %g days",
                             removed, RETENTION_DAYS)

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
                # Timestamps restart with the new stream — start tracking afresh
                trackers = {cw.side: init_tracker() for cw in CARRIAGEWAYS}
                tracks.clear()
                continue

            frame_id += 1
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000 if pos_ms > 0 else frame_id / stream_fps

            # ── Background subtraction (downscaled) ─────────────────────────
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (PROCESS_WIDTH, round(h * PROCESS_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
            small_h = small.shape[0]
            fg = bg.apply(small)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_small)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_medium)
            fg = cv2.dilate(fg, k_large)

            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_rects = [cv2.boundingRect(c) for c in cnts]
            zone_rects = []

            for cw in CARRIAGEWAYS:
                # ── Detections in this measuring zone (base-centre = ground point)
                dets = {}
                for x, y, bw, bh in all_rects:
                    # A box cut off by the frame edge has no reliable ground point
                    if (x <= EDGE_MARGIN_PX or x + bw >= PROCESS_WIDTH - EDGE_MARGIN_PX
                            or y + bh >= small_h - EDGE_MARGIN_PX):
                        continue
                    pt = (x + bw / 2, y + bh)
                    if cw.contains(pt):
                        area = cw.area_m2(x, y, bw, bh)
                        if area >= MIN_AREA_M2:
                            dets[pt] = area
                            zone_rects.append((x, y, bw, bh))

                active = update_tracker(trackers[cw.side], list(dets))

                for tid, pt in active.items():
                    key = (cw.side, tid)
                    track = tracks.get(key)
                    if track is None:
                        track = tracks[key] = new_track()
                        log.info("[%s] 🔵 NEW vehicle  id=%s%d  side=%s  frame=%d",
                                 CAMERA_NAME, cw.side, tid, cw.label, frame_id)
                    if pt not in dets:
                        continue   # missed this frame; the tracker keeps it alive

                    road_x, road_y = cw.to_road([pt])[0]
                    update_crossings(track, cw, t, road_y)
                    track["hist"].append((t, road_x, road_y))
                    track["areas"].append(dets[pt])

                    if track["speed"] is not None and not track["saved"]:
                        log.info("[%s] 📍 TRIPWIRE  vehicle=%s%d  speed=%.1f km/h  dir=%s",
                                 CAMERA_NAME, cw.side, tid, track["speed"], cw.label)
                        n_saved += save_vehicle(conn, cw, tid, track, track["speed"], "tripwire")

                # ── Tracks that just ended: fall back to the fitted speed ───
                for key in [k for k in tracks if k[0] == cw.side and k[1] not in active]:
                    track = tracks.pop(key)
                    if not track["saved"]:
                        spd = fallback_speed(track["hist"])
                        if spd is not None:
                            n_saved += save_vehicle(conn, cw, key[1], track, spd, "displacement")

            # ── Live preview frame for the dashboard ────────────────────────
            if preview_path and time.monotonic() - last_preview >= 1 / PREVIEW_FPS:
                last_preview = time.monotonic()
                shown = [
                    (f"{side}{tid}", *trackers[side]["centroids"][tid],
                     tr["speed"] or fallback_speed(tr["hist"]))
                    for (side, tid), tr in tracks.items() if tr["hist"]
                ]
                draw_overlay(frame, zone_rects, shown, n_saved, frame_id, proc_fps)
                try:
                    write_preview(frame, preview_path)
                except OSError as exc:
                    log.warning("Preview write failed (%s) — preview disabled.", exc)
                    preview_path = ""

            # ── Periodic snapshot (every ~5 s) ──────────────────────────────
            if frame_id % LOG_EVERY_N_FRAMES == 0:
                now = time.monotonic()
                proc_fps = (frame_id - fps_f0) / (now - fps_t0)
                fps_t0, fps_f0 = now, frame_id
                n_left  = sum(1 for side, _ in tracks if side == "L")
                n_right = len(tracks) - n_left
                log.info(
                    "[%s] ⏱  frame=%d  %.1f fps  active_l=%d  active_r=%d  saved=%d",
                    CAMERA_NAME, frame_id, proc_fps, n_left, n_right, n_saved,
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
            CAMERA_NAME, frame_id, n_saved,
        )


if __name__ == "__main__":
    main()
