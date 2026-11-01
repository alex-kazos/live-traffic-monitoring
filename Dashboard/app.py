"""
app.py — Live dashboard for the tracking instances.

For every camera listed in CAMERAS the dashboard shows:
  • the annotated live video (preview JPEGs written by monitor_traffic.py)
  • the tracking container's logs (read through the Docker socket)
  • the vehicle records from the shared SQLite database

Configuration env vars:
  CAMERAS          comma-separated CAMERA_NAME=container pairs, e.g.
                   "IC_Papagou=tracking-papagou,Roupaki_Toll_Station=tracking-roupaki"
  DB_PATH          SQLite file path                (default: /data/db/traffic.db)
  FRAME_DIR        preview frame directory         (default: /data/frames)
  ALERT_SPEED_KMH  speed highlighted as an alert   (default: 130)
  ANALYTICS_URL    target of the "Live analytics" link, e.g. http://localhost:8050
                   (default: no link)
"""

import asyncio
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import docker
import requests
from docker.errors import DockerException, NotFound
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH     = os.getenv("DB_PATH", "/data/db/traffic.db")
FRAME_DIR   = Path(os.getenv("FRAME_DIR", "/data/frames"))
ALERT_SPEED = float(os.getenv("ALERT_SPEED_KMH", "130"))
ANALYTICS   = os.getenv("ANALYTICS_URL", "")
STATIC_DIR  = Path(__file__).parent / "static"

LOG_TAIL      = 300     # lines sent when a log view first loads
LOG_MAX_LINES = 1000    # cap per poll if the browser fell far behind
EVENTS_LIMIT  = 100
RECORDS_LIMIT = 5000    # max rows per /records page


def parse_cameras(spec: str) -> dict[str, str]:
    """'NAME=container,NAME2=container2' → {NAME: container, ...}"""
    cameras = {}
    for item in spec.split(","):
        name, sep, container = item.partition("=")
        if sep and name.strip() and container.strip():
            cameras[name.strip()] = container.strip()
    return cameras


CAMERAS = parse_cameras(os.getenv("CAMERAS", ""))


# Browsers re-check the page and its scripts on every load (a cheap ETag match),
# so a rebuilt dashboard shows up on a plain reload instead of a stale cached copy
NO_CACHE = {"Cache-Control": "no-cache"}


class RevalidatedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


app = FastAPI(title="Traffic Monitoring Dashboard")
app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")


def container_for(camera: str) -> str:
    if camera not in CAMERAS:
        raise HTTPException(404, f"Unknown camera '{camera}'")
    return CAMERAS[camera]


# ── Docker ────────────────────────────────────────────────────────────────────
_docker = None


def get_container(name: str):
    """Look up a container by name; raises NotFound / DockerException."""
    global _docker
    if _docker is None:
        _docker = docker.from_env()
    return _docker.containers.get(name)


def container_state(name: str) -> dict:
    try:
        c = get_container(name)
    except NotFound:
        return {"status": "not found"}
    except (DockerException, requests.RequestException) as exc:
        return {"status": "unavailable", "error": str(exc)}
    state = c.attrs["State"]
    return {
        "status":     state["Status"],
        "started_at": state["StartedAt"],
        "restarts":   c.attrs.get("RestartCount", 0),
    }


def _ts_key(ts: str) -> str:
    """Docker trims trailing zeros off RFC3339Nano timestamps; pad so they sort as text."""
    base, _, frac = ts.rstrip("Z").partition(".")
    return f"{base}.{frac.ljust(9, '0')}"


# ── SQLite (read-only) ────────────────────────────────────────────────────────

def query(sql: str, params: dict) -> list[dict]:
    """Run a read-only query; empty result until the tracking service creates the DB."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise
    finally:
        conn.close()


# ── Live video ────────────────────────────────────────────────────────────────

def frame_path(camera: str) -> Path:
    # Same naming rule as FRAME_PATH in monitor_traffic.py
    return FRAME_DIR / (re.sub(r"[^\w.-]", "_", camera) + ".jpg")


def frame_age(camera: str) -> float | None:
    try:
        return round(time.time() - frame_path(camera).stat().st_mtime, 1)
    except FileNotFoundError:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/cameras")
def cameras():
    return {
        "alert_speed_kmh": ALERT_SPEED,
        "analytics_url":   ANALYTICS,
        "cameras": [{"name": n, "container": c} for n, c in CAMERAS.items()],
    }


@app.get("/api/cameras/{camera}/summary")
def summary(camera: str):
    """Container state, preview freshness and record counters for one camera."""
    container = container_for(camera)
    params = {
        "camera": camera,
        "alert":  ALERT_SPEED,
        "since":  (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    }
    totals = query("""
        SELECT COUNT(*)                     AS total,
               SUM(recorded_at >= :since)   AS last_5_min,
               ROUND(AVG(speed_kmh), 1)     AS avg_speed_kmh,
               SUM(speed_kmh > :alert)      AS over_alert
        FROM vehicle_events
        WHERE camera_name = :camera
    """, params)
    directions = query("""
        SELECT direction,
               COUNT(*)                 AS total,
               ROUND(AVG(speed_kmh), 1) AS avg_speed_kmh
        FROM vehicle_events
        WHERE camera_name = :camera
        GROUP BY direction
        ORDER BY direction
    """, params)
    return {
        "container":   container_state(container),
        "frame_age_s": frame_age(camera),
        "totals":      totals[0] if totals else {},
        "directions":  directions,
    }


@app.get("/api/cameras/{camera}/events")
def events(camera: str, after_id: int = 0):
    """Newest vehicle records with id > after_id (newest first)."""
    container_for(camera)
    rows = query("""
        SELECT id, vehicle_id, vehicle_type, speed_kmh, direction,
               movement, speed_source, recorded_at
        FROM vehicle_events
        WHERE camera_name = :camera AND id > :after_id
        ORDER BY id DESC
        LIMIT :limit
    """, {"camera": camera, "after_id": after_id, "limit": EVENTS_LIMIT})
    return {"events": rows}


@app.get("/api/cameras/{camera}/records")
def records(camera: str, since: str = "", after_id: int = 0, limit: int = RECORDS_LIMIT):
    """
    Vehicle records with id > after_id recorded at or after `since` (ISO-8601 UTC),
    oldest first — for clients that keep their own copy (LocalDashboard/dashboard.py).
    A full page means there may be more: ask again with the last id as after_id.
    """
    container_for(camera)
    rows = query("""
        SELECT id, vehicle_id, vehicle_type, speed_kmh, direction,
               movement, speed_source, recorded_at
        FROM vehicle_events
        WHERE camera_name = :camera AND id > :after_id AND recorded_at >= :since
        ORDER BY id
        LIMIT :limit
    """, {"camera": camera, "after_id": after_id, "since": since,
          "limit": max(1, min(limit, RECORDS_LIMIT))})
    return {"records": rows}


@app.get("/api/cameras/{camera}/logs")
def logs(camera: str, since: str = ""):
    """
    Container log lines newer than `since` (the `last` value of the previous
    call), or the last LOG_TAIL lines when `since` is empty.
    """
    name = container_for(camera)
    try:
        container = get_container(name)
        if since:
            # Docker filters by whole seconds; drop the already-seen lines below
            start = datetime.strptime(since[:19], "%Y-%m-%dT%H:%M:%S")
            raw = container.logs(timestamps=True,
                                 since=int(start.replace(tzinfo=timezone.utc).timestamp()))
        else:
            raw = container.logs(timestamps=True, tail=LOG_TAIL)
    except NotFound:
        return {"lines": [], "last": since, "error": f"Container '{name}' is not running."}
    except (DockerException, requests.RequestException) as exc:
        return {"lines": [], "last": since, "error": f"Docker unavailable: {exc}"}

    since_key = _ts_key(since) if since else ""
    lines, last = [], since
    for line in raw.decode("utf-8", errors="replace").splitlines():
        ts, _, text = line.partition(" ")
        if since and _ts_key(ts) <= since_key:
            continue
        lines.append(text)
        last = ts
    return {"lines": lines[-LOG_MAX_LINES:], "last": last}


@app.get("/api/cameras/{camera}/video.mjpg")
async def video(camera: str):
    """MJPEG stream: pushes the preview JPEG every time the tracker replaces it."""
    container_for(camera)
    path = frame_path(camera)

    async def frames():
        last_mtime = None
        while True:  # ends when the client disconnects
            try:
                mtime = path.stat().st_mtime_ns
                if mtime != last_mtime:
                    data = path.read_bytes()
                    last_mtime = mtime
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                           + data + b"\r\n")
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})
