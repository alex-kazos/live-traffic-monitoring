# Traffic Monitoring Pipeline

A real-time traffic monitoring system built with **Apache Kafka**, **Apache Spark Structured Streaming**, and **OpenCV**. Processes live HLS streams from the **Attiki Odos** highway cameras (naodos.gr) to detect vehicles, measure speeds, persist data to SQLite, and emit real-time alerts for speeds exceeding 130 km/h.

---

## Monitored Cameras

| Camera | ipcamlive Alias | Description |
|---|---|---|
| **I/C Papagou (Imittos Ring Road)** | `cam231` | Direction Katechaki / Mesogeia-Vrilissia |
| **Roupaki Toll Station** | `cam88` | Direction Elefsina / Athens Airport |

Stream URLs are resolved dynamically at runtime via the ipcamlive API.

Both cameras look along the motorway, so vehicles move toward or away from them. Each carriageway has a measuring zone and a ground-plane calibration (pixels → road metres) derived from the lane-marking pattern; speed is the time a vehicle takes between two tripwires a known distance apart. See [Notes/CALIBRATION.md](Notes/CALIBRATION.md) for the method and its assumptions.

---

## Pipeline overview

The default `docker-compose.yml` stack runs without Zookeeper and follows this flow:

1. Live HLS streams from `cam231` and `cam88` are consumed by the two tracking containers.
2. The tracking services publish vehicle events to Kafka topics named `traffic-stats` and `traffic-alerts`.
3. `spark-analytics` reads `traffic-stats` and answers Q1, Q2, Q3, Q5, and Q6.
4. `spark-alerts` reads `traffic-alerts`, prints real-time alerts, and stores them in SQLite.
5. `dashboard` shows each tracking instance's annotated live video, logs, and SQLite records.
6. `local-dashboard` charts those records live with Plotly, per camera.

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
| `dashboard` | Web UI on http://localhost:8090 — live video, logs, and records per camera |
| `local-dashboard` | Plotly analytics on http://localhost:8050 — live speed and count charts per camera and carriageway |
| `tunnel` | Optional public HTTPS URL for the dashboard via a Cloudflare Quick Tunnel (`--profile public` only) |
| `tunnel-domain` | Optional fixed URL on your own domain via a named Cloudflare Tunnel (`--profile domain` only) |

---

## SQLite Database

The shared SQLite database at `/data/db/traffic.db` (Docker volume `db-data`) holds two tables: the tracking services write `vehicle_events`, and the alerts consumer is the only writer of `speed_alerts`.

### `vehicle_events` table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Autoincrement primary key |
| `camera_name` | TEXT | `IC_Papagou` or `Roupaki_Toll_Station` |
| `vehicle_id` | TEXT | Track ID such as `L3` or `R12` |
| `vehicle_type` | TEXT | `car` or `truck` |
| `speed_kmh` | REAL | Measured speed in km/h |
| `direction` | TEXT | Road direction label, such as `to_Elefsina` |
| `movement` | TEXT | Direction of travel relative to the camera (`away`, `toward`, or `unknown`) |
| `speed_source` | TEXT | `tripwire` (timed between two road lines) or `displacement` (fitted, when a vehicle misses a tripwire) |
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

### Retention

Rows older than `RETENTION_DAYS` (default **7**, set to `0` to keep everything) are deleted by the service that writes them: each tracker prunes its own camera's `vehicle_events` once an hour, in small batches, and the alerts consumer prunes `speed_alerts` at startup and then at most hourly as new alerts arrive. SQLite reuses the freed pages, so the database file stops growing once it holds one retention window of data. To change the window, set it in a `.env` file next to `docker-compose.yml`:

```
RETENTION_DAYS=14
```

The other data is bounded as well: Kafka keeps topics for 24 hours (`KAFKA_LOG_RETENTION_HOURS`), each container's log is capped at 3 × 10 MB, and the preview volume holds only the newest frame per camera.

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
| Bonus | Perspective-corrected speed | `monitor_traffic.py` — ground-plane homography per carriageway ([calibration](Notes/CALIBRATION.md)) |

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

On Windows, `start_docker.bat` does the same in the background (`docker compose up -d --build`).

### Dashboard

Open **http://localhost:8090**. For each tracking instance it shows:

- **Live video** — the stream with the tracker's overlay: measuring zone per carriageway, tripwires (yellow), detections, and track IDs with speed (red above the alert threshold). The tracker writes an annotated JPEG to the `frames-data` volume up to `PREVIEW_FPS` times per second (default 5).
- **Records** — the newest `vehicle_events` rows, plus totals, last-5-minute count, and average speed per direction.
- **Logs** — the container's log output, read through the Docker socket.

To add a camera to the dashboard, add a `CAMERA_NAME=container` pair to `CAMERAS` on the `dashboard` service and give the new tracking service `FRAME_DIR` plus the `frames-data` volume.

### Live analytics (Plotly)

Open **http://localhost:8050**, or follow **Live analytics ↗** in the dashboard's header; **Live video ↗** leads back. The `local-dashboard` service (`LocalDashboard/dashboard.py`, a Plotly Dash app) starts with the rest of the stack. It reads the vehicle records from the dashboard service's API, keeps the last few hours in memory, pulls new records every 5 seconds and redraws:

- **One tab per camera**, with a panel per carriageway, plus an **All cameras** tab that puts the cameras side by side. The cameras come from the server's `CAMERAS` list, so a new camera shows up as a new tab.
- **Filters**: carriageway, vehicle type, time window (15 min to 6 h) and bin size (1, 5 or 15 min). A zoomed chart stays zoomed through the live updates.
- **Charts**: average speed per time bin by vehicle type (Q6), vehicles per bin (Q5), average speed per carriageway and vehicle type over the window, and every measured speed against the alert line. Tiles count the vehicles over the Q3 limits and over the alert speed, and a table lists the newest over-speed vehicles (Q4).

Chart times are Athens time (`TZ` on the service). Like the dashboard, it listens on `localhost` only; the tunnels publish just the dashboard, so visitors who come through a tunnel don't see the **Live analytics** link.

To chart another host's data, such as the hosted demo, run the script on your own machine instead. Use another port if the stack's copy already holds 8050:

```powershell
pip install -r LocalDashboard/requirements.txt
python LocalDashboard/dashboard.py --server https://traffic.alexkazos.com --port 8051
```

If that hostname is behind Cloudflare Access, the script can't use the emailed PIN. Create a service token under **Zero Trust → Access controls → Service credentials**, add a policy with the **Service Auth** action for it to the Access application, and set `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` before starting. If the demo host is stopped, start it from the start page first; until then the header shows the server as unreachable. The other options are `--history` (hours kept in memory, default 6) and `--refresh` (seconds, default 5); see `--help`.

### Share the dashboard online (Cloudflare Tunnel)

The pipeline itself stays on your machine. A `cloudflared` container opens an outbound connection to Cloudflare, and Cloudflare serves the dashboard on a public HTTPS address, so no port has to be opened. There are two opt-in ways to run it.

#### On your own subdomain (fixed URL)

Needs a domain whose DNS is on Cloudflare, such as `alexkazos.com`. Adding a subdomain leaves the domain's existing records alone.

1. In the Cloudflare dashboard, go to **Networking → Tunnels → Create a tunnel**, name it (for example `traffic-monitoring`), and choose **Docker**. Copy only the token, the long `eyJ…` string after `--token`; you don't need to run the command shown.
2. Put the token in `.env` next to `docker-compose.yml`. It is a secret, and `.env` is git-ignored:
   ```
   TUNNEL_TOKEN=eyJ...
   ```
3. Start the stack with the tunnel; the dashboard should show the tunnel as connected:
   ```powershell
   docker compose --profile domain up -d --build
   ```
4. On the tunnel's **Routes** tab, select **Add route → Published application**: subdomain `traffic`, domain `alexkazos.com`, **Service URL** `http://dashboard:8000`. Use the compose service name, not `localhost:8090`, because `cloudflared` runs in its own container. Cloudflare creates the DNS record, and the dashboard is live at `https://traffic.alexkazos.com`. Keep it to one level of subdomain; deeper names need a paid certificate.
5. Optional, to let only reviewers in: in **Zero Trust → Access controls → Applications**, create a **Self-hosted** application for `traffic.alexkazos.com` with an Allow policy for emails ending in `@aueb.gr` (plus your own address). New Zero Trust accounts sign users in with a Cloudflare account by default, so also add **One-time PIN** under **Integrations → Identity providers** so visitors can sign in with an emailed code. Without an Access application, anyone with the link can open the dashboard.

#### Without a domain (temporary URL)

The `tunnel` service uses a Quick Tunnel: no Cloudflare account is needed, and the address is a random `https://<words>.trycloudflare.com` that changes whenever the container restarts.

```powershell
docker compose --profile public up -d --build
docker logs tunnel    # look for the https://….trycloudflare.com URL
```

Visitors have to sign in with a one-time PIN emailed to an allowed address. By default that is any `@aueb.gr` address; to allow others, set `ALLOWED_MAIL` in `.env` (comma-separated, `*@domain` allows a whole domain):

```
ALLOWED_MAIL=*@aueb.gr,you@example.com
```

#### Things to know

- Either URL works only while the stack is running, so the host must stay on and online. For round-the-clock access, run the same `docker compose` command on a small Linux VPS (about 4 vCPU and 8 GB RAM for the two trackers, Kafka and Spark).
- Each viewer receives both MJPEG video streams (5 fps by default), which comes out of the host's upload bandwidth; lower `PREVIEW_FPS` if that is a problem. Cloudflare's free-plan terms expect video to be served through a paid product, so this suits a few reviewers, not a widely shared live feed.
- Both profiles are opt-in: a plain `docker compose up` still keeps the dashboard on `localhost` only.

### Start on demand (AWS EC2 + Cloudflare Worker)

To keep a cloud host cheap, it runs only while someone is looking at it. When the host is off, `traffic.alexkazos.com` shows a **Start demo** page instead of an error. Clicking it starts the EC2 instance, the page reloads into the dashboard after about 3–5 minutes, and the host powers itself off a few hours later. The pieces:

| File | Role |
|---|---|
| `deploy/aws/traffic-monitoring.service` | systemd unit: `docker compose --profile domain up -d` on every boot |
| `deploy/aws/traffic-autostop.{service,timer}` | Powers the host off `HOURS` after it starts; EC2 then stops the instance and its compute billing |
| `deploy/aws/install.sh` | Installs both units (`sudo deploy/aws/install.sh [HOURS]`, default 3) |
| `deploy/aws/iam-policy.json` | Least-privilege policy for the Worker: start this one instance, read instance state |
| `worker/demo-starter.js` | Cloudflare Worker on the dashboard hostname: passes requests through, or serves the start page while the tunnel is down |

1. **On the EC2 host**, from the repo, run `sudo deploy/aws/install.sh`. Keep the host up longer while you work on it with `sudo systemctl stop traffic-autostop.timer`. Check that the instance's **Shutdown behavior** is **Stop** (EC2 → Actions → Instance settings), which is the default.
2. **In AWS IAM**, create a user without console access and attach an inline policy from `deploy/aws/iam-policy.json`. Fill in the region, the account ID and the instance ID. Create an access key for it.
3. **In Cloudflare**, go to **Workers & Pages → Create → Worker**. Replace the code with `worker/demo-starter.js` and deploy it. Under **Settings → Variables and Secrets**, add the variables `AWS_REGION` and `INSTANCE_ID` and the secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Under **Domains & Routes**, add the route `traffic.alexkazos.com/*`.
4. **Recommended:** put a Cloudflare Access application on `traffic.alexkazos.com` (see above), so only allowed visitors can start the host.

The instance gets a new public IP on every start. The tunnel doesn't care, but SSH needs the new address; the DNS name stays `traffic.alexkazos.com`.

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
├── Dashboard/
│   ├── Dockerfile
│   ├── app.py                           # FastAPI: MJPEG video, logs, records API
│   └── static/                          # index.html, app.js, style.css
├── LocalDashboard/
│   ├── Dockerfile
│   ├── dashboard.py                     # Plotly Dash: live analytics per camera
│   ├── assets/style.css
│   └── requirements.txt
├── Other/
│   ├── monitor_traffic_supervision.py   # optional YOLOv8 + ByteTrack variant
│   ├── get-model.bat                    # downloads the optional model weight
│   └── requirements.txt                 # optional experiment dependencies
├── Split/                               # legacy split-video utilities; not used by docker compose
└── start_docker.bat                     # Windows launcher: docker compose up -d --build
```
