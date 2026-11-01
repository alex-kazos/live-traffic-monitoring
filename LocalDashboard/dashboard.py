"""
dashboard.py — Live Plotly dashboard of the vehicle records, run on your own machine.

Connects to the dashboard service's API (Dashboard/app.py), keeps the last few
hours of vehicle records in memory, pulls new ones every few seconds, and charts
them per camera: average speed and vehicle count per time bin, average speed per
carriageway and vehicle type, every measured speed, and the over-speed alerts.

docker compose starts it with the rest of the stack (service local-dashboard).
To run it on its own, for example against the hosted server:

    pip install -r LocalDashboard/requirements.txt
    python LocalDashboard/dashboard.py                                   # local stack
    python LocalDashboard/dashboard.py --server https://traffic.alexkazos.com

then open http://localhost:8050.

Options (or env vars):
  --server   SERVER_URL     dashboard service to read from  (default: http://localhost:8090)
  --live-url LIVE_URL       target of the "Live video" link (default: the --server URL)
  --history  HISTORY_HOURS  hours of records kept in memory (default: 6)
  --refresh  REFRESH_S      seconds between updates         (default: 5)
  --host     HOST           address to listen on            (default: 127.0.0.1)
  --port     PORT           port this dashboard listens on  (default: 8050)
  TZ                        time zone of the chart times    (default: the machine's)

  CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET
      Cloudflare Access service token, for a server behind Cloudflare Access
"""

import argparse
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

# ── Configuration ──────────────────────────────────────────────────────────────
PAGE_SIZE   = 5000                      # records per API request (server caps it too)
WINDOWS_MIN = [15, 30, 60, 180, 360]    # time-window choices
BINS_MIN    = [1, 5, 15]                # bin-size choices
NICE_BINS   = [1, 2, 5, 10, 15, 30, 60]
MAX_BINS    = 72                        # bins get coarser past this, so bars stay readable
LIMITS_KMH  = {"car": 90, "truck": 80}  # Q3 speed limits per vehicle type
ALERT_ROWS  = 25
ALL         = "__all__"                 # camera selector value for the overview

COLUMNS = ["id", "vehicle_id", "vehicle_type", "speed_kmh", "direction",
           "movement", "speed_source", "recorded_at"]

# ── Look ──────────────────────────────────────────────────────────────────────
# Vehicle-type colors are categorical slots checked for CVD separation on the
# dark surface; CRITICAL is a status color for over-alert speeds, never a series
TYPE_COLORS = {"car": "#3987e5", "truck": "#d95926"}
OTHER_TYPE  = "#199e70"
CRITICAL    = "#f85149"
TEXT, MUTED = "#e6edf3", "#8b949e"
GRID, AXIS  = "#222a33", "#30363d"
PANEL_2     = "#1c222b"

pio.templates["traffic"] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
              size=12, color=MUTED),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=list(TYPE_COLORS.values()) + [OTHER_TYPE],
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, showgrid=False,
               ticks="outside", tickcolor=AXIS, title_standoff=8),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, rangemode="tozero",
               title_standoff=8),
    # Legend along the top edge, above the panel titles
    legend=dict(orientation="h", x=0, xanchor="left", y=1, yref="container", yanchor="top",
                font_color=TEXT),
    hoverlabel=dict(bgcolor=PANEL_2, bordercolor=AXIS, font_color=TEXT),
    margin=dict(l=56, r=24, t=64, b=40),
    bargap=0.2, bargroupgap=0.08, barcornerradius=3,
))
TEMPLATE = "plotly_dark+traffic"
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

log = logging.getLogger("local_dashboard")


# ── Helpers ───────────────────────────────────────────────────────────────────

def local_tz():
    """Current UTC offset of the local time zone; looked up per call, so DST changes apply."""
    return datetime.now().astimezone().tzinfo


def pretty(name: str) -> str:
    return name.replace("_", " ")


def minutes_label(minutes: int) -> str:
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h"


def ago(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    return f"{s // 3600}h {s % 3600 // 60:02d}m ago"


def describe(exc: Exception) -> str:
    """Short reason for a failed sync, shown in the header."""
    if isinstance(exc, requests.JSONDecodeError):
        return "got a web page instead of data (a sign-in page? see CF_ACCESS_* in --help)"
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code
        return f"HTTP {code}" + (" (host is down)" if code in (502, 503, 504, 530) else "")
    if isinstance(exc, requests.ConnectionError):
        return "can't connect"
    if isinstance(exc, requests.Timeout):
        return "timed out"
    return str(exc)


def to_frame(rows: list[dict], camera: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["camera"] = camera
    df["time_utc"] = pd.to_datetime(df["recorded_at"], utc=True, format="ISO8601")
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"])
    return df.drop(columns="recorded_at")


# ── Records ───────────────────────────────────────────────────────────────────

class RecordStore:
    """
    Copy of the server's vehicle records for the last `history`, one DataFrame
    per camera, topped up from /api/cameras/{camera}/records by a background thread.
    """

    def __init__(self, server: str, history: timedelta, headers: dict):
        self.server = server.rstrip("/")
        self.history = history
        self.http = requests.Session()
        self.http.headers.update(headers)
        self.last_id: dict[str, int] = {}   # touched by the sync thread only

        # Swapped as a whole under the lock; the Dash callbacks read a snapshot
        self.lock = threading.Lock()
        self.cameras: list[str] = []
        self.alert_speed = 130.0
        self.frames: dict[str, pd.DataFrame] = {}
        self.synced_at: datetime | None = None
        self.error: str | None = None

    def run(self, every_s: float) -> None:
        while True:
            self.sync()
            time.sleep(every_s)

    def sync(self) -> None:
        since = datetime.now(timezone.utc) - self.history
        try:
            config = self._get("/api/cameras")
            cameras = [c["name"] for c in config["cameras"]]
            frames = {}
            for camera in cameras:
                df = self.frames.get(camera)
                rows = self._fetch(camera, since)
                if rows:
                    new = to_frame(rows, camera)
                    df = new if df is None else pd.concat([df, new], ignore_index=True)
                    self.last_id[camera] = rows[-1]["id"]
                if df is not None:
                    frames[camera] = df[df["time_utc"] >= since]
        except (requests.RequestException, ValueError, KeyError) as exc:
            with self.lock:
                self.error = describe(exc)
            log.warning("Sync with %s failed: %s", self.server, self.error)
            return
        with self.lock:
            self.cameras = cameras
            self.alert_speed = float(config["alert_speed_kmh"])
            self.frames = frames
            self.synced_at = datetime.now(timezone.utc)
            self.error = None

    def _get(self, path: str, **params) -> dict:
        res = self.http.get(self.server + path, params=params, timeout=15)
        res.raise_for_status()
        return res.json()

    def _fetch(self, camera: str, since: datetime) -> list[dict]:
        """Records newer than the ones already held for `camera`, page by page."""
        rows, after = [], self.last_id.get(camera, 0)
        while True:
            page = self._get(f"/api/cameras/{quote(camera, safe='')}/records",
                             since=since.isoformat(), after_id=after,
                             limit=PAGE_SIZE)["records"]
            rows += page
            if len(page) < PAGE_SIZE:
                return rows
            after = page[-1]["id"]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "cameras":   list(self.cameras),
                "frames":    dict(self.frames),
                "alert":     self.alert_speed,
                "synced_at": self.synced_at,
                "error":     self.error,
            }


# ── Figures ───────────────────────────────────────────────────────────────────

def empty_figure(message: str, height: int) -> go.Figure:
    fig = go.Figure(layout=dict(template=TEMPLATE, height=height,
                                xaxis_visible=False, yaxis_visible=False))
    fig.add_annotation(text=message, showarrow=False, font_color=MUTED,
                       xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def finish(fig: go.Figure, height: int, uirevision: str, time_axis: bool = True) -> go.Figure:
    """Shared layout; uirevision keeps zoom and legend clicks across live updates."""
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1],
                                               font=dict(color=TEXT, size=13)))
    fig.update_layout(template=TEMPLATE, height=height, uirevision=uirevision,
                      legend_title_text="")
    if time_axis:
        fig.update_xaxes(tickformat="%H:%M", nticks=8)
    return fig


def bin_frame(df: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    start = df["time"].dt.floor(f"{bin_min}min")
    end = start + pd.Timedelta(minutes=bin_min)
    return df.assign(bin=start, span=start.dt.strftime("%H:%M") + "–" + end.dt.strftime("%H:%M"))


def speed_per_bin(df, bin_min, facets, colors, uirev) -> go.Figure:
    """Average speed per time bin by vehicle type, one panel per carriageway/camera."""
    g = (bin_frame(df, bin_min)
         .groupby(["facet", "bin", "span", "vehicle_type"], as_index=False)
         .agg(avg=("speed_kmh", "mean"), n=("id", "size")))
    fig = px.bar(g, x="bin", y="avg", color="vehicle_type", facet_col="facet",
                 barmode="group", custom_data=["span", "n"],
                 category_orders={"facet": facets}, color_discrete_map=colors,
                 labels={"avg": "Average speed (km/h)", "bin": "", "vehicle_type": "Vehicle type"})
    fig.update_traces(xperiod=bin_min * 60_000, xperiodalignment="middle",
                      hovertemplate="%{customdata[0]}<br><b>%{y:.1f} km/h</b> average"
                                    " · %{customdata[1]} vehicles<extra>%{fullData.name}</extra>")
    return finish(fig, 340, uirev)


def count_per_bin(df, bin_min, facets, colors, uirev) -> go.Figure:
    """Vehicles per time bin, stacked by vehicle type (Q5)."""
    g = (bin_frame(df, bin_min)
         .groupby(["facet", "bin", "span", "vehicle_type"], as_index=False)
         .agg(n=("id", "size")))
    fig = px.bar(g, x="bin", y="n", color="vehicle_type", facet_col="facet",
                 barmode="relative", custom_data=["span"],
                 category_orders={"facet": facets}, color_discrete_map=colors,
                 labels={"n": "Vehicles", "bin": "", "vehicle_type": "Vehicle type"})
    fig.update_traces(xperiod=bin_min * 60_000, xperiodalignment="middle",
                      hovertemplate="%{customdata[0]}<br><b>%{y}</b> vehicles<extra>%{fullData.name}</extra>")
    return finish(fig, 300, uirev)


def speed_per_lane(df, lanes, colors, uirev) -> go.Figure:
    """Average speed over the whole window, per carriageway and vehicle type."""
    g = (df.groupby(["lane", "vehicle_type"], as_index=False)
           .agg(avg=("speed_kmh", "mean"), n=("id", "size")))
    fig = px.bar(g, y="lane", x="avg", color="vehicle_type", orientation="h",
                 barmode="group", custom_data=["n"], text="avg",
                 # reversed twice, so lanes read top-down with car above truck in each
                 category_orders={"lane": lanes[::-1]}, color_discrete_map=colors,
                 labels={"avg": "Average speed (km/h)", "lane": "", "vehicle_type": "Vehicle type"})
    fig.update_traces(texttemplate="%{x:.0f}", textposition="outside", cliponaxis=False,
                      textfont_color=TEXT,
                      hovertemplate="%{y}<br><b>%{x:.1f} km/h</b> average"
                                    " · %{customdata[0]} vehicles<extra>%{fullData.name}</extra>")
    fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    fig.update_xaxes(showgrid=True)
    return finish(fig, 300, uirev, time_axis=False)


def speed_scatter(df, alert, facets, colors, uirev) -> go.Figure:
    """Every measured speed over time, with the alert threshold."""
    measured = df.dropna(subset=["speed_kmh"])
    fig = px.scatter(measured, x="time", y="speed_kmh", color="vehicle_type", facet_col="facet",
                     render_mode="webgl", category_orders={"facet": facets},
                     color_discrete_map=colors,
                     custom_data=["vehicle_id", "direction_label", "speed_source"],
                     labels={"speed_kmh": "Speed (km/h)", "time": "", "vehicle_type": "Vehicle type"})
    fig.update_traces(marker=dict(size=7, opacity=0.8),
                      hovertemplate="%{x|%H:%M:%S} · %{customdata[0]}<br><b>%{y:.1f} km/h</b>"
                                    " · %{customdata[1]}<br>%{customdata[2]}"
                                    "<extra>%{fullData.name}</extra>")
    finish(fig, 340, uirev)
    fig.add_hline(y=alert, line=dict(color=CRITICAL, width=1),
                  annotation=dict(text=f"Alert {alert:g} km/h", font_color=CRITICAL,
                                  bgcolor="rgba(22,27,34,0.85)",
                                  xanchor="left", x=0, yanchor="bottom"))
    return fig


# ── Page pieces ───────────────────────────────────────────────────────────────

def field(label: str, control) -> html.Div:
    return html.Div(className="field", children=[html.Label(label), control])


def card(title_id: str, body, wide: bool = False, note: str = "") -> html.Section:
    return html.Section(className="card wide" if wide else "card", children=[
        html.Div(className="card-head", children=[
            html.H3(id=title_id), html.Span(note, className="card-note")]),
        body,
    ])


def graph(graph_id: str) -> dcc.Graph:
    return dcc.Graph(id=graph_id, config=GRAPH_CONFIG,
                     figure=empty_figure("Loading…", 300))


def kpi_tiles(df: pd.DataFrame, now: pd.Timestamp, window: int, alert: float) -> list:
    speeds = df["speed_kmh"]
    recent = int((df["time_utc"] >= now - pd.Timedelta(minutes=5)).sum())
    latest = ago((now - df["time_utc"].max()).total_seconds()) if len(df) else "none yet"
    over_limit = int((speeds > df["vehicle_type"].map(LIMITS_KMH)).sum())
    over_alert = int((speeds > alert).sum())
    avg = f"{speeds.mean():.1f}" if speeds.notna().any() else "—"

    def tile(label, value, sub, status=""):
        return html.Div(className=f"kpi {status}".strip(), children=[
            html.Span(label, className="kpi-label"),
            html.Span(value, className="kpi-value"),
            html.Span(sub, className="kpi-sub"),
        ])

    return [
        tile("Vehicles", f"{len(df):,}", f"in the last {minutes_label(window)}"),
        tile("Last 5 min", f"{recent:,}", f"latest {latest}"),
        tile("Average speed", avg, "km/h"),
        tile("Over the limit", f"{over_limit:,}", "cars > 90 · trucks > 80 km/h"),
        tile("Over-speed alerts", f"{over_alert:,}", f"> {alert:g} km/h",
             "critical" if over_alert else ""),
    ]


def alerts_table(df: pd.DataFrame, alert: float, show_camera: bool):
    rows = df[df["speed_kmh"] > alert].sort_values("time_utc", ascending=False).head(ALERT_ROWS)
    if rows.empty:
        return html.P(f"No vehicle over {alert:g} km/h in this window.", className="empty")
    head = ["Time"] + (["Camera"] if show_camera else []) + ["Vehicle", "Type", "Speed", "Carriageway", "Source"]
    body = []
    for r in rows.itertuples():
        cells = [html.Td(r.time.strftime("%H:%M:%S"), className="muted")]
        if show_camera:
            cells.append(html.Td(pretty(r.camera)))
        cells += [html.Td(r.vehicle_id, className="mono"), html.Td(r.vehicle_type),
                  html.Td(f"{r.speed_kmh:.1f} km/h", className="num speed"),
                  html.Td(r.direction_label), html.Td(r.speed_source, className="muted")]
        body.append(html.Tr(cells))
    return html.Table([
        html.Thead(html.Tr([html.Th(h, className="num" if h == "Speed" else None) for h in head])),
        html.Tbody(body),
    ])


# ── App ───────────────────────────────────────────────────────────────────────

def build_app(store: RecordStore, refresh_s: float, live_url: str) -> Dash:
    history_min = int(store.history.total_seconds() // 60)
    windows = [w for w in WINDOWS_MIN if w <= history_min] or [history_min]
    radio = dict(className="segmented", inline=True)

    app = Dash(__name__, title="Traffic Monitoring · Analytics", update_title=None)
    app.layout = html.Div([
        html.Header(className="topbar", children=[
            html.H1(["Traffic Monitoring", html.Span("Live analytics")]),
            html.Div(className="topbar-meta", children=[
                html.Span("Connecting…", id="status", className="status off"),
                html.A("Live video ↗", href=live_url, target="_blank", rel="noopener"),
            ]),
        ]),
        html.Main(className="page", children=[
            dcc.RadioItems(id="camera", options=[{"label": "All cameras", "value": ALL}],
                           value=ALL, className="tabs", inline=True),
            html.Div(className="filters", children=[
                field("Carriageway", dcc.Dropdown(id="lanes", multi=True,
                                                  placeholder="All carriageways")),
                field("Vehicle type", dcc.Dropdown(id="types", multi=True, placeholder="All types",
                                                   options=[{"label": t, "value": t} for t in TYPE_COLORS])),
                field("Time window", dcc.RadioItems(
                    id="window", value=60 if 60 in windows else windows[-1],
                    options=[{"label": minutes_label(w), "value": w} for w in windows], **radio)),
                field("Bin size", dcc.RadioItems(
                    id="bin", value=5,
                    options=[{"label": f"{b} min", "value": b} for b in BINS_MIN], **radio)),
            ]),
            html.Div(id="kpis", className="kpis"),
            html.Div(className="grid", children=[
                card("speed-title", graph("speed-bins"), wide=True),
                card("count-title", graph("count-bins")),
                card("lane-title", graph("lane-speed"), note="selected window"),
                card("scatter-title", graph("speeds"), wide=True, note="one dot per vehicle"),
                card("alerts-title", html.Div(id="alerts", className="table-wrap"), wide=True,
                     note="newest first"),
            ]),
        ]),
        dcc.Interval(id="tick", interval=int(refresh_s * 1000)),
    ])

    @app.callback(
        Output("status", "children"), Output("status", "className"),
        Output("camera", "options"),
        Input("tick", "n_intervals"), State("camera", "options"))
    def update_status(_, current_options):
        snap = store.snapshot()
        options = [{"label": "All cameras", "value": ALL}] + [
            {"label": pretty(c), "value": c} for c in snap["cameras"]]
        if snap["error"]:
            status = (f"Server unreachable — {snap['error']}", "status bad")
        elif snap["synced_at"] is None:
            status = (f"Loading records from {store.server}…", "status off")
        else:
            local = snap["synced_at"].astimezone()
            status = (f"Live · synced {local:%H:%M:%S}", "status ok")
        return *status, (no_update if options == current_options else options)

    @app.callback(
        Output("lanes", "options"), Output("lanes", "value"),
        Input("camera", "value"), Input("tick", "n_intervals"), State("lanes", "value"))
    def update_lanes(camera, _, selected):
        snap = store.snapshot()
        cameras = snap["cameras"] if camera == ALL else [camera]
        options = []
        for cam in cameras:
            df = snap["frames"].get(cam)
            for direction in sorted(df["direction"].unique()) if df is not None else []:
                label = pretty(direction) if camera != ALL else f"{pretty(cam)} · {pretty(direction)}"
                options.append({"label": label, "value": f"{cam}|{direction}"})
        valid = {o["value"] for o in options}
        value = [v for v in (selected or []) if v in valid]
        if ctx.triggered_id == "tick" and value == (selected or []):
            value = no_update
        return options, value

    @app.callback(
        Output("kpis", "children"),
        Output("speed-title", "children"), Output("speed-bins", "figure"),
        Output("count-title", "children"), Output("count-bins", "figure"),
        Output("lane-title", "children"), Output("lane-speed", "figure"),
        Output("scatter-title", "children"), Output("speeds", "figure"),
        Output("alerts-title", "children"), Output("alerts", "children"),
        Input("tick", "n_intervals"), Input("camera", "value"), Input("lanes", "value"),
        Input("types", "value"), Input("window", "value"), Input("bin", "value"))
    def render(_, camera, lanes, types, window, bin_min):
        snap = store.snapshot()
        alert = snap["alert"]
        overview = camera == ALL
        cameras = snap["cameras"] if overview else [camera]
        now = pd.Timestamp.now(tz="UTC")

        frames = [snap["frames"][c] for c in cameras if c in snap["frames"]]
        df = pd.concat(frames, ignore_index=True) if frames else to_frame([], "")
        df = df[df["time_utc"] >= now - pd.Timedelta(minutes=window)]
        if lanes:
            df = df[(df["camera"] + "|" + df["direction"]).isin(lanes)]
        if types:
            df = df[df["vehicle_type"].isin(types)]
        df = df.assign(
            time=df["time_utc"].dt.tz_convert(local_tz()).dt.tz_localize(None),
            direction_label=df["direction"].map(pretty),
            facet=(df["camera"] if overview else df["direction"]).map(pretty),
            lane=(df["camera"].map(pretty) + " · " if overview else "") + df["direction"].map(pretty),
        )

        # Panels: one per camera in the overview, one per carriageway for a camera
        facets = ([pretty(c) for c in cameras] if overview
                  else sorted(df["facet"].unique()))
        lane_order = sorted(df["lane"].unique(), key=lambda l: (
            [pretty(c) for c in cameras].index(l.split(" · ")[0]) if overview else 0, l))
        colors = {t: TYPE_COLORS.get(t, OTHER_TYPE) for t in df["vehicle_type"].unique()}

        # Coarser bins for long windows, so a panel holds at most MAX_BINS bars
        bin_min = next((b for b in NICE_BINS if b >= max(bin_min, math.ceil(window / MAX_BINS))),
                       NICE_BINS[-1])
        uirev = f"{camera}|{window}|{bin_min}"
        per = minutes_label(bin_min)
        split = "camera" if overview else "carriageway"
        titles = [
            f"Average speed per {per} by vehicle type and {split}",
            f"Vehicles per {per}",
            "Average speed by carriageway and vehicle type",
            "Measured speeds",
            f"Over-speed alerts (> {alert:g} km/h)",
        ]

        if df.empty:
            message = ("Waiting for the first records…" if snap["synced_at"] is None
                       else "No records in this window yet")
            figures = [empty_figure(message, h) for h in (340, 300, 300, 340)]
        else:
            figures = [
                speed_per_bin(df, bin_min, facets, colors, uirev),
                count_per_bin(df, bin_min, facets, colors, uirev),
                speed_per_lane(df, lane_order, colors, uirev),
                speed_scatter(df, alert, facets, colors, uirev),
            ]

        return (kpi_tiles(df, now, window, alert),
                titles[0], figures[0], titles[1], figures[1],
                titles[2], figures[2], titles[3], figures[3],
                titles[4], alerts_table(df, alert, overview))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Plotly dashboard of the vehicle records, per camera.",
        epilog="For a server behind Cloudflare Access, set CF_ACCESS_CLIENT_ID and "
               "CF_ACCESS_CLIENT_SECRET to a service token.")
    parser.add_argument("--server", default=os.getenv("SERVER_URL", "http://localhost:8090"),
                        help="dashboard service to read from (default: %(default)s)")
    parser.add_argument("--live-url", default=os.getenv("LIVE_URL"),
                        help='target of the "Live video" link (default: the --server URL)')
    parser.add_argument("--history", type=float, default=float(os.getenv("HISTORY_HOURS", "6")),
                        help="hours of records kept in memory (default: %(default)s)")
    parser.add_argument("--refresh", type=float, default=float(os.getenv("REFRESH_S", "5")),
                        help="seconds between updates (default: %(default)s)")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"),
                        help="address to listen on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8050")),
                        help="port this dashboard listens on (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # one line per poll otherwise

    headers = {}
    if os.getenv("CF_ACCESS_CLIENT_ID"):
        headers = {"CF-Access-Client-Id":     os.environ["CF_ACCESS_CLIENT_ID"],
                   "CF-Access-Client-Secret": os.getenv("CF_ACCESS_CLIENT_SECRET", "")}

    store = RecordStore(args.server, timedelta(hours=args.history), headers)
    threading.Thread(target=store.run, args=(args.refresh,), daemon=True).start()

    log.info("Reading %s — open http://localhost:%d", store.server, args.port)
    app = build_app(store, args.refresh, args.live_url or store.server)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
