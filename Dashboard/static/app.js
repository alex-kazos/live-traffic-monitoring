"use strict";

const POLL_MS = 2000;
const STALE_AFTER_S = 10;     // preview older than this = stream stalled
const MAX_RECORDS = 200;
const MAX_LOG_LINES = 1500;

let alertSpeed = 130;

// ── Helpers ──────────────────────────────────────────────────────────────────
async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

const pretty = (name) => name.replace(/_/g, " ");
const fixed = (value, digits = 0) => (value == null ? "—" : Number(value).toFixed(digits));

// Python and Docker timestamps carry 6–9 fractional digits; Date wants 3
const parseTime = (iso) => new Date(iso.replace(/(\.\d{3})\d+/, "$1"));

function duration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

// A link to localhost only works for a viewer on this machine, not through the tunnel
const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"];
function linkWorksHere(url) {
  return !LOCAL_HOSTS.includes(new URL(url).hostname) || LOCAL_HOSTS.includes(location.hostname);
}

function logClass(line) {
  if (/\b(ERROR|CRITICAL)\b|Traceback/.test(line)) return "err";
  if (/\bWARNING\b/.test(line)) return "warn";
  if (line.includes("SAVED")) return "saved";
  return "";
}

// ── One camera card ──────────────────────────────────────────────────────────
class CameraView {
  constructor({ name, container }) {
    this.name = name;
    this.lastEventId = 0;
    this.logSince = "";
    this.live = null;

    const root = document.getElementById("camera-template").content.firstElementChild.cloneNode(true);
    const $ = (selector) => root.querySelector(selector);
    this.badge = $(".badge");
    this.uptime = $(".uptime");
    this.video = $(".video");
    this.img = $(".video img");
    this.overlay = $(".video-overlay");
    this.directions = $(".directions");
    this.tbody = $("tbody");
    this.logs = $(".logs");
    this.logNote = $(".log-note");
    this.stats = Object.fromEntries([...root.querySelectorAll("[data-stat]")].map((el) => [el.dataset.stat, el]));

    $(".camera-name").textContent = pretty(name);
    $(".container-name").textContent = container;
    root.querySelector(".alert-label").textContent = `Over ${alertSpeed} km/h`;
    this.img.alt = `Live view: ${pretty(name)}`;
    this.img.addEventListener("error", () => setTimeout(() => this.connectVideo(), 3000));
    document.getElementById("cameras").append(root);

    this.connectVideo();
    this.poll();
  }

  api(path) {
    return `/api/cameras/${encodeURIComponent(this.name)}/${path}`;
  }

  connectVideo() {
    this.img.src = this.api(`video.mjpg?t=${Date.now()}`);
  }

  async poll() {
    await Promise.allSettled([this.updateSummary(), this.updateEvents(), this.updateLogs()]);
    setTimeout(() => this.poll(), POLL_MS);
  }

  async updateSummary() {
    let summary;
    try {
      summary = await getJSON(this.api("summary"));
    } catch (err) {
      this.setState("bad", "Dashboard unreachable", `Dashboard API unreachable (${err.message}).`);
      return;
    }
    const { container, frame_age_s: age, totals, directions } = summary;

    if (container.status === "running") {
      if (age == null) this.setState("warn", "Waiting for stream", "Waiting for the first frame…");
      else if (age > STALE_AFTER_S) this.setState("warn", "Stream stalled", `No new frame for ${duration(age)} — the tracker is reconnecting to the stream.`);
      else this.setState("ok", "Live");
      const restarts = container.restarts ? ` · ${container.restarts} restart${container.restarts > 1 ? "s" : ""}` : "";
      this.uptime.textContent = `up ${duration((Date.now() - parseTime(container.started_at)) / 1000)}${restarts}`;
    } else {
      const labels = {
        "not found":   ["off", "Not running", "Container is not running. Start it with docker compose up -d."],
        "unavailable": ["bad", "Docker unavailable", "Can't reach the Docker socket from the dashboard."],
        "restarting":  ["bad", "Restarting", "Container is crash-looping — check the logs below."],
      };
      const status = container.status;
      const [state, text, detail] = labels[status] ??
        ["off", status[0].toUpperCase() + status.slice(1), `Container is ${status}.`];
      this.setState(state, text, detail);
      this.uptime.textContent = "";
    }

    this.stats.total.textContent = totals.total ?? 0;
    this.stats.last_5_min.textContent = totals.last_5_min ?? 0;
    this.stats.avg_speed_kmh.textContent = fixed(totals.avg_speed_kmh, 1);
    this.stats.over_alert.textContent = totals.over_alert ?? 0;

    this.directions.replaceChildren(...directions.map((d) => {
      const chip = document.createElement("span");
      chip.className = "direction";
      const label = document.createElement("b");
      label.textContent = pretty(d.direction);
      chip.append(label, ` · ${d.total} vehicles · avg ${fixed(d.avg_speed_kmh, 1)} km/h`);
      return chip;
    }));
  }

  setState(state, text, detail = "") {
    this.badge.className = `badge ${state}`;
    this.badge.textContent = text;
    const live = state === "ok";
    this.video.classList.toggle("offline", !live);
    this.overlay.textContent = detail;
    if (live && this.live === false) this.connectVideo();   // stream came back
    this.live = live;
  }

  async updateEvents() {
    const { events } = await getJSON(this.api(`events?after_id=${this.lastEventId}`));
    if (!events.length) return;
    const initial = this.lastEventId === 0;
    this.lastEventId = events[0].id;

    const arrows = { away: "↑ ", toward: "↓ ", left: "← ", right: "→ " };
    const rows = events.map((e) => {
      const tr = document.createElement("tr");
      if (!initial) tr.classList.add("new");
      if (e.speed_kmh > alertSpeed) tr.classList.add("alert");
      tr.append(
        cell(parseTime(e.recorded_at).toLocaleTimeString(), "muted"),
        cell(e.vehicle_id, "mono"),
        cell(e.vehicle_type),
        cell(`${fixed(e.speed_kmh, 1)} km/h`, "num"),
        cell((arrows[e.movement] ?? "") + pretty(e.direction)),
        cell(e.speed_source, "muted"),
      );
      return tr;
    });
    this.tbody.prepend(...rows);
    while (this.tbody.rows.length > MAX_RECORDS) this.tbody.lastElementChild.remove();
  }

  async updateLogs() {
    const res = await getJSON(this.api(`logs?since=${encodeURIComponent(this.logSince)}`));
    this.logNote.textContent = res.error ?? "";
    this.logNote.classList.toggle("error", Boolean(res.error));
    this.logSince = res.last || this.logSince;
    if (!res.lines.length) return;

    const box = this.logs;
    const following = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
    box.append(...res.lines.map((line) => {
      const div = document.createElement("div");
      div.textContent = line;
      div.className = logClass(line);
      return div;
    }));
    while (box.childElementCount > MAX_LOG_LINES) box.firstElementChild.remove();
    if (following) box.scrollTop = box.scrollHeight;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
async function main() {
  const grid = document.getElementById("cameras");
  try {
    const config = await getJSON("/api/cameras");
    alertSpeed = config.alert_speed_kmh;
    document.getElementById("meta").textContent =
      `Alert threshold ${alertSpeed} km/h · refreshes every ${POLL_MS / 1000}s`;
    if (config.analytics_url && linkWorksHere(config.analytics_url)) {
      const link = document.getElementById("analytics-link");
      link.href = config.analytics_url;
      link.hidden = false;
    }
    if (!config.cameras.length) {
      grid.innerHTML = '<p class="message">No cameras configured. Set <code>CAMERAS</code> on the dashboard service.</p>';
      return;
    }
    config.cameras.forEach((camera) => new CameraView(camera));
  } catch (err) {
    const p = document.createElement("p");
    p.className = "message error";
    p.textContent = `Could not load the camera list: ${err.message}`;
    grid.replaceChildren(p);
  }
}

main();
