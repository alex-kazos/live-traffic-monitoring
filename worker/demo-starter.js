/**
 * demo-starter.js — Cloudflare Worker in front of the dashboard hostname.
 *
 * While the EC2 host is running, requests pass straight through to the tunnel.
 * When it is stopped, the tunnel is down and Cloudflare answers 530; page loads
 * then get a "Start demo" page instead, which starts the instance and reloads
 * once the dashboard answers. The host powers itself off again later
 * (deploy/aws/traffic-autostop.timer).
 *
 * Bindings:
 *   AWS_REGION             variable, e.g. "eu-north-1"
 *   INSTANCE_ID            variable, e.g. "i-0123456789abcdef0"
 *   AWS_ACCESS_KEY_ID      secret  ─┐ an IAM user limited to deploy/aws/iam-policy.json
 *   AWS_SECRET_ACCESS_KEY  secret  ─┘
 */

const DEMO = "/__demo";
// Tunnel down (530), or tunnel up but the dashboard container is not answering yet
const DOWN = new Set([502, 503, 504, 530]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith(`${DEMO}/`)) {
      try {
        if (url.pathname === `${DEMO}/status`) return json(await status(env, url));
        if (url.pathname === `${DEMO}/start`) {
          if (request.method !== "POST") return json({ error: "Use POST" }, 405);
          return json(await start(env));
        }
        return json({ error: "Not found" }, 404);
      } catch (err) {
        return json({ error: err.message }, 502);
      }
    }

    const res = await fetch(request);
    if (!DOWN.has(res.status) || !wantsPage(request)) return res;
    return new Response(START_PAGE, {
      status: 503,
      headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
    });
  },
};

// ── Demo control ─────────────────────────────────────────────────────────────

async function status(env, url) {
  if (await dashboardUp(url)) return { state: "running", ready: true };
  return { state: await instanceState(env), ready: false };
}

async function start(env) {
  const state = await instanceState(env);
  if (state !== "stopped") return { state, started: false };
  const xml = await ec2(env, { Action: "StartInstances", "InstanceId.1": env.INSTANCE_ID });
  return { state: tag(xml, "currentState") || "pending", started: true };
}

async function dashboardUp(url) {
  // Same-zone subrequests skip this Worker and go to the tunnel
  try {
    const res = await fetch(`${url.origin}/api/cameras`, { headers: { accept: "application/json" } });
    return res.ok;
  } catch {
    return false;
  }
}

async function instanceState(env) {
  const xml = await ec2(env, { Action: "DescribeInstances", "InstanceId.1": env.INSTANCE_ID });
  return tag(xml, "instanceState") || "unknown";
}

/** <wrapper><code>…</code><name>STATE</name></wrapper> → STATE */
function tag(xml, wrapper) {
  const m = xml.match(new RegExp(`<${wrapper}>\\s*<code>\\d+</code>\\s*<name>([a-z-]+)</name>`));
  return m ? m[1] : null;
}

function wantsPage(request) {
  return request.method === "GET" && (request.headers.get("accept") || "").includes("text/html");
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

// ── EC2 Query API with Signature Version 4 ───────────────────────────────────

async function ec2(env, params) {
  const host = `ec2.${env.AWS_REGION}.amazonaws.com`;
  const body = new URLSearchParams({ ...params, Version: "2016-11-15" }).toString();
  const headers = {
    "content-type": "application/x-www-form-urlencoded; charset=utf-8",
    host,
    "x-amz-date": amzDate(new Date()),
  };
  headers.authorization = await authorization({
    method: "POST", path: "/", query: "", headers, body,
    region: env.AWS_REGION, service: "ec2",
    accessKeyId: env.AWS_ACCESS_KEY_ID, secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
  });
  delete headers.host; // set by fetch from the URL

  const res = await fetch(`https://${host}/`, { method: "POST", headers, body });
  const text = await res.text();
  if (!res.ok) {
    const code = (text.match(/<Code>([^<]+)<\/Code>/) || [])[1] || res.status;
    const message = (text.match(/<Message>([^<]+)<\/Message>/) || [])[1] || "";
    throw new Error(`EC2 ${params.Action} failed: ${code} ${message}`);
  }
  return text;
}

function amzDate(d) {
  return d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, ""); // 20150830T123600Z
}

/** Authorization header for a request whose headers already include host and x-amz-date. */
async function authorization({ method, path, query, headers, body, region, service,
                                      accessKeyId, secretAccessKey }) {
  const date = headers["x-amz-date"];
  const names = Object.keys(headers).map((h) => h.toLowerCase()).sort();
  const lower = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  const canonicalHeaders = names.map((h) => `${h}:${String(lower[h]).trim()}\n`).join("");
  const signedHeaders = names.join(";");

  const canonicalRequest = [method, path, query, canonicalHeaders, signedHeaders, await sha256Hex(body)]
    .join("\n");
  const scope = `${date.slice(0, 8)}/${region}/${service}/aws4_request`;
  const stringToSign = ["AWS4-HMAC-SHA256", date, scope, await sha256Hex(canonicalRequest)].join("\n");

  let key = new TextEncoder().encode(`AWS4${secretAccessKey}`);
  for (const part of [date.slice(0, 8), region, service, "aws4_request"]) {
    key = await hmac(key, part);
  }
  const signature = hex(await hmac(key, stringToSign));
  return `AWS4-HMAC-SHA256 Credential=${accessKeyId}/${scope}, ` +
         `SignedHeaders=${signedHeaders}, Signature=${signature}`;
}

async function hmac(key, data) {
  const k = await crypto.subtle.importKey("raw", key, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return new Uint8Array(await crypto.subtle.sign("HMAC", k, new TextEncoder().encode(data)));
}

async function sha256Hex(data) {
  return hex(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(data))));
}

function hex(bytes) {
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ── Start page ───────────────────────────────────────────────────────────────

const START_PAGE = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic Monitoring Demo</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#5b6475; --accent:#2563eb; --border:#e2e6ee; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1219; --card:#181d27; --text:#e7eaf0; --muted:#9aa3b5; --accent:#5b8dff; --border:#2a3140; }
  }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:grid; place-items:center; padding:16px;
         background:var(--bg); color:var(--text); font:16px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
  main { width:100%; max-width:440px; background:var(--card); border:1px solid var(--border);
         border-radius:14px; padding:28px; text-align:center; }
  h1 { font-size:1.35rem; margin:0 0 6px; }
  p { color:var(--muted); margin:0 0 20px; }
  button { font:inherit; font-weight:600; color:#fff; background:var(--accent); border:0;
           border-radius:10px; padding:12px 22px; cursor:pointer; }
  button:disabled { opacity:.55; cursor:default; }
  #status { margin-top:18px; min-height:1.5em; font-size:.95rem; color:var(--muted); }
  .spin { display:inline-block; width:.9em; height:.9em; margin-right:.4em; vertical-align:-.1em;
          border:2px solid currentColor; border-right-color:transparent; border-radius:50%;
          animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<main>
  <h1>Live Traffic Monitoring</h1>
  <p>The demo server is asleep to save cost. Starting it takes about 3–5 minutes;
     it switches itself off again after a few hours.</p>
  <button id="start" disabled>Checking…</button>
  <div id="status"></div>
</main>
<script>
  const btn = document.getElementById("start");
  const out = document.getElementById("status");
  const MESSAGES = {
    pending: "Server is booting…",
    running: "Server is up — starting Kafka, Spark and the trackers…",
    stopping: "Server is shutting down — it can be started again in a moment…",
    "shutting-down": "Server is shutting down…",
  };
  let timer;

  function show(text, busy) {
    out.innerHTML = (busy ? '<span class="spin"></span>' : "") + text;
  }

  async function poll() {
    clearTimeout(timer);
    try {
      const s = await (await fetch("/__demo/status", { cache: "no-store" })).json();
      if (s.ready) { show("Ready — loading the dashboard…", true); location.reload(); return; }
      if (s.state === "stopped") {
        btn.disabled = false; btn.textContent = "▶ Start demo"; show("");
        return;
      }
      btn.disabled = true; btn.textContent = "Starting…";
      show(MESSAGES[s.state] || "Server state: " + s.state, true);
    } catch (e) {
      show("Could not reach the server status — retrying…", true);
    }
    timer = setTimeout(poll, 5000);
  }

  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Starting…";
    show("Asking AWS to start the server…", true);
    try {
      const r = await fetch("/__demo/start", { method: "POST" });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "HTTP " + r.status);
    } catch (e) {
      show("Could not start the server: " + e.message);
      btn.disabled = false; btn.textContent = "▶ Start demo";
      return;
    }
    timer = setTimeout(poll, 5000);
  });

  poll();
</script>
</body>
</html>`;
