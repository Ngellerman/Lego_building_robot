"""
web_ui/server.py

Flask server for shoe-segment approval.

Pipeline mode (called from process_pointcloud.py):
  1. process_pointcloud runs CV pipeline, writes segments.json + mask .npy files
  2. run_approval_server(artifacts_dir, event) is called in a daemon thread
  3. User opens http://localhost:5000 — cards load automatically from segments.json
  4. User toggles selections, clicks "Confirm Selection"
  5. POST /api/confirm writes approved_segments.json, shuts down server, sets event
  6. process_pointcloud unblocks and publishes filtered point clouds

Standalone use:
  python3 web_ui/server.py
  # requires segments.json to already exist in artifacts_cv/
  # Confirm writes approved_segments.json but does not shut down the server

Routes:
  GET  /                       — main page
  GET  /artifacts_cv/<filename>   — serve images from artifacts_cv/
  GET  /api/segments           — read artifacts_cv/segments.json
  POST /api/confirm            — write approved IDs, re-render overlay, shut down
"""

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.serving import make_server

CV_DIR    = Path(__file__).parent.parent / "cv"
ARTIFACTS = Path(__file__).parent.parent / "artifacts_cv"
sys.path.insert(0, str(CV_DIR))

from save_segments import save_segments
from segment_image import visualise

SEGMENTS_JSON = ARTIFACTS / "segments.json"
APPROVED_JSON = ARTIFACTS / "approved_segments.json"
ORIG_IMAGE    = ARTIFACTS / "01_raw_image.png"
OVERLAY_IMAGE = ARTIFACTS / "02_segmentation_overlay.png"

app = Flask(__name__)

_server = None
_shutdown_event: threading.Event | None = None

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path) as f:
        return json.load(f)

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _rerender_overlay(approved_ids):
    if not ORIG_IMAGE.exists() or not SEGMENTS_JSON.exists():
        return
    approved_set = set(approved_ids)
    detections = []
    for seg in _read_json(SEGMENTS_JSON)["segments"]:
        if seg["id"] not in approved_set:
            continue
        mask_path = ARTIFACTS / f"mask_{seg['id']}.npy"
        if not mask_path.exists():
            continue
        detections.append(SimpleNamespace(
            label=seg["label"],
            confidence=seg["confidence"],
            class_scores=seg["class_scores"],
            box=seg["box"],
            mask=np.load(mask_path),
        ))
    image = Image.open(ORIG_IMAGE).convert("RGB")
    visualise(image, detections).save(OVERLAY_IMAGE)

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/artifacts_cv/<path:filename>")
def serve_artifact(filename):
    return send_from_directory(ARTIFACTS, filename)

@app.get("/api/segments")
def api_segments():
    if not SEGMENTS_JSON.exists():
        return jsonify({"segments": []})
    return jsonify(_read_json(SEGMENTS_JSON))

@app.post("/api/confirm")
def api_confirm():
    body = request.get_json(force=True)
    approved_ids = body.get("approved_ids", [])
    label_overrides = body.get("label_overrides", {})
    _write_json(APPROVED_JSON, {
        "approved_ids": approved_ids,
        "label_overrides": {str(k): v for k, v in label_overrides.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _rerender_overlay(approved_ids)
    if _shutdown_event is not None:
        _shutdown_event.set()
    if _server is not None:
        threading.Thread(target=_server.shutdown, daemon=True).start()
    return jsonify({"ok": True, "approved_ids": approved_ids})

@app.get("/")
def index():
    return HTML

# ── pipeline integration ──────────────────────────────────────────────────────

def run_approval_server(artifacts_dir: Path, event: threading.Event,
                        host: str = "0.0.0.0", port: int = 5000):
    """Start the Flask approval server (blocking). Called from process_pointcloud in a thread."""
    global ARTIFACTS, SEGMENTS_JSON, APPROVED_JSON, ORIG_IMAGE, OVERLAY_IMAGE
    global _server, _shutdown_event
    ARTIFACTS     = Path(artifacts_dir)
    SEGMENTS_JSON = ARTIFACTS / "segments.json"
    APPROVED_JSON = ARTIFACTS / "approved_segments.json"
    ORIG_IMAGE    = ARTIFACTS / "01_raw_image.png"
    OVERLAY_IMAGE = ARTIFACTS / "02_segmentation_overlay.png"
    _shutdown_event = event
    _server = make_server(host, port, app)
    _server.serve_forever()  # blocks until _server.shutdown() is called

# ── frontend ──────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shoe Segment Approval</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: system-ui, sans-serif;
    background: #111;
    color: #eee;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    padding: 16px 24px;
    background: #1a1a1a;
    border-bottom: 1px solid #333;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  header .subtitle { font-size: 0.8rem; color: #888; margin-top: 2px; }

  .main { display: flex; flex: 1; overflow: hidden; }

  /* left: overlay */
  .panel-image {
    flex: 1 1 55%;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-right: 1px solid #2a2a2a;
    overflow: auto;
  }
  .panel-image h2 {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
  }
  #img-wrap img { width: 100%; border-radius: 6px; border: 1px solid #2a2a2a; display: block; }
  .placeholder {
    display: flex; align-items: center; justify-content: center;
    min-height: 220px; border: 1px dashed #333; border-radius: 6px;
    color: #555; font-size: 0.88rem;
  }

  /* right: cards */
  .panel-segments { flex: 0 0 340px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-segments-header {
    padding: 14px 16px 10px;
    border-bottom: 1px solid #2a2a2a;
    display: flex; align-items: center; justify-content: space-between;
  }
  .panel-segments-header h2 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #666;
  }
  .count-badge { font-size: 0.75rem; color: #aaa; }
  .cards-scroll { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }

  .card {
    background: #1c1c1c; border: 2px solid #2e2e2e; border-radius: 8px;
    padding: 12px; display: flex; gap: 12px; align-items: flex-start;
    cursor: pointer; transition: border-color 0.15s, background 0.15s; user-select: none;
  }
  .card:hover { border-color: #444; }
  .card.selected { border-color: #4a9eff; background: #141e2e; }
  .card.deselected { opacity: 0.4; }
  .card-check { margin-top: 2px; width: 18px; height: 18px; flex-shrink: 0; accent-color: #4a9eff; cursor: pointer; }
  .card-thumb { width: 72px; height: 72px; object-fit: contain; border-radius: 4px; background: #111; flex-shrink: 0; }
  .card-info { flex: 1; min-width: 0; }
  .card-label { font-size: 0.9rem; font-weight: 600; text-transform: capitalize; margin-bottom: 4px; }
  .card-conf  { font-size: 0.75rem; color: #888; margin-bottom: 6px; }
  .score-bars { display: flex; flex-direction: column; gap: 3px; }
  .score-row  { display: flex; align-items: center; gap: 6px; font-size: 0.7rem; color: #777; }
  .score-name { width: 60px; flex-shrink: 0; text-align: right; }
  .score-track { flex: 1; background: #2a2a2a; border-radius: 2px; height: 4px; }
  .score-fill  { height: 4px; border-radius: 2px; background: #4a9eff; }
  .score-val   { width: 32px; flex-shrink: 0; }

  .label-select {
    background: #2a2a2a; color: #eee; border: 1px solid #444;
    border-radius: 4px; padding: 3px 6px; font-size: 0.85rem;
    margin-bottom: 4px; cursor: pointer; width: 100%;
  }
  .label-select:focus { outline: none; border-color: #4a9eff; }

  /* bottom bar */
  .bottom-bar {
    padding: 14px 20px; border-top: 1px solid #2a2a2a; background: #161616;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  .status { flex: 1; font-size: 0.8rem; color: #888; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .status.ok  { color: #4caf81; }
  .status.err { color: #e05a5a; }

  button {
    padding: 8px 18px; border-radius: 6px; border: none;
    font-size: 0.85rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.15s, background 0.15s;
  }
  button:disabled { opacity: 0.35; cursor: default; }
  .btn-confirm { background: #2d5a8e; color: #fff; }
  .btn-confirm:hover:not(:disabled) { background: #3870b0; }
</style>
</head>
<body>

<header>
  <h1>Shoe Segment Approval</h1>
  <div class="subtitle">Review detected segments and confirm your selection to continue the pipeline.</div>
</header>

<div class="main">
  <div class="panel-image">
    <h2>Detection Overlay</h2>
    <div id="img-wrap" class="placeholder">Loading overlay...</div>
  </div>

  <div class="panel-segments">
    <div class="panel-segments-header">
      <h2>Detected Segments</h2>
      <span class="count-badge" id="count-badge"></span>
    </div>
    <div class="cards-scroll" id="cards">
      <div style="color:#555; font-size:0.85rem; padding:8px;">Loading segments...</div>
    </div>
  </div>
</div>

<div class="bottom-bar">
  <div class="status" id="status">Loading segments from pipeline...</div>
  <button class="btn-confirm" id="btn-confirm" disabled onclick="confirmSelection()">Confirm Selection</button>
</div>

<script>
let segments = [];
let selected = new Set();
let labelOverrides = {};  // {id: label}

// ── load on startup ───────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', loadSegments);

async function loadSegments() {
  try {
    const res = await fetch('/api/segments');
    const data = await res.json();
    segments = data.segments || [];
    selected = new Set(segments.map(s => s.id));
    renderOverlay();
    renderCards();
    if (segments.length === 0) {
      setStatus('No segments found. Is segments.json written?', 'err');
    } else {
      setStatus(`Found ${segments.length} segment(s). Toggle selections then confirm.`);
      document.getElementById('btn-confirm').disabled = false;
    }
  } catch (e) {
    setStatus('Failed to load segments: ' + e.message, 'err');
  }
}

// ── overlay ───────────────────────────────────────────────────────────────────

function renderOverlay() {
  const wrap = document.getElementById('img-wrap');
  wrap.className = '';
  wrap.innerHTML = '';
  const img = document.createElement('img');
  img.src = '/artifacts_cv/02_segmentation_overlay.png?' + Date.now();
  img.alt = 'detection overlay';
  img.onerror = () => { wrap.className = 'placeholder'; wrap.textContent = 'No overlay image.'; };
  wrap.appendChild(img);
}

// ── cards ─────────────────────────────────────────────────────────────────────

function renderCards() {
  const container = document.getElementById('cards');
  container.innerHTML = '';
  document.getElementById('count-badge').textContent =
    segments.length ? `${selected.size} / ${segments.length} selected` : '';

  if (!segments.length) {
    container.innerHTML = '<div style="color:#555;font-size:0.85rem;padding:8px;">No segments detected.</div>';
    return;
  }

  segments.forEach(seg => {
    const isSel = selected.has(seg.id);
    const card = document.createElement('div');
    card.className = 'card ' + (isSel ? 'selected' : 'deselected');
    card.onclick = () => toggleCard(seg.id);

    const scoreBars = Object.entries(seg.class_scores)
      .sort((a, b) => b[1] - a[1])
      .map(([name, val]) => `
        <div class="score-row">
          <span class="score-name">${name}</span>
          <div class="score-track"><div class="score-fill" style="width:${(val*100).toFixed(1)}%"></div></div>
          <span class="score-val">${(val*100).toFixed(0)}%</span>
        </div>`).join('');

    const currentLabel = labelOverrides[seg.id] || seg.label;
    const classes = ['sneaker', 'flip flop', 'slipper'];
    const options = classes.map(c =>
      `<option value="${c}" ${c === currentLabel ? 'selected' : ''}>${c}</option>`
    ).join('');

    card.innerHTML = `
      <input type="checkbox" class="card-check" ${isSel ? 'checked' : ''}
             onclick="event.stopPropagation(); toggleCard(${seg.id})">
      <img class="card-thumb" src="/artifacts_cv/${seg.crop_image}?v=${Date.now()}" alt="${seg.label}">
      <div class="card-info">
        <select class="label-select" onclick="event.stopPropagation()"
                onchange="setLabelOverride(${seg.id}, this.value)">${options}</select>
        <div class="card-conf">conf ${(seg.confidence*100).toFixed(1)}% &nbsp;·&nbsp; id ${seg.id}</div>
        <div class="score-bars">${scoreBars}</div>
      </div>`;
    container.appendChild(card);
  });
}

function toggleCard(id) {
  if (selected.has(id)) selected.delete(id);
  else selected.add(id);
  renderCards();
}

function setLabelOverride(id, label) {
  const seg = segments.find(s => s.id === id);
  if (seg && label === seg.label) delete labelOverrides[id];
  else labelOverrides[id] = label;
}

// ── confirm ───────────────────────────────────────────────────────────────────

async function confirmSelection() {
  const ids = [...selected];
  const btn = document.getElementById('btn-confirm');
  btn.disabled = true;
  setStatus('Confirming...');
  try {
    const res = await fetch('/api/confirm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({approved_ids: ids, label_overrides: labelOverrides}),
    });
    const data = await res.json();
    if (data.ok) {
      setStatus(`Confirmed — ${ids.length} segment(s) approved. Pipeline continuing. You may close this tab.`, 'ok');
      renderOverlay();
    } else {
      setStatus(data.error || 'Confirm failed.', 'err');
      btn.disabled = false;
    }
  } catch (e) {
    setStatus('Error: ' + e.message, 'err');
    btn.disabled = false;
  }
}

// ── helpers ───────────────────────────────────────────────────────────────────

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status' + (cls ? ' ' + cls : '');
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
