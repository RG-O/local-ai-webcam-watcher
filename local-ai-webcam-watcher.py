"""
Local AI Webcam Monitor
-------------------
Connects to an RTSP camera, captures screenshots at a configurable interval,
sends rolling batches to a local Ollama vision model, and publishes an ntfy
notification when the AI response begins with a configured trigger word.

Also provides:
- A system tray icon
- A LAN-accessible Flask settings/history page
- Persistent settings in settings.json
- Persistent AI response history in history.json

Install:
    pip install opencv-python flask ollama requests pystray pillow

Run:
    python rtsp_ollama_monitor.py

Then open:
    http://localhost:5000
or from another device on your LAN:
    http://<THIS-PC-IP>:5000
"""

import io
import json
import os
import re
import socket
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import ollama
import requests
from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from PIL import Image, ImageDraw
import pystray


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "settings.json"
HISTORY_FILE = APP_DIR / "history.json"

DEFAULT_SETTINGS = {
    "rtsp_url": "rtsp://username:password@192.168.1.100:554/stream1",
    "screenshot_interval_seconds": 5.0,
    "batch_size": 4,
    "ollama_model": "qwen2.5vl:7b",
    "prompt": (
        "Review these camera screenshots in chronological order. "
        "If you detect something that should trigger an alert, begin your "
        "response with ALERT. Otherwise begin your response with NO. "
        "Then briefly explain what you observed."
    ),
    "trigger_word": "ALERT",
    "ntfy_server": "https://ntfy.sh",
    "ntfy_topic": "",
    "ntfy_title": "RTSP Camera AI Alert",
    "send_latest_image_with_notification": True,
    "start_watching_on_startup": False,
    "web_port": 5000,
    "max_history_items": 250,
    "jpeg_quality": 85,
    "max_image_width": 1280,
    "max_image_height": 720,
}

settings_lock = threading.Lock()
history_lock = threading.Lock()
frames_lock = threading.Lock()
state_lock = threading.Lock()

settings = {}
history = []
frame_buffer = deque()
latest_frame_jpeg = None

stop_event = threading.Event()
settings_changed_event = threading.Event()
watching_event = threading.Event()

runtime_state = {
    "watching": False,
    "camera_connected": False,
    "last_capture_time": None,
    "last_ai_time": None,
    "last_ai_duration_seconds": None,
    "last_error": "",
    "analysis_in_progress": False,
    "capture_count": 0,
    "analysis_count": 0,
}

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_json(path, data):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def load_settings():
    global settings

    loaded = {}
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
        except Exception as exc:
            print(f"Could not load settings.json: {exc}")

    settings = DEFAULT_SETTINGS.copy()
    settings.update(loaded)
    save_settings()


def save_settings():
    with settings_lock:
        save_json(SETTINGS_FILE, settings)


def load_history():
    global history

    if not HISTORY_FILE.exists():
        history = []
        return

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
            history = loaded if isinstance(loaded, list) else []
    except Exception as exc:
        print(f"Could not load history.json: {exc}")
        history = []


def add_history_item(item):
    with history_lock:
        history.insert(0, item)

        with settings_lock:
            max_items = max(1, int(settings.get("max_history_items", 250)))

        del history[max_items:]
        save_json(HISTORY_FILE, history)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_settings_snapshot():
    with settings_lock:
        return dict(settings)


def update_runtime_state(**changes):
    with state_lock:
        runtime_state.update(changes)


def get_runtime_state():
    with state_lock:
        return dict(runtime_state)


def set_watching(enabled):
    """Start or stop camera monitoring without shutting down the app."""
    global frame_buffer

    enabled = bool(enabled)

    if enabled:
        watching_event.set()
        update_runtime_state(watching=True, last_error="")
        print("Watching started.")
    else:
        watching_event.clear()

        # Force the camera worker to release its RTSP connection promptly.
        settings_changed_event.set()

        # Discard the rolling batch so old screenshots are not mixed with
        # new screenshots when watching is started again.
        with frames_lock:
            frame_buffer.clear()

        update_runtime_state(
            watching=False,
            camera_connected=False,
        )
        print("Watching stopped.")


def toggle_watching():
    set_watching(not watching_event.is_set())


def get_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def first_word(text):
    """
    Return the first actual word-like token.

    Examples:
        "ALERT Person seen" -> "ALERT"
        "**ALERT** Person seen" -> "ALERT"
        "\nAlert: Person seen" -> "Alert"
    """
    match = re.search(r"[A-Za-z0-9_-]+", text or "")
    return match.group(0) if match else ""


def resize_frame(frame, max_width, max_height):
    height, width = frame.shape[:2]

    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))

    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return frame

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def frame_to_jpeg(frame, quality):
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )

    if not success:
        raise RuntimeError("OpenCV could not encode camera frame as JPEG.")

    return encoded.tobytes()


def get_ollama_models():
    """
    Return locally installed Ollama model names.

    Supports both dict-style and object-style responses from versions
    of the Ollama Python package.
    """
    try:
        response = ollama.list()

        if hasattr(response, "models"):
            model_items = response.models
        else:
            model_items = response.get("models", [])

        names = []

        for model in model_items:
            if hasattr(model, "model"):
                name = model.model
            elif hasattr(model, "name"):
                name = model.name
            elif isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = None

            if name:
                names.append(str(name))

        return sorted(set(names), key=str.lower)
    except Exception as exc:
        print(f"Could not retrieve Ollama models: {exc}")
        return []


def extract_ollama_content(response):
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            return message.content or ""

    if isinstance(response, dict):
        return response.get("message", {}).get("content", "") or ""

    return str(response)


def extract_ollama_stats(response):
    """
    Ollama response metadata varies by package/version/model.
    Grab useful timing/token fields when available.
    """
    def get_value(name):
        if hasattr(response, name):
            return getattr(response, name)
        if isinstance(response, dict):
            return response.get(name)
        return None

    return {
        "prompt_eval_count": get_value("prompt_eval_count"),
        "eval_count": get_value("eval_count"),
        "total_duration_ns": get_value("total_duration"),
        "load_duration_ns": get_value("load_duration"),
    }


# ---------------------------------------------------------------------------
# ntfy
# ---------------------------------------------------------------------------

def send_ntfy_notification(ai_response, image_bytes=None):
    current = get_settings_snapshot()

    server = current["ntfy_server"].strip().rstrip("/")
    topic = current["ntfy_topic"].strip()

    if not server or not topic:
        raise RuntimeError("ntfy server or topic is not configured.")

    url = f"{server}/{topic}"
    title = current.get("ntfy_title", "RTSP Camera AI Alert")

    # ntfy supports a text message body. If image attachment is enabled,
    # send the image as the request body and place the AI response in the
    # Message header so the notification still contains the full response.
    if image_bytes and current.get("send_latest_image_with_notification", True):
        headers = {
            "Title": title,
            "Message": ai_response,
            "Filename": f"camera-alert-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg",
            "Content-Type": "image/jpeg",
        }

        response = requests.put(
            url,
            data=image_bytes,
            headers=headers,
            timeout=30,
        )
    else:
        headers = {"Title": title}

        response = requests.post(
            url,
            data=ai_response.encode("utf-8"),
            headers=headers,
            timeout=30,
        )

    response.raise_for_status()


# ---------------------------------------------------------------------------
# Ollama analysis
# ---------------------------------------------------------------------------

def run_ollama_analysis(batch):
    current = get_settings_snapshot()

    model = current["ollama_model"].strip()
    prompt = current["prompt"]
    trigger_word = current["trigger_word"].strip()

    if not model:
        raise RuntimeError("No Ollama model is configured.")

    if not batch:
        return

    update_runtime_state(analysis_in_progress=True)

    started = time.monotonic()
    timestamp = datetime.now().isoformat(timespec="seconds")

    try:
        # Ollama vision messages accept multiple images on one user message.
        # Bytes are passed directly so temporary image files are not needed.
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": list(batch),
                }
            ],
        )

        duration = time.monotonic() - started
        content = extract_ollama_content(response).strip()
        stats = extract_ollama_stats(response)

        response_first_word = first_word(content)
        triggered = bool(
            trigger_word
            and response_first_word
            and response_first_word.casefold() == trigger_word.casefold()
        )

        notification_sent = False
        notification_error = ""

        if triggered:
            try:
                latest_image = batch[-1] if current.get(
                    "send_latest_image_with_notification", True
                ) else None

                send_ntfy_notification(content, latest_image)
                notification_sent = True
            except Exception as exc:
                notification_error = str(exc)
                print(f"ntfy notification failed: {exc}")

        item = {
            "timestamp": timestamp,
            "model": model,
            "response": content,
            "first_word": response_first_word,
            "trigger_word": trigger_word,
            "triggered": triggered,
            "notification_sent": notification_sent,
            "notification_error": notification_error,
            "batch_size": len(batch),
            "duration_seconds": round(duration, 3),
            "prompt_eval_count": stats["prompt_eval_count"],
            "eval_count": stats["eval_count"],
        }

        add_history_item(item)

        update_runtime_state(
            last_ai_time=timestamp,
            last_ai_duration_seconds=round(duration, 3),
            last_error=notification_error,
            analysis_count=get_runtime_state()["analysis_count"] + 1,
        )

        print(
            f"[{timestamp}] AI: {content} "
            f"(triggered={triggered}, {duration:.2f}s)"
        )

    except Exception as exc:
        duration = time.monotonic() - started
        error = f"Ollama analysis failed: {exc}"
        print(error)

        add_history_item(
            {
                "timestamp": timestamp,
                "model": model,
                "response": "",
                "first_word": "",
                "trigger_word": trigger_word,
                "triggered": False,
                "notification_sent": False,
                "notification_error": "",
                "batch_size": len(batch),
                "duration_seconds": round(duration, 3),
                "error": error,
            }
        )

        update_runtime_state(
            last_ai_time=timestamp,
            last_ai_duration_seconds=round(duration, 3),
            last_error=error,
        )

    finally:
        update_runtime_state(analysis_in_progress=False)


def start_analysis_if_possible():
    state = get_runtime_state()

    # Avoid stacking several expensive Ollama calls on top of each other.
    if state["analysis_in_progress"]:
        return

    current = get_settings_snapshot()
    required = max(1, int(current["batch_size"]))

    with frames_lock:
        if len(frame_buffer) < required:
            return

        batch = list(frame_buffer)[-required:]

    thread = threading.Thread(
        target=run_ollama_analysis,
        args=(batch,),
        daemon=True,
        name="OllamaAnalysis",
    )
    thread.start()


# ---------------------------------------------------------------------------
# RTSP capture worker
# ---------------------------------------------------------------------------

def camera_worker():
    global latest_frame_jpeg, frame_buffer

    capture = None
    active_url = None
    next_capture_time = 0.0

    while not stop_event.is_set():
        # When watching is stopped, release the RTSP connection and idle.
        # The Flask web UI and tray icon remain active so monitoring can be
        # started again at any time.
        if not watching_event.is_set():
            if capture is not None:
                capture.release()
                capture = None

            active_url = None
            update_runtime_state(
                watching=False,
                camera_connected=False,
            )
            time.sleep(0.1)
            continue

        update_runtime_state(watching=True)

        current = get_settings_snapshot()
        rtsp_url = current["rtsp_url"].strip()
        interval = max(0.1, float(current["screenshot_interval_seconds"]))
        desired_batch_size = max(1, int(current["batch_size"]))

        # Keep enough frames for the rolling batch, but discard old frames
        # immediately if the user reduces the configured batch size.
        with frames_lock:
            if frame_buffer.maxlen != desired_batch_size:
                frame_buffer = deque(
                    list(frame_buffer)[-desired_batch_size:],
                    maxlen=desired_batch_size,
                )

        # Reconnect whenever the RTSP URL changes or capture isn't open.
        if (
            capture is None
            or not capture.isOpened()
            or active_url != rtsp_url
            or settings_changed_event.is_set()
        ):
            settings_changed_event.clear()

            if capture is not None:
                capture.release()

            active_url = rtsp_url

            if not rtsp_url:
                update_runtime_state(
                    camera_connected=False,
                    last_error="RTSP URL is empty.",
                )
                time.sleep(1)
                continue

            print(f"Connecting to RTSP camera: {rtsp_url}")
            capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            if not capture.isOpened():
                update_runtime_state(
                    camera_connected=False,
                    last_error="Could not connect to RTSP stream.",
                )
                print("Could not connect to RTSP stream. Retrying...")
                time.sleep(5)
                continue

            update_runtime_state(camera_connected=True, last_error="")
            next_capture_time = time.monotonic()

        # Continuously read so the frame is reasonably fresh when the
        # configured screenshot interval arrives.
        success, frame = capture.read()

        if not success or frame is None:
            update_runtime_state(
                camera_connected=False,
                last_error="RTSP frame read failed; reconnecting.",
            )
            capture.release()
            capture = None
            time.sleep(1)
            continue

        update_runtime_state(camera_connected=True)

        now = time.monotonic()

        if now < next_capture_time:
            time.sleep(0.01)
            continue

        try:
            processed = resize_frame(
                frame,
                current["max_image_width"],
                current["max_image_height"],
            )

            jpeg = frame_to_jpeg(
                processed,
                current["jpeg_quality"],
            )

            timestamp = datetime.now().isoformat(timespec="seconds")

            with frames_lock:
                frame_buffer.append(jpeg)
                latest_frame_jpeg = jpeg

            previous_count = get_runtime_state()["capture_count"]

            update_runtime_state(
                last_capture_time=timestamp,
                capture_count=previous_count + 1,
                last_error="",
            )

            print(
                f"[{timestamp}] Screenshot captured "
                f"({len(frame_buffer)}/{desired_batch_size} in rolling batch)"
            )

            # Once the rolling buffer is full, every new screenshot creates
            # a new analysis unless the prior Ollama call is still running.
            start_analysis_if_possible()

        except Exception as exc:
            error = f"Screenshot processing failed: {exc}"
            print(error)
            update_runtime_state(last_error=error)

        next_capture_time = now + interval

    if capture is not None:
        capture.release()


# ---------------------------------------------------------------------------
# Flask web UI
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = r"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Local AI Webcam Monitor</title>
    <style>
        :root {
            color-scheme: light dark;
            --bg: #111827;
            --card: #1f2937;
            --border: #374151;
            --text: #f3f4f6;
            --muted: #9ca3af;
            --accent: #3b82f6;
            --good: #10b981;
            --bad: #ef4444;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        .page {
            max-width: 1100px;
            margin: 0 auto;
            padding: 24px;
        }
        h1 { margin-top: 0; }
        h2 { margin-top: 0; font-size: 20px; }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 18px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 14px;
        }
        label {
            display: block;
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 5px;
        }
        input, select, textarea, button {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 7px;
            padding: 10px;
            font: inherit;
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        button {
            background: var(--accent);
            color: white;
            border: 0;
            cursor: pointer;
            font-weight: bold;
        }
        .checkbox-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 24px;
        }
        .checkbox-row input {
            width: auto;
        }
        .status {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 8px;
        }
        .status div {
            padding: 9px;
            background: rgba(255,255,255,.04);
            border-radius: 6px;
        }
        .muted { color: var(--muted); }
        .good { color: var(--good); }
        .bad { color: var(--bad); }
        .history-item {
            border-top: 1px solid var(--border);
            padding: 14px 0;
        }
        .history-item:first-child { border-top: 0; }
        .history-response {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            margin-top: 8px;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 8px;
            font-size: 12px;
            margin-left: 6px;
            background: rgba(255,255,255,.09);
        }
        img.preview {
            width: 100%;
            max-width: 640px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .latest-ai {
            max-width: 640px;
            margin-top: 12px;
            padding: 12px;
            background: rgba(255,255,255,.04);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .latest-ai-header {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }
        .latest-ai-response {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .actions {
            display: flex;
            gap: 10px;
        }
        .actions form {
            flex: 1;
        }
    </style>
</head>
<body>
<div class="page">
    <h1>Local AI Webcam Monitor</h1>

    <div class="card">
        <h2>Status</h2>
        <div class="status">
            <div>Watching:
                <strong id="status-watching" class="{{ 'good' if state.watching else 'bad' }}">
                    {{ 'Running' if state.watching else 'Stopped' }}
                </strong>
            </div>
            <div>Camera:
                <strong id="status-camera" class="{{ 'good' if state.camera_connected else 'bad' }}">
                    {{ 'Connected' if state.camera_connected else 'Disconnected' }}
                </strong>
            </div>
            <div>Captured: <strong id="status-captured">{{ state.capture_count }}</strong></div>
            <div>AI calls: <strong id="status-ai-calls">{{ state.analysis_count }}</strong></div>
            <div>AI running:
                <strong id="status-ai-running">{{ 'Yes' if state.analysis_in_progress else 'No' }}</strong>
            </div>
            <div>Last capture:
                <strong id="status-last-capture">{{ state.last_capture_time or 'Never' }}</strong>
            </div>
            <div>Last AI:
                <strong id="status-last-ai">{{ state.last_ai_time or 'Never' }}</strong>
            </div>
            <div>Last AI duration:
                <strong id="status-last-ai-duration">
                {{ state.last_ai_duration_seconds ~ ' sec'
                   if state.last_ai_duration_seconds is not none else 'N/A' }}
                </strong>
            </div>
        </div>
        <p id="status-error" class="bad" style="{{ '' if state.last_error else 'display:none;' }}">
            <strong>Last error:</strong> <span id="status-error-text">{{ state.last_error }}</span>
        </p>

        <form method="post" action="{{ url_for('toggle_watching_route') }}"
              style="margin:16px 0; max-width:260px;">
            <button id="watching-toggle-button" type="submit">
                {{ 'Stop Watching' if state.watching else 'Start Watching' }}
            </button>
        </form>

        <p class="muted">
            LAN address: http://{{ local_ip }}:{{ settings.web_port }}
        </p>
        <img id="latest-preview" class="preview" src="{{ url_for('latest_image') }}?t={{ cache_buster }}"
             alt="Latest camera screenshot">

        <div class="latest-ai">
            <div class="latest-ai-header">
                <strong>Most Recent AI Response</strong>
                <span id="latest-ai-trigger"
                      class="pill {{ 'good' if history and history[0].triggered else '' }}">
                    {{ 'TRIGGERED' if history and history[0].triggered else 'NOT TRIGGERED' }}
                </span>
                <span id="latest-ai-time" class="muted">
                    {{ history[0].timestamp if history else '' }}
                </span>
            </div>
            <div id="latest-ai-response"
                 class="latest-ai-response {{ 'bad' if history and history[0].error else '' }}">
                {% if history %}
                    {{ history[0].error if history[0].error else history[0].response }}
                {% else %}
                    No AI responses yet.
                {% endif %}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Preferences</h2>
        <form method="post" action="{{ url_for('save_preferences') }}">
            <div class="grid">
                <div>
                    <label>RTSP URL</label>
                    <input name="rtsp_url" value="{{ settings.rtsp_url }}" required>
                </div>

                <div>
                    <label>Screenshot Interval (seconds)</label>
                    <input type="number" min="0.1" step="0.1"
                           name="screenshot_interval_seconds"
                           value="{{ settings.screenshot_interval_seconds }}" required>
                </div>

                <div>
                    <label>Rolling Batch Size</label>
                    <input type="number" min="1" max="50"
                           name="batch_size"
                           value="{{ settings.batch_size }}" required>
                </div>

                <div>
                    <label>Ollama Model</label>
                    <select name="ollama_model">
                        {% for model in models %}
                            <option value="{{ model }}"
                                {{ 'selected' if model == settings.ollama_model else '' }}>
                                {{ model }}
                            </option>
                        {% endfor %}
                        {% if settings.ollama_model not in models %}
                            <option selected value="{{ settings.ollama_model }}">
                                {{ settings.ollama_model }}
                            </option>
                        {% endif %}
                    </select>
                </div>

                <div>
                    <label>Trigger Word</label>
                    <input name="trigger_word"
                           value="{{ settings.trigger_word }}" required>
                </div>

                <div>
                    <label>ntfy Server</label>
                    <input name="ntfy_server"
                           value="{{ settings.ntfy_server }}" required>
                </div>

                <div>
                    <label>ntfy Topic</label>
                    <input name="ntfy_topic"
                           value="{{ settings.ntfy_topic }}">
                </div>

                <div>
                    <label>ntfy Notification Title</label>
                    <input name="ntfy_title"
                           value="{{ settings.ntfy_title }}">
                </div>

                <div>
                    <label>Max Image Width</label>
                    <input type="number" min="64"
                           name="max_image_width"
                           value="{{ settings.max_image_width }}">
                </div>

                <div>
                    <label>Max Image Height</label>
                    <input type="number" min="64"
                           name="max_image_height"
                           value="{{ settings.max_image_height }}">
                </div>

                <div>
                    <label>JPEG Quality (1-100)</label>
                    <input type="number" min="1" max="100"
                           name="jpeg_quality"
                           value="{{ settings.jpeg_quality }}">
                </div>

                <div>
                    <label>Maximum History Entries</label>
                    <input type="number" min="1"
                           name="max_history_items"
                           value="{{ settings.max_history_items }}">
                </div>
            </div>

            <div style="margin-top:14px">
                <label>AI Prompt</label>
                <textarea name="prompt" required>{{ settings.prompt }}</textarea>
            </div>

            <div class="checkbox-row">
                <input type="checkbox"
                       name="send_latest_image_with_notification"
                       id="send_image"
                       {{ 'checked' if settings.send_latest_image_with_notification else '' }}>
                <label for="send_image" style="margin:0">
                    Attach latest screenshot to triggered ntfy notifications
                </label>
            </div>

            <div class="checkbox-row" style="margin-top:12px;">
                <input type="checkbox"
                       name="start_watching_on_startup"
                       id="start_watching_on_startup"
                       {{ 'checked' if settings.start_watching_on_startup else '' }}>
                <label for="start_watching_on_startup" style="margin:0">
                    Start watching automatically when the program starts
                </label>
            </div>

            <div style="margin-top:16px">
                <button type="submit">Save Preferences</button>
            </div>
        </form>
    </div>

    <div class="card">
        <h2>AI Response History</h2>

        <div class="actions">
            <form method="post" action="{{ url_for('run_test_notification') }}">
                <button type="submit">Send Test ntfy Notification</button>
            </form>
            <form method="post" action="{{ url_for('clear_history') }}"
                  onsubmit="return confirm('Clear all response history?');">
                <button type="submit">Clear History</button>
            </form>
        </div>

        <div id="history-list">
        {% if history %}
            {% for item in history %}
                <div class="history-item">
                    <strong>{{ item.timestamp }}</strong>

                    {% if item.triggered %}
                        <span class="pill good">TRIGGERED</span>
                    {% endif %}

                    {% if item.notification_sent %}
                        <span class="pill">ntfy sent</span>
                    {% endif %}

                    <div class="muted">
                        Model: {{ item.model or 'N/A' }} |
                        Batch: {{ item.batch_size or 'N/A' }} |
                        Duration: {{ item.duration_seconds or 'N/A' }} sec
                        {% if item.prompt_eval_count is defined and item.prompt_eval_count is not none %}
                            | Input tokens: {{ item.prompt_eval_count }}
                        {% endif %}
                        {% if item.eval_count is defined and item.eval_count is not none %}
                            | Output tokens: {{ item.eval_count }}
                        {% endif %}
                    </div>

                    {% if item.error %}
                        <div class="history-response bad">{{ item.error }}</div>
                    {% else %}
                        <div class="history-response">{{ item.response }}</div>
                    {% endif %}

                    {% if item.notification_error %}
                        <div class="bad">
                            Notification error: {{ item.notification_error }}
                        </div>
                    {% endif %}
                </div>
            {% endfor %}
        {% else %}
            <p class="muted">No AI responses yet.</p>
        {% endif %}
        </div>
    </div>
</div>
<script>
    let lastCaptureCount = {{ state.capture_count }};
    let lastAnalysisCount = {{ state.analysis_count }};

    function escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function updateStatus(state) {
        const watching = document.getElementById('status-watching');
        watching.textContent = state.watching ? 'Running' : 'Stopped';
        watching.className = state.watching ? 'good' : 'bad';

        const camera = document.getElementById('status-camera');
        camera.textContent = state.camera_connected ? 'Connected' : 'Disconnected';
        camera.className = state.camera_connected ? 'good' : 'bad';

        document.getElementById('status-captured').textContent = state.capture_count;
        document.getElementById('status-ai-calls').textContent = state.analysis_count;
        document.getElementById('status-ai-running').textContent = state.analysis_in_progress ? 'Yes' : 'No';
        document.getElementById('status-last-capture').textContent = state.last_capture_time || 'Never';
        document.getElementById('status-last-ai').textContent = state.last_ai_time || 'Never';
        document.getElementById('status-last-ai-duration').textContent =
            state.last_ai_duration_seconds == null ? 'N/A' : `${state.last_ai_duration_seconds} sec`;
        document.getElementById('watching-toggle-button').textContent =
            state.watching ? 'Stop Watching' : 'Start Watching';

        const errorBox = document.getElementById('status-error');
        document.getElementById('status-error-text').textContent = state.last_error || '';
        errorBox.style.display = state.last_error ? '' : 'none';
    }

    function updateLatestAiResponse(history) {
        const indicator = document.getElementById('latest-ai-trigger');
        const timestamp = document.getElementById('latest-ai-time');
        const response = document.getElementById('latest-ai-response');

        if (!history.length) {
            indicator.textContent = 'NOT TRIGGERED';
            indicator.className = 'pill';
            timestamp.textContent = '';
            response.textContent = 'No AI responses yet.';
            response.className = 'latest-ai-response';
            return;
        }

        const item = history[0];
        indicator.textContent = item.triggered ? 'TRIGGERED' : 'NOT TRIGGERED';
        indicator.className = item.triggered ? 'pill good' : 'pill';
        timestamp.textContent = item.timestamp || '';
        response.textContent = item.error || item.response || '';
        response.className = item.error
            ? 'latest-ai-response bad'
            : 'latest-ai-response';
    }

    function renderHistory(history) {
        updateLatestAiResponse(history);
        const container = document.getElementById('history-list');

        if (!history.length) {
            container.innerHTML = '<p class="muted">No AI responses yet.</p>';
            return;
        }

        container.innerHTML = history.map(item => {
            const triggered = item.triggered ? '<span class="pill good">TRIGGERED</span>' : '';
            const sent = item.notification_sent ? '<span class="pill">ntfy sent</span>' : '';
            const inputTokens = item.prompt_eval_count != null
                ? ` | Input tokens: ${escapeHtml(item.prompt_eval_count)}` : '';
            const outputTokens = item.eval_count != null
                ? ` | Output tokens: ${escapeHtml(item.eval_count)}` : '';
            const mainText = item.error
                ? `<div class="history-response bad">${escapeHtml(item.error)}</div>`
                : `<div class="history-response">${escapeHtml(item.response)}</div>`;
            const notificationError = item.notification_error
                ? `<div class="bad">Notification error: ${escapeHtml(item.notification_error)}</div>` : '';

            return `
                <div class="history-item">
                    <strong>${escapeHtml(item.timestamp)}</strong>
                    ${triggered}
                    ${sent}
                    <div class="muted">
                        Model: ${escapeHtml(item.model || 'N/A')} |
                        Batch: ${escapeHtml(item.batch_size || 'N/A')} |
                        Duration: ${escapeHtml(item.duration_seconds ?? 'N/A')} sec
                        ${inputTokens}${outputTokens}
                    </div>
                    ${mainText}
                    ${notificationError}
                </div>`;
        }).join('');
    }

    async function refreshHistory() {
        const response = await fetch('{{ url_for("api_history") }}', {cache: 'no-store'});
        if (!response.ok) return;
        renderHistory(await response.json());
    }

    async function refreshLiveData() {
        try {
            const response = await fetch('{{ url_for("api_status") }}', {cache: 'no-store'});
            if (!response.ok) return;

            const state = await response.json();
            updateStatus(state);

            if (state.capture_count !== lastCaptureCount) {
                lastCaptureCount = state.capture_count;
                document.getElementById('latest-preview').src =
                    '{{ url_for("latest_image") }}?t=' + Date.now();
            }

            if (state.analysis_count !== lastAnalysisCount) {
                lastAnalysisCount = state.analysis_count;
                await refreshHistory();
            }
        } catch (error) {
            console.debug('Live update failed:', error);
        }
    }

    setInterval(refreshLiveData, 1000);
</script>
</body>
</html>
"""


@app.get("/")
def index():
    current = get_settings_snapshot()
    state = get_runtime_state()
    models = get_ollama_models()

    with history_lock:
        history_snapshot = list(history)

    return render_template_string(
        PAGE_TEMPLATE,
        settings=current,
        state=state,
        models=models,
        history=history_snapshot,
        local_ip=get_local_ip(),
        cache_buster=int(time.time()),
    )


@app.post("/save")
def save_preferences():
    old = get_settings_snapshot()

    updated = dict(old)

    updated["rtsp_url"] = request.form.get("rtsp_url", "").strip()
    updated["screenshot_interval_seconds"] = max(
        0.1, float(request.form.get("screenshot_interval_seconds", 5))
    )
    updated["batch_size"] = max(
        1, int(request.form.get("batch_size", 4))
    )
    updated["ollama_model"] = request.form.get("ollama_model", "").strip()
    updated["prompt"] = request.form.get("prompt", "").strip()
    updated["trigger_word"] = request.form.get("trigger_word", "").strip()
    updated["ntfy_server"] = request.form.get("ntfy_server", "").strip()
    updated["ntfy_topic"] = request.form.get("ntfy_topic", "").strip()
    updated["ntfy_title"] = request.form.get("ntfy_title", "").strip()
    updated["max_image_width"] = max(
        64, int(request.form.get("max_image_width", 1280))
    )
    updated["max_image_height"] = max(
        64, int(request.form.get("max_image_height", 720))
    )
    updated["jpeg_quality"] = min(
        100, max(1, int(request.form.get("jpeg_quality", 85)))
    )
    updated["max_history_items"] = max(
        1, int(request.form.get("max_history_items", 250))
    )
    updated["send_latest_image_with_notification"] = (
        "send_latest_image_with_notification" in request.form
    )
    updated["start_watching_on_startup"] = (
        "start_watching_on_startup" in request.form
    )

    with settings_lock:
        settings.clear()
        settings.update(updated)

    save_settings()

    # Signal the camera worker if a camera-sensitive preference changed.
    reconnect_keys = {
        "rtsp_url",
        "batch_size",
        "max_image_width",
        "max_image_height",
        "jpeg_quality",
    }

    if any(old.get(key) != updated.get(key) for key in reconnect_keys):
        settings_changed_event.set()

    return redirect(url_for("index"))


@app.post("/toggle-watching")
def toggle_watching_route():
    toggle_watching()
    return redirect(url_for("index"))


@app.get("/latest.jpg")
def latest_image():
    with frames_lock:
        image = latest_frame_jpeg

    if image is None:
        placeholder = Image.new("RGB", (640, 360), (30, 30, 30))
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 20), "Waiting for RTSP camera...", fill=(230, 230, 230))
        buffer = io.BytesIO()
        placeholder.save(buffer, format="JPEG")
        image = buffer.getvalue()

    return app.response_class(image, mimetype="image/jpeg")


@app.post("/test-ntfy")
def run_test_notification():
    try:
        send_ntfy_notification(
            "Test notification from Local AI Webcam Monitor.",
            None,
        )
        update_runtime_state(last_error="")
    except Exception as exc:
        update_runtime_state(last_error=f"Test ntfy notification failed: {exc}")

    return redirect(url_for("index"))


@app.post("/clear-history")
def clear_history():
    global history

    with history_lock:
        history = []
        save_json(HISTORY_FILE, history)

    return redirect(url_for("index"))


@app.get("/api/status")
def api_status():
    return jsonify(get_runtime_state())


@app.get("/api/history")
def api_history():
    with history_lock:
        return jsonify(list(history))


def web_server_worker():
    current = get_settings_snapshot()

    # host=0.0.0.0 makes the page accessible to other devices on the LAN.
    app.run(
        host="0.0.0.0",
        port=int(current["web_port"]),
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# ---------------------------------------------------------------------------
# System tray
# ---------------------------------------------------------------------------

def create_tray_image():
    image = Image.new("RGB", (64, 64), (30, 41, 59))
    draw = ImageDraw.Draw(image)

    # Simple camera body.
    draw.rounded_rectangle((10, 18, 54, 48), radius=7, fill=(59, 130, 246))
    draw.ellipse((24, 23, 45, 44), fill=(17, 24, 39))
    draw.ellipse((29, 28, 40, 39), fill=(147, 197, 253))
    draw.rectangle((17, 12, 31, 20), fill=(59, 130, 246))

    return image


def open_web_ui(icon=None, item=None):
    current = get_settings_snapshot()
    webbrowser.open(f"http://127.0.0.1:{current['web_port']}")


def toggle_watching_from_tray(icon, item):
    toggle_watching()

    # Refresh callable menu text immediately on platforms that support it.
    try:
        icon.update_menu()
    except Exception:
        pass


def watching_menu_text(item):
    return "Stop Watching" if watching_event.is_set() else "Start Watching"


def quit_app(icon, item):
    print("Shutting down...")
    stop_event.set()
    watching_event.set()
    icon.stop()


def tray_worker():
    icon = pystray.Icon(
        "rtsp_ollama_monitor",
        create_tray_image(),
        "Local AI Webcam Monitor",
        menu=pystray.Menu(
            pystray.MenuItem("Open Settings / History", open_web_ui),
            pystray.MenuItem(watching_menu_text, toggle_watching_from_tray),
            pystray.MenuItem("Quit", quit_app),
        ),
    )

    icon.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_settings()
    load_history()

    current = get_settings_snapshot()

    # Watching is stopped by default. Only start automatically when the
    # saved startup preference explicitly enables it. This setting is
    # applied only at program launch; the tray/web controls manage the
    # current session after that.
    set_watching(bool(current.get("start_watching_on_startup", False)))

    print("Local AI Webcam Monitor")
    print("--------------------")
    print(f"Local web UI: http://127.0.0.1:{current['web_port']}")
    print(f"LAN web UI:   http://{get_local_ip()}:{current['web_port']}")
    print()
    print("Close the app using the system tray icon.")

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True,
        name="RTSPCapture",
    )
    camera_thread.start()

    web_thread = threading.Thread(
        target=web_server_worker,
        daemon=True,
        name="WebServer",
    )
    web_thread.start()

    # pystray generally behaves most reliably when its event loop is run
    # on the main thread.
    tray_worker()

    stop_event.set()


if __name__ == "__main__":
    main()