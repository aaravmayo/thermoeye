import os, time, threading, json, secrets
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import cv2

from engine import ThermoEye, MQTTSender, open_capture, MQTT_BROKER, MQTT_PORT, SITE_ID

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)

# Simple user database (use proper DB in production)
USERS_FILE = os.path.join(DATA_FOLDER, "users.json")
STREAMS_FILE = os.path.join(DATA_FOLDER, "streams.json")
ALERTS_FILE = os.path.join(DATA_FOLDER, "alerts.json")

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            pass
    return default or {}

def save_json(filepath, data):
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DEBUG] Saved to {filepath}: {data}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {filepath}: {e}")
        return False

# Initialize default admin user if file doesn't exist
if not os.path.exists(USERS_FILE):
    default_users = {
        "admin": {
            "password": generate_password_hash("admin123"),
            "role": "admin",
            "created": datetime.now().isoformat()
        }
    }
    save_json(USERS_FILE, default_users)

streams_config = load_json(STREAMS_FILE, {})
alerts_history_raw = load_json(ALERTS_FILE, [])
# Ensure alerts_history is always a list
alerts_history = alerts_history_raw if isinstance(alerts_history_raw, list) else []

# Global processing state
state = {
    "source": None,
    "cap": None,
    "running": False,
    "frame_id": 0,
    "frame": None,
    "lock": threading.Lock(),
    "engine": None,
    "mqtt": None,
    "active_stream_id": None,
    "stats": {
        "peak_density": 0.0,
        "avg_density": 0.0,
        "total_alerts": 0,
        "uptime_start": None
    }
}

def video_loop():
    density_samples = []
    while state["running"]:
        if state["cap"] is None:
            time.sleep(0.05)
            continue
        ret, frame = state["cap"].read()
        if not ret:
            if isinstance(state["source"], str):
                if state["source"].lower().endswith(tuple(ALLOWED_EXT)):
                    state["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if state["engine"]:
                        state["engine"].reset()
                    continue
                else:
                    try:
                        state["cap"].release()
                    except:
                        pass
                    time.sleep(0.25)
                    state["cap"] = open_capture(state["source"])
                    continue
            else:
                break
        
        state["frame_id"] += 1
        processed = state["engine"].process_frame(frame, state["frame_id"])
        
        # Update stats
        density = state["engine"].current_density
        density_samples.append(density)
        if len(density_samples) > 100:
            density_samples.pop(0)
        
        state["stats"]["peak_density"] = max(state["stats"]["peak_density"], density)
        state["stats"]["avg_density"] = sum(density_samples) / len(density_samples)
        
        # Log alerts
        if state["engine"].alert_active and state["engine"].last_alert_time:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "density": round(density, 3),
                "people": state["engine"].estimated_people,
                "stream_id": state["active_stream_id"]
            }
            # Ensure alerts_history is a list before appending
            if not isinstance(alerts_history, list):
                alerts_history.clear()
                alerts_history.extend([])
            alerts_history.append(alert)
            if len(alerts_history) > 1000:
                alerts_history.pop(0)
            state["stats"]["total_alerts"] = len(alerts_history)
            # Save only last 100 alerts to file
            try:
                save_json(ALERTS_FILE, alerts_history[-100:])
            except:
                pass
        
        with state["lock"]:
            state["frame"] = processed
        time.sleep(0.01)

    if state["cap"] is not None:
        try:
            state["cap"].release()
        except:
            pass

def gen_mjpeg():
    while True:
        with state["lock"]:
            frame = state["frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        jpg_bytes = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")

# Authentication decorator
def login_required(f):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        # Load users fresh from file
        users = load_json(USERS_FILE, {})
        if username in users and check_password_hash(users[username]["password"], password):
            session["user"] = username
            session["role"] = users[username].get("role", "user")
            session.permanent = True
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "error")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        
        if not username or not password:
            flash("Username and password required", "error")
        elif password != confirm:
            flash("Passwords do not match", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters", "error")
        else:
            # Load users fresh from file
            users = load_json(USERS_FILE, {})
            if username in users:
                flash("Username already exists", "error")
            else:
                users[username] = {
                    "password": generate_password_hash(password),
                    "role": "user",
                    "created": datetime.now().isoformat()
                }
                # Save updated users
                save_json(USERS_FILE, users)
                flash("Account created successfully! Please login.", "success")
                return redirect(url_for("login"))
    
    return render_template("register.html")

@app.route("/")
@login_required
def dashboard():
    thr = state["engine"].density_threshold if state["engine"] else 0.45
    op = state["engine"].heatmap_opacity if state["engine"] else 0.70
    
    uptime = None
    if state["stats"]["uptime_start"]:
        delta = datetime.now() - state["stats"]["uptime_start"]
        uptime = str(delta).split('.')[0]
    
    # Ensure stats have valid values
    stats = {
        "peak_density": state["stats"].get("peak_density", 0.0),
        "avg_density": state["stats"].get("avg_density", 0.0),
        "total_alerts": state["stats"].get("total_alerts", 0),
        "uptime_start": state["stats"].get("uptime_start")
    }
    
    return render_template("dashboard.html",
                         running=state["running"],
                         threshold=int(thr * 100),
                         opacity=int(op * 100),
                         source=str(state["source"]) if state["source"] is not None else "",
                         streams=streams_config,
                         active_stream=state["active_stream_id"],
                         stats=stats,
                         uptime=uptime,
                         user=session.get("user"),
                         role=session.get("role"))

@app.route("/streams")
@login_required
def streams_page():
    return render_template("streams.html", streams=streams_config, user=session.get("user"))

@app.route("/streams/add", methods=["POST"])
@login_required
def add_stream():
    name = request.form.get("name", "").strip()
    url = request.form.get("url", "").strip()
    stream_type = request.form.get("type", "rtsp")
    
    if not name or not url:
        flash("Name and URL required", "error")
        return redirect(url_for("streams_page"))
    
    stream_id = f"stream_{int(time.time())}"
    streams_config[stream_id] = {
        "name": name,
        "url": url,
        "type": stream_type,
        "created": datetime.now().isoformat(),
        "owner": session.get("user")
    }
    save_json(STREAMS_FILE, streams_config)
    flash("Stream added successfully", "success")
    return redirect(url_for("streams_page"))

@app.route("/streams/delete/<stream_id>", methods=["POST"])
@login_required
def delete_stream(stream_id):
    if stream_id in streams_config:
        if session.get("role") == "admin" or streams_config[stream_id].get("owner") == session.get("user"):
            del streams_config[stream_id]
            save_json(STREAMS_FILE, streams_config)
            flash("Stream deleted", "success")
        else:
            flash("Permission denied", "error")
    return redirect(url_for("streams_page"))

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
@login_required
def start():
    source_type = request.form.get("source_type")
    src = None
    stream_id = None
    
    if source_type == "webcam":
        src = 0
    elif source_type == "stream":
        stream_id = request.form.get("stream_id")
        if stream_id and stream_id in streams_config:
            src = streams_config[stream_id]["url"]
        else:
            flash("Invalid stream selected", "error")
            return redirect(url_for("dashboard"))
    elif source_type == "url":
        url = request.form.get("stream_url", "").strip()
        if not url:
            flash("Please provide a stream URL", "error")
            return redirect(url_for("dashboard"))
        src = url
    elif source_type == "file":
        f = request.files.get("video_file")
        if not f:
            flash("Please upload a video file", "error")
            return redirect(url_for("dashboard"))
        name = secure_filename(f.filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            flash("Unsupported file type", "error")
            return redirect(url_for("dashboard"))
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], name)
        f.save(filepath)
        src = filepath
    else:
        flash("Invalid source type", "error")
        return redirect(url_for("dashboard"))

    try:
        thr = float(request.form.get("threshold", "45")) / 100.0
        op = float(request.form.get("opacity", "70")) / 100.0
        thr = max(0.0, min(1.0, thr))
        op = max(0.3, min(1.0, op))
    except:
        thr, op = 0.45, 0.70

    mqtt = MQTTSender(MQTT_BROKER, MQTT_PORT, client_id=f"thermo-web-{SITE_ID}-{session.get('user')}")
    engine = ThermoEye(density_threshold=thr, heatmap_opacity=op, alert_cooldown=3,
                      site_id=SITE_ID, mqtt_sender=mqtt)

    cap = open_capture(src)
    if not cap.isOpened():
        mqtt.close()
        flash("Cannot open video source", "error")
        return redirect(url_for("dashboard"))

    state["running"] = False
    time.sleep(0.1)

    state["source"] = src
    state["cap"] = cap
    state["engine"] = engine
    state["mqtt"] = mqtt
    state["frame_id"] = 0
    state["frame"] = None
    state["active_stream_id"] = stream_id
    state["running"] = True
    state["stats"]["uptime_start"] = datetime.now()
    state["stats"]["peak_density"] = 0.0
    state["stats"]["avg_density"] = 0.0

    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    flash("Stream started successfully", "success")
    return redirect(url_for("dashboard"))

@app.route("/stop", methods=["POST"])
@login_required
def stop():
    state["running"] = False
    state["stats"]["uptime_start"] = None
    if state["mqtt"]:
        try:
            state["mqtt"].close()
        except:
            pass
    flash("Stream stopped", "info")
    return redirect(url_for("dashboard"))

@app.route("/set_params", methods=["POST"])
@login_required
def set_params():
    if state["engine"]:
        try:
            thr = float(request.form.get("threshold", "45")) / 100.0
            op = float(request.form.get("opacity", "70")) / 100.0
            state["engine"].density_threshold = max(0.0, min(1.0, thr))
            state["engine"].heatmap_opacity = max(0.3, min(1.0, op))
            flash("Parameters updated", "success")
        except:
            flash("Invalid parameters", "error")
    return redirect(url_for("dashboard"))

@app.route("/api/stats")
@login_required
def api_stats():
    density = state["engine"].current_density if state["engine"] else 0.0
    people = state["engine"].estimated_people if state["engine"] else 0
    
    return jsonify({
        "running": state["running"],
        "density": round(density, 3),
        "people": people,
        "peak_density": round(state["stats"]["peak_density"], 3),
        "avg_density": round(state["stats"]["avg_density"], 3),
        "total_alerts": state["stats"]["total_alerts"],
        "alert_active": state["engine"].alert_active if state["engine"] else False
    })

@app.route("/alerts")
@login_required
def alerts_page():
    # Ensure alerts_history is a list and get last 50 in reverse
    alerts_list = alerts_history if isinstance(alerts_history, list) else []
    recent = alerts_list[-50:][::-1] if alerts_list else []
    return render_template("alerts.html", alerts=recent, user=session.get("user"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)