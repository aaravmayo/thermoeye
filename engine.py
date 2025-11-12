# engine.py
import cv2
import numpy as np
import time
from collections import deque
import json, socket
import paho.mqtt.client as mqtt

MQTT_ENABLED = True
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_QOS = 1
MQTT_TOPIC_BASE = "thermo/alerts"
SITE_ID = socket.gethostname().replace(".", "-") or "site-001"

class MQTTSender:
    def __init__(self, broker, port, client_id):
        self.client = None
        if MQTT_ENABLED and mqtt is not None:
            self.client = mqtt.Client(client_id=client_id, clean_session=False)
            try:
                self.client.connect(broker, port, keepalive=60)
                self.client.loop_start()
                print(f"[MQTT] Connected to {broker}:{port} as {client_id}")
            except Exception as e:
                print("[MQTT] connect failed:", e)
                self.client = None
        else:
            print("[MQTT] Disabled or paho-mqtt not installed")

    def publish(self, topic, payload_dict, qos=MQTT_QOS, retain=False):
        if not self.client:
            return
        try:
            msg = json.dumps(payload_dict, separators=(',', ':'))
            self.client.publish(topic, msg, qos=qos, retain=retain)
        except Exception as e:
            print("[MQTT] publish failed:", e)

    def close(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                print("[MQTT] Disconnected")
            except:
                pass

class ThermoEye:
    def __init__(self, density_threshold=0.45, heatmap_opacity=0.70,
                 alert_cooldown=3, site_id=SITE_ID, mqtt_sender=None):
        self.density_threshold = density_threshold
        self.heatmap_opacity = heatmap_opacity
        self.alert_cooldown = alert_cooldown
        self.site_id = site_id
        self.mqtt = mqtt_sender

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self.density_history = deque(maxlen=15)
        self.grid_size = 25
        self.blur_kernel_size = 51
        self.estimated_people = 0
        self.current_density = 0.0
        self.prev_gray = None
        self.last_alert_time = 0
        self.alert_active = False

    def reset(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self.prev_gray = None
        self.density_history.clear()
        self.alert_active = False

    def detect_camera_pan(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        change_percentage = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
        self.prev_gray = gray
        return change_percentage > 0.6

    def maybe_send_alert(self, frame_id):
        now = time.time()
        over = self.current_density > self.density_threshold

        if over and ((not self.alert_active) or (now - self.last_alert_time >= self.alert_cooldown)):
            payload = {
                "type": "alert",
                "site_id": self.site_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "density": round(float(self.current_density), 3),
                "threshold": round(float(self.density_threshold), 3),
                "estimated_people": int(self.estimated_people),
                "frame_id": int(frame_id),
                "status": "HIGH"
            }
            if self.mqtt:
                self.mqtt.publish(f"{MQTT_TOPIC_BASE}/{self.site_id}", payload)
            self.last_alert_time = now
            self.alert_active = True

        if (not over) and self.alert_active and (now - self.last_alert_time >= 2):
            payload = {
                "type": "alert",
                "site_id": self.site_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "density": round(float(self.current_density), 3),
                "threshold": round(float(self.density_threshold), 3),
                "estimated_people": int(self.estimated_people),
                "frame_id": int(frame_id),
                "status": "CLEAR"
            }
            if self.mqtt:
                self.mqtt.publish(f"{MQTT_TOPIC_BASE}/{self.site_id}", payload)
            self.last_alert_time = now
            self.alert_active = False

    def process_frame(self, frame, frame_id):
        extreme_pan = self.detect_camera_pan(frame)

        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        learning_rate = 0.03 if extreme_pan else 0.01
        fg_mask = self.bg_subtractor.apply(processed, learningRate=learning_rate)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.dilate(fg_mask, kernel_large, iterations=1)

        h, w = fg_mask.shape
        gh = h // self.grid_size
        gw = w // self.grid_size
        density_map = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                y1, y2 = i*self.grid_size, (i+1)*self.grid_size
                x1, x2 = j*self.grid_size, (j+1)*self.grid_size
                cell = fg_mask[y1:y2, x1:x2]
                density_map[i, j] = np.sum(cell > 0) / (self.grid_size ** 2)
        overall_density = float(np.mean(density_map))
        if extreme_pan:
            overall_density *= 0.5

        self.density_history.append(overall_density)
        self.current_density = float(np.mean(self.density_history))

        total_fg = int(np.sum(fg_mask > 0))
        self.estimated_people = max(1, total_fg // 800)
        # Divide by 3.5 to adjust for overcounting
        self.estimated_people = max(1, int(self.estimated_people / 3.5))
        if extreme_pan:
            self.estimated_people = int(self.estimated_people * 0.5)

        normalized = (density_map * 255).astype(np.uint8)
        resized = cv2.resize(normalized, (w, h))
        blurred = cv2.GaussianBlur(resized, (self.blur_kernel_size, self.blur_kernel_size), 0)
        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        display = cv2.addWeighted(frame, 1.0 - self.heatmap_opacity, heatmap, self.heatmap_opacity, 0)

        # Overlay
        overlay = display.copy()
        panel_h = 190 if extreme_pan else 170
        cv2.rectangle(overlay, (10, 10), (420, panel_h), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.65, overlay, 0.35, 0)

        cv2.putText(display, "THERMO EYE - Crowd Monitor", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        color = (0, 255, 0) if self.current_density < self.density_threshold else (0, 100, 255)
        cv2.putText(display, f"Density: {self.current_density:.1%}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Threshold: {self.density_threshold:.0%}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "Mode: Motion Analysis", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        if extreme_pan:
            cv2.putText(display, "⚠ Fast Panning...", (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.current_density > self.density_threshold:
            cv2.rectangle(display, (0, h - 70), (w, h), (0, 0, 200), -1)
            cv2.putText(display, "⚠ HIGH CROWD DENSITY!", (w//2 - 200, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        legend_x, legend_y = w - 180, 20
        gradient = np.linspace(0, 255, 160, dtype=np.uint8)
        gradient = np.tile(gradient, (30, 1))
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        display[legend_y:legend_y+30, legend_x:legend_x+160] = gradient_colored
        cv2.putText(display, "Low",  (legend_x,        legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(display, "High", (legend_x + 125,  legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        self.maybe_send_alert(frame_id)
        return display

def open_capture(source):
    if isinstance(source, str) and (
        source.startswith(("rtsp://", "rtmp://", "http://", "https://")) or source.endswith(".m3u8")
    ):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)
