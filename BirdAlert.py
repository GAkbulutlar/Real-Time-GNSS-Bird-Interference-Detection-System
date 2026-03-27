import os
import cv2
import time
import smtplib
import ssl
import traceback
import numpy as np
import requests
import requests.adapters
import urllib3
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv

from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

import sys
print(sys.executable)
print(sys.path)
# ----------------------------
# CONFIG (LOADED FROM ENVIRONMENT VARIABLES)
# ----------------------------
# For RTSP:  "rtsp://ip:554/stream_path"  (set CAMERA_USER/CAMERA_PASS below)
# For HTTP:  "http://ip:port/snap.jpeg"
# For HTTPS: "https://ip:port/snap.jpeg"  (self-signed cert is OK)
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "https://62.154.166.188:3333")
CAMERA_USER = os.getenv("CAMERA_USER", "dev")
CAMERA_PASS = os.getenv("CAMERA_PASS", "")
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")
TARGET_CLASS_NAMES = {os.getenv("TARGET_CLASS_NAMES", "bird")}
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))

CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "2"))
REQUIRED_HITS = int(os.getenv("REQUIRED_HITS", "3"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "300"))

SAVE_DIR = os.getenv("SAVE_DIR", "alerts")

SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _smtp_send(msg):
    """Send via internal relay (port 25, no auth/TLS) or SSL (port 465)."""
    if SMTP_PORT == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            if SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    else:  # port 25 internal relay — no auth, no TLS required
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


def send_email_with_image(subject: str, body: str, image_path: str):
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)
    with open(image_path, "rb") as f:
        img_data = f.read()
    msg.add_attachment(
        img_data,
        maintype="image",
        subtype="jpeg",
        filename=os.path.basename(image_path),
    )
    _smtp_send(msg)


def send_test_email():
    """Send a plain-text test email to verify SMTP settings."""
    print(f"[TEST] Sending test email to {EMAIL_TO} via {SMTP_SERVER}:{SMTP_PORT}...")
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = "Bird Alert - Test Email"
    msg.set_content(
        "This is a test email from the Bird Alert monitoring script.\n"
        f"SMTP server : {SMTP_SERVER}:{SMTP_PORT}\n"
        f"From        : {EMAIL_FROM}\n"
        f"To          : {EMAIL_TO}\n"
        f"Time        : {datetime.now()}\n"
    )
    _smtp_send(msg)
    print("[TEST] Test email sent successfully.")


def _is_http_source(source: str) -> bool:
    return source.lower().startswith("http://") or source.lower().startswith("https://")


class _LegacySSLAdapter(requests.adapters.HTTPAdapter):
    """Allows connecting to cameras with old/weak TLS configurations."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


def _make_session():
    s = requests.Session()
    s.mount("https://", _LegacySSLAdapter())
    return s


def connect_camera(source):
    if _is_http_source(source):
        return None  # HTTP mode uses requests, no VideoCapture needed
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame_http(source: str):
    """Fetch a single JPEG snapshot from an HTTP/HTTPS camera."""
    try:
        auth = (CAMERA_USER, CAMERA_PASS) if CAMERA_USER else None
        session = _make_session()
        resp = session.get(source, auth=auth, verify=False, timeout=5)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        return True, frame
    except Exception as e:
        print(f"[WARN] HTTP frame fetch failed: {e}")
        return False, None


# Common INSTAR / generic IP-camera snapshot paths
_SNAPSHOT_CANDIDATES = [
    "/tmpfs/snap.jpg",
    "/tmpfs/auto.jpg",
    "/snap.cgi?chn=0",
    "/cgi-bin/hi3510/snap.cgi?chn=0",
    "/snapshot.cgi",
    "/cgi-bin/snapshot.cgi",
    "/jpg/image.jpg",
    "/image.jpg",
    "/mjpg/video.mjpg",
    "/video.cgi",
]


def probe_snapshot_path(base: str) -> str:
    """Try candidate paths and return the first one that returns a decodable image."""
    print("[INFO] Probing camera for correct snapshot path...")
    auth = (CAMERA_USER, CAMERA_PASS) if CAMERA_USER else None
    session = _make_session()
    for path in _SNAPSHOT_CANDIDATES:
        url = base.rstrip("/") + path
        try:
            resp = session.get(url, auth=auth, verify=False, timeout=5)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    print(f"[INFO] Snapshot path found: {url}")
                    return url
                else:
                    print(f"[INFO]   {path} -> HTTP 200 but not an image (HTML login page?)")
            else:
                print(f"[INFO]   {path} -> HTTP {resp.status_code}")
        except Exception as e:
            print(f"[INFO]   {path} -> {e}")
    print("[WARN] Could not auto-detect snapshot path. Set CAMERA_SOURCE manually.")
    return base


def detect_bird(model, frame):
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    annotated = r.plot()

    found = False
    best_conf = 0.0

    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names.get(cls_id, str(cls_id)).lower()
            if cls_name in TARGET_CLASS_NAMES:
                found = True
                best_conf = max(best_conf, conf)

    return found, annotated, best_conf


def main():
    ensure_dir(SAVE_DIR)
    model = YOLO(MODEL_NAME)

    consecutive_hits = 0
    last_alert_ts = 0
    last_check = 0
    cap = None

    print("[INFO] Bird monitor started (24/7 mode). Press Ctrl+C to stop.")

    http_mode = _is_http_source(CAMERA_SOURCE)
    active_source = CAMERA_SOURCE
    if http_mode:
        active_source = probe_snapshot_path(CAMERA_SOURCE)

    while True:
        try:
            if http_mode:
                ok, frame = read_frame_http(active_source)
                if not ok or frame is None:
                    print("[WARN] Failed to read HTTP frame. Retrying in 5s...")
                    time.sleep(5)
                    continue
            else:
                if cap is None or not cap.isOpened():
                    print("[WARN] Camera not connected. Retrying in 5 seconds...")
                    if cap is not None:
                        cap.release()
                    time.sleep(5)
                    cap = connect_camera(CAMERA_SOURCE)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Failed to read frame. Reconnecting...")
                    cap.release()
                    cap = None
                    continue

            now = time.time()
            if now - last_check < CHECK_INTERVAL_SEC:
                time.sleep(0.05)
                continue
            last_check = now

            found, annotated, best_conf = detect_bird(model, frame)

            if found:
                consecutive_hits += 1
                print(f"[INFO] Bird detected {consecutive_hits}/{REQUIRED_HITS}, conf={best_conf:.2f}")
            else:
                if consecutive_hits > 0:
                    print("[INFO] No bird this cycle. Resetting counter.")
                consecutive_hits = 0

            if consecutive_hits >= REQUIRED_HITS and (now - last_alert_ts) >= ALERT_COOLDOWN_SEC:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(SAVE_DIR, f"bird_{ts}.jpg")
                cv2.imwrite(image_path, annotated)

                subject = f"Bird Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                body = (
                    "Bird detected near GNSS antennas.\n"
                    f"Confidence: {best_conf:.2f}\n"
                    f"Time: {datetime.now()}\n"
                )

                try:
                    send_email_with_image(subject, body, image_path)
                    print(f"[ALERT] Email sent: {image_path}")
                    last_alert_ts = now
                    consecutive_hits = 0
                except Exception as e:
                    print(f"[ERROR] Email send failed: {e}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
            break
        except Exception:
            print("[ERROR] Unexpected runtime error:")
            traceback.print_exc()
            time.sleep(3)

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    import sys as _sys
    if "--test-email" in _sys.argv:
        try:
            send_test_email()
        except Exception:
            traceback.print_exc()
            input("Press Enter to exit...")
    else:
        try:
            main()
        except Exception:
            print("[FATAL] Script crashed:")
            traceback.print_exc()
            input("Press Enter to exit...")