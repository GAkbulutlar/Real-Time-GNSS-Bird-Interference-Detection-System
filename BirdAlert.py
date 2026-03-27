# ============================================================================
# Real-Time GNSS Bird Interference Detection System
# ============================================================================
# This script monitors a camera feed for bird detection using YOLOv8 AI model.
# When birds are detected, it sends alerts via email with annotated images.
# ============================================================================

# Standard library imports
import os              # File and directory operations
import cv2             # OpenCV - Computer vision library for image processing
import time            # Time-related functions
import smtplib         # SMTP protocol for sending emails
import ssl             # SSL/TLS encryption for secure connections
import traceback       # Error traceback printing
from datetime import datetime  # Date and time handling
from email.message import EmailMessage  # Email message formatting

# Third-party imports
import numpy as np     # NumPy - Numerical computing library
import requests        # HTTP library for camera communication
import requests.adapters  # HTTP adapter customization
import urllib3         # HTTP client for URL handling
import warnings       # Warning control

# Suppress SSL certificate warnings (for self-signed camera certificates)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# YOLOv8 - AI model for object detection
from ultralytics import YOLO

# Debug: Print Python executable and path
import sys
print(sys.executable)
print(sys.path)
# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES
# ============================================================================
# Camera Source Options:
#   - RTSP stream:  "rtsp://ip:554/stream_path"  (requires CAMERA_USER/CAMERA_PASS)
#   - HTTP request: "http://ip:port/snap.jpeg"  (supports basic auth)
#   - HTTPS request: "https://ip:port/snap.jpeg"  (supports self-signed certificates)
CAMERA_SOURCE = "https://Your camera IP or hostname here"  # Example: "http://192.168.1.100:8080/snap.jpeg"
CAMERA_USER = "Credentials if needed, else leave empty"   # Camera username for authentication
CAMERA_PASS = "Credentials if needed, else leave empty"   # Camera password for authentication

# AI Model Configuration
# Available models: yolov8n.pt (nano - fastest), yolov8s.pt (small), yolov8m.pt (medium - balanced), yolov8l.pt (large - most accurate)
MODEL_NAME = "yolov8n.pt"  # Using nano model for speed

# Detection Settings
TARGET_CLASS_NAMES = {"bird"}     # Only alert on birds (YOLOv8 trained to detect: bird, person, car, etc.)
CONF_THRESHOLD = 0.75             # Confidence threshold: 0.0-1.0 (higher = stricter, fewer false positives)

# Alert Logic Settings
CHECK_INTERVAL_SEC = 2            # How often to check camera feed (in seconds)
REQUIRED_HITS = 3                 # Number of consecutive detections before triggering alert
ALERT_COOLDOWN_SEC = 300          # Wait time between alerts (5 minutes) to prevent spam

# Storage Settings
SAVE_DIR = "alerts"               # Directory where detected bird images are saved

# Email/SMTP Configuration (for sending alerts)
SMTP_SERVER = "your SMTP server here"  # Example: "smtp.gmail.com" or "smtp.trimble.com"
SMTP_PORT = "your SMTP port here"      # Port: 25 (plain), 587 (TLS), 465 (SSL)
SMTP_USER = "SMTP username here (if required, else leave empty)"      # Username for SMTP authentication
SMTP_PASSWORD = ""                     # Password for SMTP authentication
EMAIL_FROM = SMTP_USER              # Sender email address (usually same as SMTP_USER)
EMAIL_TO = "your email here"        # Recipient email address for alerts


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: str):
    """Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _smtp_send(msg):
    """Send email via SMTP server with appropriate security (SSL or plain).
    
    Args:
        msg (EmailMessage): Email message object to send
        
    Note:
        - Port 465: Uses SMTP_SSL (SSL/TLS encryption from start)
        - Port 25/587: Uses plain SMTP (may require STARTTLS)
    """
    if SMTP_PORT == 465:  # SSL port - secure connection from the start
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            if SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    else:  # Port 25 - internal relay (no auth/TLS) or port 587 (TLS)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

def send_email_with_image(subject: str, body: str, image_path: str):
    """Send email alert with bird detection image attached.
    
    Args:
        subject (str): Email subject line
        body (str): Email body text
        image_path (str): Path to the annotated bird detection image to attach
    """
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)
    
    # Attach the bird detection image
    with open(image_path, "rb") as f:
        img_data = f.read()
    msg.add_attachment(
        img_data,
        maintype="image",
        subtype="jpeg",
        filename=os.path.basename(image_path),
    )
    
    # Send via SMTP
    _smtp_send(msg)


def send_test_email():
    """Test SMTP configuration by sending a test email.
    
    Useful for verifying:
    - SMTP server connectivity
    - Email credentials
    - Email address configuration
    """
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


# ============================================================================
# CAMERA CONNECTION FUNCTIONS
# ============================================================================

def _is_http_source(source: str) -> bool:
    """Check if camera source is HTTP/HTTPS (snapshot-based) vs RTSP/file.
    
    Args:
        source (str): Camera source URL or path
        
    Returns:
        bool: True if HTTP/HTTPS source, False otherwise (RTSP, file, etc.)
    """
    return source.lower().startswith("http://") or source.lower().startswith("https://")


class _LegacySSLAdapter(requests.adapters.HTTPAdapter):
    """Custom HTTPS adapter for connecting to IP cameras with weak/old SSL certificates.
    
    Many IP cameras use self-signed or outdated certificates. This adapter:
    - Disables hostname verification
    - Ignores certificate validity
    - Uses weak cipher suites for compatibility
    - Enables legacy SSL connections
    
    WARNING: This reduces security. Only use in trusted network environments.
    """
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False                          # Don't verify hostname matches certificate
        ctx.verify_mode = ssl.CERT_NONE                    # Don't verify certificate authenticity
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")            # Allow weak ciphers
        ctx.options |= 0x4                                 # Enable legacy mode (OP_LEGACY_SERVER_CONNECT)
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


def _make_session():
    """Create HTTP session with legacy SSL adapter for camera communication.
    
    Returns:
        requests.Session: Session configured for weak SSL certificates
    """
    s = requests.Session()
    s.mount("https://", _LegacySSLAdapter())
    return s


def connect_camera(source):
    """Connect to camera via RTSP stream or file.
    
    Args:
        source (str): RTSP URL or file path (HTTP/HTTPS handled separately)
        
    Returns:
        cv2.VideoCapture: Video capture object, or None if HTTP source
    """
    if _is_http_source(source):
        return None  # HTTP/HTTPS use requests library, not VideoCapture
    
    # RTSP or file source
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
    return cap


def read_frame_http(source: str):
    """Fetch a single JPEG snapshot from an HTTP/HTTPS camera.
    
    Args:
        source (str): HTTP/HTTPS URL of camera snapshot
        
    Returns:
        tuple: (success: bool, frame: numpy.ndarray or None)
    """
    try:
        # Setup basic authentication if credentials provided
        auth = (CAMERA_USER, CAMERA_PASS) if CAMERA_USER else None
        session = _make_session()
        
        # Fetch image from camera
        resp = session.get(source, auth=auth, verify=False, timeout=5)
        resp.raise_for_status()  # Raise exception on HTTP errors
        
        # Decode JPEG image from response
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return False, None
        return True, frame
    except Exception as e:
        print(f"[WARN] HTTP frame fetch failed: {e}")
        return False, None


# ============================================================================
# CAMERA AUTO-DETECTION
# ============================================================================
# Common snapshot path patterns for various IP camera manufacturers
# (INSTAR, Hikvision, Dahua, generic cameras, etc.)
_SNAPSHOT_CANDIDATES = [
    "/tmpfs/snap.jpg",                    # INSTAR and others
    "/tmpfs/auto.jpg",                    # INSTAR
    "/snap.cgi?chn=0",                    # Generic CGI
    "/cgi-bin/hi3510/snap.cgi?chn=0",    # Hikvision
    "/snapshot.cgi",                      # Generic
    "/cgi-bin/snapshot.cgi",              # Generic
    "/jpg/image.jpg",                     # Generic
    "/image.jpg",                         # Generic
    "/mjpg/video.mjpg",                   # Motion JPEG
    "/video.cgi",                         # Generic
]


def probe_snapshot_path(base: str) -> str:
    """Auto-detect the correct snapshot path for an IP camera.
    
    Tries common snapshot paths and returns the first one that returns a valid image.
    This is useful when camera snapshot URL is unknown.
    
    Args:
        base (str): Base camera URL (e.g., "http://192.168.1.100:8080")
        
    Returns:
        str: Valid snapshot URL, or base URL if none found
    """
    print("[INFO] Probing camera for correct snapshot path...")
    auth = (CAMERA_USER, CAMERA_PASS) if CAMERA_USER else None
    session = _make_session()
    
    # Try each candidate path
    for path in _SNAPSHOT_CANDIDATES:
        url = base.rstrip("/") + path
        try:
            resp = session.get(url, auth=auth, verify=False, timeout=5)
            if resp.status_code == 200:
                # Verify response is valid JPEG image
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

# using yolo to detect birds in the frame, returns (found:bool, annotated_frame:image, best_conf:float)
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


# ============================================================================
# MAIN MONITORING LOOP
# ============================================================================

def main():
    """Main monitoring loop - continuously checks camera for birds and sends alerts."""
    # Create output directory for detected bird images
    ensure_dir(SAVE_DIR)
    
    # Load YOLOv8 model for bird detection
    model = YOLO(MODEL_NAME)

    # Detection state variables
    consecutive_hits = 0      # Counter for consecutive bird detections
    last_alert_ts = 0         # Timestamp of last alert (for cooldown)
    last_check = 0            # Timestamp of last detection check
    cap = None                # Video capture object (for RTSP/file sources)

    print("[INFO] Bird monitor started (24/7 mode). Press Ctrl+C to stop.")

    # Detect camera source type (HTTP/HTTPS vs RTSP/file)
    http_mode = _is_http_source(CAMERA_SOURCE)
    active_source = CAMERA_SOURCE
    
    # For HTTP cameras, auto-detect correct snapshot path
    if http_mode:
        active_source = probe_snapshot_path(CAMERA_SOURCE)

    # ========== Main 24/7 monitoring loop ==========
    while True:
        try:
            # ===== FRAME ACQUISITION =====
            if http_mode:
                # Fetch frame from HTTP/HTTPS snapshot
                ok, frame = read_frame_http(active_source)
                if not ok or frame is None:
                    print("[WARN] Failed to read HTTP frame. Retrying in 5s...")
                    time.sleep(5)
                    continue
            else:
                # Connect/reconnect to RTSP or file source
                if cap is None or not cap.isOpened():
                    print("[WARN] Camera not connected. Retrying in 5 seconds...")
                    if cap is not None:
                        cap.release()
                    time.sleep(5)
                    cap = connect_camera(CAMERA_SOURCE)
                    continue

                # Read frame from video stream
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Failed to read frame. Reconnecting...")
                    cap.release()
                    cap = None
                    continue

            # ===== RATE LIMITING =====
            # Respect CHECK_INTERVAL_SEC to avoid CPU overload
            now = time.time()
            if now - last_check < CHECK_INTERVAL_SEC:
                time.sleep(0.05)
                continue
            last_check = now

            # ===== BIRD DETECTION =====
            found, annotated, best_conf = detect_bird(model, frame)

            # ===== DETECTION COUNTING =====
            # Require REQUIRED_HITS consecutive detections before alerting
            if found:
                consecutive_hits += 1
                print(f"[INFO] Bird detected {consecutive_hits}/{REQUIRED_HITS}, conf={best_conf:.2f}")
            else:
                if consecutive_hits > 0:
                    print("[INFO] No bird this cycle. Resetting counter.")
                consecutive_hits = 0

            # ===== ALERT TRIGGER =====
            # Send alert when threshold met AND cooldown period has passed
            if consecutive_hits >= REQUIRED_HITS and (now - last_alert_ts) >= ALERT_COOLDOWN_SEC:
                # Save annotated frame with timestamp
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(SAVE_DIR, f"bird_{ts}.jpg")
                cv2.imwrite(image_path, annotated)

                # Compose alert email
                subject = f"Bird Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                body = (
                    "Bird detected near GNSS antennas.\n"
                    f"Confidence: {best_conf:.2f}\n"
                    f"Time: {datetime.now()}\n"
                )

                # Send email alert
                try:
                    send_email_with_image(subject, body, image_path)
                    print(f"[ALERT] Email sent: {image_path}")
                    last_alert_ts = now           # Update cooldown timer
                    consecutive_hits = 0          # Reset detection counter
                except Exception as e:
                    print(f"[ERROR] Email send failed: {e}")

        except KeyboardInterrupt:
            # User pressed Ctrl+C to stop
            print("\n[INFO] Stopped by user.")
            break
        except Exception:
            # Unexpected error - log and continue
            print("[ERROR] Unexpected runtime error:")
            traceback.print_exc()
            time.sleep(3)

    # Clean up camera connection on exit
    if cap is not None:
        cap.release()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    
    # Check for command-line arguments
    if "--test-email" in _sys.argv:
        # Test email functionality
        print("Running in TEST EMAIL mode...")
        print("This will send a test email to verify SMTP configuration.")
        try:
            send_test_email()
        except Exception:
            print("[ERROR] Test email failed:")
            traceback.print_exc()
            input("Press Enter to exit...")
    else:
        # Normal operation - start 24/7 monitoring
        print("Running in MONITOR mode...")
        print("Continuous bird detection active. Press Ctrl+C to stop.")
        try:
            main()
        except Exception:
            print("[FATAL] Script crashed:")
            traceback.print_exc()
            input("Press Enter to exit...")