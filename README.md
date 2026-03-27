# Real-Time GNSS Bird Interference Detection System

Real-time bird detection system designed to monitor GNSS antennas and identify potential sources of signal degradation caused by nearby birds.

## Background

GNSS signals can be affected by objects near the antenna, a phenomenon known as multipath.  
On our rooftop GNSS installations, we occasionally observe signal outliers. It is often difficult to determine whether these anomalies are caused by firmware issues, hardware limitations, or environmental factors. Birds are one of the most common environmental causes.

## Solution

In this project, I installed a surveillance camera to continuously monitor GNSS antennas.  
A real-time computer vision script detects birds when they land near the antennas, captures snapshots, and notifies users.

This enables engineers to quickly understand whether observed GNSS performance anomalies may be related to bird activity.


<img width="1362" height="576" alt="image" src="https://github.com/user-attachments/assets/7651902a-35d9-43e1-b4c3-3af26aa2645b" />

GNSS Antennas at Roof



<img width="583" height="537" alt="Screenshot 2026-03-27 154356" src="https://github.com/user-attachments/assets/2cb24297-5b0e-43a9-a1a4-886ba95aa5a0" />


Bird Alert Notification


<img width="2730" height="1366" alt="Gemini_Generated_Image_hi87evhi87evhi87" src="https://github.com/user-attachments/assets/65fea622-e71a-4164-a9f6-864946b60701" />

Detected Bird


## Key Features

- Real-time video stream processing  
- Advanced bird detection using **YOLOv8 AI model**  
- Automatic snapshot capture upon detection  
- Notification system via email with annotated images  
- Multiple camera source support (HTTP/HTTPS snapshots, RTSP streams)  
- Timestamp logging for correlation with GNSS data  
- Configurable detection thresholds and alert cooldown periods  
- Auto-detection of camera snapshot paths  

## Technology Stack & AI Model

This project leverages state-of-the-art computer vision technology for real-time object detection:

### YOLOv8 - Object Detection Model

The system uses **YOLOv8 (You Only Look Once v8)**, a cutting-edge deep learning model for real-time object detection.

**What is YOLOv8?**
- YOLO is a single-stage object detection algorithm that processes entire images in one forward pass
- Extremely fast and suitable for real-time applications
- Provides both object classification and localization (bounding boxes)
- Pre-trained on COCO dataset (91 classes including birds, people, vehicles, etc.)

**Why YOLOv8?**
- ✅ **Real-Time Performance**: Processes frames at high speed without significant latency
- ✅ **High Accuracy**: Achieves excellent detection accuracy even in complex environments
- ✅ **Lightweight**: Available in multiple sizes (nano, small, medium, large) for performance/accuracy trade-offs
- ✅ **Pre-trained**: Comes with weights trained on millions of images
- ✅ **Easy to Use**: Simple Python API through the `ultralytics` library

**Model Variants Used:**
- `yolov8n.pt` (Nano) - Fastest, most efficient, good for resource-constrained devices
- Other available models: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

### Key Libraries & Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **opencv-python** | >=4.8.0 | Computer vision library for image processing, manipulation, and display |
| **numpy** | >=1.24.0 | Numerical computing for array operations and image data handling |
| **ultralytics** | >=8.0.0 | YOLOv8 framework and pre-trained model provider |
| **requests** | >=2.31.0 | HTTP library for fetching camera snapshots from HTTP/HTTPS sources |
| **urllib3** | >=2.0.0 | Advanced HTTP client for URL handling and connection pooling |
| **python-dotenv** | >=1.0.0 | Load environment variables from .env file for secure credential storage |
| **Pillow** | >=10.0.0 | Python Imaging Library for additional image processing capabilities |

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Virtual environment (venv, conda, etc.)
- Camera source (HTTP/HTTPS snapshot URL or RTSP stream)
- SMTP server for email notifications

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone <repository-url>
cd Real-Time-GNSS-Bird-Interference-Detection-System
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including YOLOv8 model.

### Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# Camera Configuration
CAMERA_SOURCE=https://your-camera-ip:port/snapshot
CAMERA_USER=username
CAMERA_PASS=password

# AI Model
MODEL_NAME=yolov8n.pt

# Detection Settings
CONF_THRESHOLD=0.45
REQUIRED_HITS=3
CHECK_INTERVAL_SEC=2
ALERT_COOLDOWN_SEC=300

# Email Configuration
SMTP_SERVER=smtp.example.com
SMTP_PORT=25
SMTP_USER=your-email@example.com
SMTP_PASSWORD=your-password
EMAIL_FROM=your-email@example.com
EMAIL_TO=recipient@example.com
```

### Step 4: Test Configuration

```bash
# Test email sending
python BirdAlert.py --test-email
```

### Step 5: Run the System

```bash
python BirdAlert.py
```

## How It Works

### Detection Pipeline

```
Camera Frame
    ↓
[Acquire Frame] (HTTP snapshot or RTSP stream)
    ↓
[Resize & Preprocess] (normalize for YOLOv8)
    ↓
[YOLOv8 Inference] (run through neural network)
    ↓
[Filter Detections] (keep only birds, threshold confidence)
    ↓
[Count Consecutive Detections]
    ↓
[Alert Logic] (required hits + cooldown)
    ↓
[Save & Email] (capture annotated frame + send alert)
```

### Detection Process

1. **Frame Acquisition**: System continuously fetches frames from camera
2. **YOLOv8 Processing**: Each frame is processed by YOLOv8 model
3. **Class Filtering**: Only "bird" detections are retained (confidence >= threshold)
4. **Consecutive Hit Counter**: Requires N consecutive detections to reduce false positives
5. **Alert Generation**: When threshold reached, saves annotated image and sends email
6. **Cooldown**: Prevents alert spam by enforcing minimum interval between notifications

### Confidence Scores

YOLOv8 assigns confidence scores (0.0 - 1.0) to each detection:
- **0.8 - 1.0**: Very confident detection
- **0.6 - 0.8**: Good confidence
- **0.4 - 0.6**: Moderate confidence (many false positives)
- **< 0.4**: Low confidence (typically filtered out)

Default threshold is 0.45, tuned to catch most real birds while maintaining acceptable false positive rate.



## Engineering Insight

This project does not attempt to prevent interference, but instead provides observability into a potential root cause of GNSS anomalies.

By correlating timestamps of detected bird activity with GNSS performance logs, it enables faster debugging and supports validation of environmental interference hypotheses.

## Use Case

This system is particularly useful in environments where GNSS accuracy is critical and environmental factors need to be monitored and analyzed systematically.

## Output & Monitoring

### Detected Bird Images

When birds are detected, the system saves annotated images to the `alerts/` directory:
- Format: `bird_YYYYMMDD_HHMMSS.jpg`
- Contains: Annotated frame with bounding boxes and confidence scores
- Example: `bird_20260327_154230.jpg`

### Console Logging

The script provides real-time console output:

```
[INFO] Bird monitor started (24/7 mode). Press Ctrl+C to stop.
[INFO] Probing camera for correct snapshot path...
[INFO] Snapshot path found: https://camera-ip:3333/tmpfs/snap.jpg
[INFO] Bird detected 1/3, conf=0.87
[INFO] Bird detected 2/3, conf=0.92
[INFO] Bird detected 3/3, conf=0.89
[ALERT] Email sent: alerts/bird_20260327_154230.jpg
```

### Email Alerts

When alerts are triggered, recipients receive emails with:
- Detection timestamp
- Confidence score
- Attached annotated image showing bird location with bounding box

## Configuration Examples

### High Accuracy (Reduce False Positives)
```
CONF_THRESHOLD=0.75
REQUIRED_HITS=5
CHECK_INTERVAL_SEC=1
```
Best for: Environments where false alerts are costly

### High Sensitivity (Catch All Detections)
```
CONF_THRESHOLD=0.35
REQUIRED_HITS=1
CHECK_INTERVAL_SEC=3
```
Best for: Safety-critical environments where missing a detection is worse than false alerts

### Balanced (Recommended)
```
CONF_THRESHOLD=0.45
REQUIRED_HITS=3
CHECK_INTERVAL_SEC=2
ALERT_COOLDOWN_SEC=300
```
Best for: General deployments

## Troubleshooting

### Issue: "Cannot find correct snapshot path"
**Solution**: Manually inspect camera web interface and find the snapshot URL

### Issue: SSL Certificate Error
**Solution**: The system handles self-signed certificates, but if issues persist:
```
- Verify camera URL is reachable: curl -k https://camera-ip:port
- Check firewall settings
- Verify SSL certificate is valid
```

### Issue: False Positives (Detecting non-birds)
**Solution**: Increase confidence threshold:
```
CONF_THRESHOLD=0.65  # Default is 0.45
REQUIRED_HITS=5      # Default is 3
```

### Issue: Missed Detections (Real birds not detected)
**Solution**: Decrease confidence threshold and required hits:
```
CONF_THRESHOLD=0.35  # Lower = more sensitive
REQUIRED_HITS=2      # Lower = faster alerts
```

### Issue: Email not sending
**Solution**: Test SMTP configuration:
```bash
python BirdAlert.py --test-email
```

Check:
- SMTP server is reachable on configured port
- Email credentials are correct
- Firewall allows outbound SMTP connections
- SMTP port is not blocked (try 587 if 25 is blocked)




## Hardware Setup

The system is deployed on a rooftop environment where GNSS antennas are installed and exposed to external environmental factors.

### Components

- **GNSS Antennas**  
  Installed on the rooftop for signal reception and performance evaluation.

- **Surveillance Camera**  
  A 9008 model camera with 1080p Full HD is mounted to monitor the antenna area in real time.

- **Processing Unit**  
  A local machine/server processes the incoming video stream and runs the bird detection algorithm.

- **Network Connection**  
  Ensures real-time video streaming and enables notification delivery to users.

### Setup Overview

The camera is positioned to have a clear field of view covering all GNSS antennas.  
It continuously streams video to the processing unit, where frames are analyzed in real time for bird detection.

This setup allows continuous environmental monitoring without interfering with the GNSS system itself.
