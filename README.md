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
- Bird detection using computer vision  
- Automatic snapshot capture upon detection  
- Notification system for detected events  
- Timestamp logging for correlation with GNSS data  

## Engineering Insight

This project does not attempt to prevent interference, but instead provides observability into a potential root cause of GNSS anomalies.

By correlating timestamps of detected bird activity with GNSS performance logs, it enables faster debugging and supports validation of environmental interference hypotheses.

## Use Case

This system is particularly useful in environments where GNSS accuracy is critical and environmental factors need to be monitored and analyzed systematically.


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
