# Object Detection RTSP YOLO

A real-time object detection system that streams from RTSP sources and performs YOLOv8 inference with NVIDIA GPU acceleration. Features a split-screen dashboard comparing raw webcam feed against AI-detected objects.

## Features

- **Real-time RTSP Streaming**: Connects to RTSP streams with optimized buffering to prevent lag
- **YOLOv8 Object Detection**: GPU-accelerated inference using YOLOv8s model
- **Split-Screen Dashboard**: Live display of raw feed (left) vs. AI detections (right)
- **Multi-Object Tracking**: Detects and counts people, mugs, mobile phones, and books
- **Performance Monitoring**: Displays streaming FPS, inference FPS, and GPU latency metrics
- **Video Recording**: Saves the split-screen pipeline output as AVI video
- **Latency Alerts**: Warns when GPU processing exceeds 45ms threshold

## Demo Video Output

View sample output videos showcasing the object detection pipeline:

[![Watch Demo Video](https://github.com/sfgrahman/Object_detection_rtsp_yolo/raw/main/demo_preview.gif)](https://github.com/sfgrahman/Object_detection_rtsp_yolo/raw/main/demo_video.mp4)

**Click the GIF above to download full video** | [Direct MP4 Link](https://github.com/sfgrahman/Object_detection_rtsp_yolo/raw/main/demo_video.mp4)

**Sample Output Features:**
- Split-screen display (Raw Feed vs. AI Detections)
- Real-time object counting
- GPU latency metrics
- Bounding boxes with class labels
- Performance FPS overlay

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Ultralytics YOLOv8
- NVIDIA GPU with CUDA support
- RTSP stream source

## Installation

```bash
pip install opencv-python ultralytics torch torchvision
```

## RTSP Server Setup

### Option 1: Using MediaMTX (Recommended)

MediaMTX is a lightweight, self-contained RTSP server that works on Linux, macOS, and Windows.

#### Installation

**Linux:**
```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.0_linux_amd64.tar.gz
chmod +x mediamtx
```

**macOS:**
```bash
brew install mediamtx
```

**Windows:**
Download from [MediaMTX Releases](https://github.com/bluenviron/mediamtx/releases)

#### Configuration

Create `mediamtx.yml`:

```yaml
# Port configuration
rtspPort: 8554

# Path configurations
paths:
  # Stream from webcam
  webcam:
    source: rtsps://example.com/stream  # External RTSP source
    
  # Or stream from local file
  file_stream:
    source: file /path/to/video.mp4
    
  # Or stream from V4L2 device (Linux)
  v4l2_stream:
    source: v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! rtph264pay pt=96 ! rtspclientsink location=rtsp://localhost:8554/v4l2_stream
```

#### Running MediaMTX

```bash
./mediamtx mediamtx.yml
```

Server will be available at: `rtsp://localhost:8554/webcam`

---

### Option 2: Using FFmpeg RTSP Server

Stream directly from a video file or webcam using FFmpeg:

```bash
# Stream from webcam (Linux)
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/webcam

# Stream from webcam (macOS)
ffmpeg -f avfoundation -i "0" -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/webcam

# Stream from video file (loop)
ffmpeg -stream_loop -1 -i video.mp4 -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/webcam
```

---

### Option 3: Using GStreamer

GStreamer provides flexible streaming capabilities:

```bash
# Stream from webcam (Linux)
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=5000

# Or use rtspsrc/rtmpsrc to relay existing streams
gst-launch-1.0 rtspsrc location="rtsp://external-source.com/stream" ! rtph264depay ! h264parse ! rtph264pay ! rtspsink location=rtsp://0.0.0.0:8554/webcam
```

---

### Option 4: Using Docker (MediaMTX)

Quick setup with Docker:

```bash
docker run --rm -p 8554:8554 -v $(pwd)/mediamtx.yml:/mediamtx.yml \
  bluenviron/mediamtx:latest
```

Or use Docker Compose:

```yaml
version: '3'
services:
  rtsp-server:
    image: bluenviron/mediamtx:latest
    ports:
      - "8554:8554"
    volumes:
      - ./mediamtx.yml:/mediamtx.yml
    environment:
      RTSP_PROTOCOLS: tcp
```

Run with: `docker-compose up`

---

### Testing RTSP Server

Verify the server is working:

```bash
# Using ffplay
ffplay rtsp://localhost:8554/webcam

# Using VLC
vlc rtsp://localhost:8554/webcam

# Using curl (check if stream is accessible)
curl -v rtsp://localhost:8554/webcam
```

---

### Common RTSP Server URLs

Use these in the detector if you already have streams running:

```python
# Local webcam via MediaMTX
RTSP_URL = "rtsp://localhost:8554/webcam"

# IP camera (Hikvision, Dahua, etc.)
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream"

# OnVif camera
RTSP_URL = "rtsp://admin:password@192.168.1.100/onvif/profile1"

# External cloud stream
RTSP_URL = "rtsp://example.com/live/stream"
```

## Configuration

Edit the following in `rtsp_detector.py`:

- **RTSP_URL**: Change the stream source (default: `rtsp://localhost:8554/webcam`)
- **Model**: Currently using `yolov8s.pt` (small model). Options: `yolov8n.pt`, `yolov8m.pt`, `yolov8l.pt`
- **GPU Device**: Set to `"cuda:0"` for the first GPU (change if using different device)
- **Detection Classes**: Configured for COCO classes: Person (0), Mug (41), Mobile (67), Book (73)
- **Confidence Threshold**: Set to 0.25 (adjust for sensitivity)
- **Mobile Confidence Threshold**: Set to 0.65 (higher confidence for mobiles)
- **Latency Warning**: 45ms threshold for GPU spike alerts

## Usage

```bash
python rtsp_detector.py
```

### Controls

- **Press 'q'**: Exit the application and save the recording

### Output

- **Live Display**: OpenCV window showing split-screen dashboard
- **Recording**: Saved as `recordings/dashboard_YYYYMMDD_HHMMSS.avi`

## Code Structure

### RTSPStreamReader Class

Handles background RTSP streaming to prevent frame drops:

- **`__init__(rtsp_url)`**: Initializes video capture and threading
- **`_reader()`**: Background thread that continuously reads frames and maintains raw FPS
- **`read()`**: Returns the latest frame from the queue
- **`release()`**: Stops the stream and releases resources

### main()

Main detection pipeline:

1. **Model Loading**: Loads YOLOv8s with GPU acceleration
2. **Stream Connection**: Connects to RTSP source
3. **Frame Processing**: 
   - Reads frames from RTSP stream
   - Runs YOLOv8 inference on GPU
   - Filters detections by confidence thresholds
   - Counts objects by class
4. **Visualization**:
   - Renders bounding boxes and labels
   - Displays metrics (FPS, latency, object counts)
   - Creates split-screen output
5. **Recording**: Writes video output at 30 FPS

## Performance Metrics

- **Stream Ingest FPS**: Rate at which frames are captured from RTSP source
- **Inference FPS**: Rate at which YOLOv8 processes frames (GPU dependent)
- **GPU Latency**: Time taken for single inference in milliseconds
- **Object Counts**: Real-time count of detected persons, mugs, mobiles, and books

## Output Directory

Recordings are saved to:
```
./recordings/dashboard_YYYYMMDD_HHMMSS.avi
```

Directory is created automatically if it doesn't exist.

## Notes

- The system uses a queue with `maxsize=1` to keep the pipeline synchronized
- Old frames are dropped to maintain real-time performance
- GPU latency spikes trigger console alerts for performance debugging
- Confidence threshold for Mobile (0.65) is higher than other classes to reduce false positives

## Troubleshooting

- **No frames displayed**: Check RTSP URL and network connectivity
- **Low FPS**: Verify GPU is being used, reduce `imgsz` from 640 to 416 for faster inference
- **CUDA errors**: Ensure NVIDIA drivers are installed and PyTorch is compiled with CUDA support
- **High latency**: Reduce model size (use `yolov8n.pt`) or lower image resolution