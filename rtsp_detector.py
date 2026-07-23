import os
import cv2
import queue
import threading
import time
from datetime import datetime
from ultralytics import YOLO


class RTSPStreamReader:
    """Handles the RTSP stream in a background thread to prevent lag."""

    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        
        # Ingest Tracking Variables
        self.raw_fps = 0
        self._frame_count = 0
        self._start_time = time.time()
        
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # --- TRACK RAW STREAMING INGEST SPEED ---
            self._frame_count += 1
            elapsed = time.time() - self._start_time
            if elapsed >= 1.0:
                self.raw_fps = int(self._frame_count / elapsed)
                self._frame_count = 0
                self._start_time = time.time()

            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()


def main():
    RTSP_URL = "rtsp://localhost:8554/webcam"

    print("Loading YOLOv8 Small model for NVIDIA GPU acceleration...")
    model = YOLO("yolov8s.pt").to("cuda:0")

    model.model.names[0] = "Person"
    model.model.names[41] = "Mug"
    model.model.names[67] = "Mobile"
    model.model.names[73] = "Book"

    print(f"Connecting to RTSP Stream: {RTSP_URL}")
    stream = RTSPStreamReader(RTSP_URL)

    print("Waiting for initial stream dimensions...")
    first_frame = None
    while first_frame is None:
        first_frame = stream.read()
        time.sleep(0.01)

    h, w, _ = first_frame.shape
    combined_width = w * 2 

    output_dir = "recordings"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = os.path.join(output_dir, f"dashboard_{timestamp}.avi")
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30.0, (combined_width, h))
    print(f"Recording initialized: {OUTPUT_FILE}")

    # AI Processing Trackers
    ai_fps_display = 0
    frame_count = 0
    start_time = time.time()
    LATENCY_WARN_THRESHOLD_MS = 45.0  

    print("Starting Split-Screen Pipeline. Press 'q' to exit.")

    try:
        while True:
            raw_frame = stream.read()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            detection_frame = raw_frame.copy()
            inference_start = time.time()

            results = model(
                detection_frame, 
                stream=True, 
                classes=[0, 41, 67, 73], 
                conf=0.25,
                imgsz=640,  
                device=0, 
            )

            counts = {"Book": 0, "Person": 0, "Mug": 0, "Mobile": 0}

            for r in results:
                valid_indices = []
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls)
                    conf_score = float(box.conf)

                    if cls_id == 67 and conf_score < 0.65:
                        continue

                    valid_indices.append(i)

                    if cls_id == 73: counts["Book"] += 1
                    elif cls_id == 0: counts["Person"] += 1
                    elif cls_id == 41: counts["Mug"] += 1
                    elif cls_id == 67: counts["Mobile"] += 1

                r.boxes = r.boxes[valid_indices]
                detection_frame = r.plot()

            inference_time_ms = (time.time() - inference_start) * 1000

            if inference_time_ms > LATENCY_WARN_THRESHOLD_MS:
                print(f"[BENCHMARK ALERT] GPU Latency Spike: {inference_time_ms:.1f}ms")

            # Stable AI display FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                ai_fps_display = int(frame_count / elapsed_time)
                frame_count = 0
                start_time = time.time()

            # --- RENDER RAW METRICS (LEFT FRAME) ---
            raw_metrics = [
                ("RAW WEBCAM FEED", (20, 40), (0, 255, 0)),
                (f"Stream Ingest FPS: {stream.raw_fps}", (20, 80), (100, 255, 100))
            ]
            for text, coords, color in raw_metrics:
                cv2.putText(raw_frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(raw_frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- RENDER AI METRICS (RIGHT FRAME) ---
            ai_metrics = [
                (f"Inference FPS: {ai_fps_display}", (20, 40), (0, 255, 0)),
                (f"GPU Latency: {inference_time_ms:.1f} ms", (20, 80), (255, 100, 100) if inference_time_ms > LATENCY_WARN_THRESHOLD_MS else (100, 255, 255)),
                (f"Mugs: {counts['Mug']} | Books: {counts['Book']}", (20, 120), (255, 255, 255)),
                (f"People: {counts['Person']} | Mobiles: {counts['Mobile']}", (20, 160), (255, 255, 255)),
                ("● RECORDING LIVE", (20, h - 30), (0, 0, 255))
            ]
            for text, coords, color in ai_metrics:
                cv2.putText(detection_frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(detection_frame, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            split_screen = cv2.hconcat([raw_frame, detection_frame])
            video_writer.write(split_screen)
            cv2.imshow("Live Pipeline Dashboard (Raw Left vs AI Right)", split_screen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Cleaning up resources and saving file buffers...")
        video_writer.release()
        stream.release()
        cv2.destroyAllWindows()
        print("Finished!")


if __name__ == "__main__":
    main()
