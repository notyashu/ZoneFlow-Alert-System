# main.py
import os
import cv2
import time
import torch
import json
import numpy as np
import argparse
import threading
import queue
from collections import deque, defaultdict
from ultralytics import YOLO
from dotenv import load_dotenv

from region_editor import load_regions
from telegram_worker import TelegramWorker

load_dotenv()

# --- CONFIGURATION ---
MODEL_PATH           = "yolo11n.pt"
RTSP_URL             = os.getenv("RTSP_URL")
CONF_THRESHOLD       = 0.2
IMAGE_SIZE           = 640
TARGET_FPS           = 10.0
FRAME_INTERVAL       = 1.0 / TARGET_FPS
ALERT_COOLDOWN       = 20    # seconds
ENTRANCE_SENSITIVITY = 22    # pixels for movement check
TRAIL_DURATION       = 1.5   # seconds for trail visibility

# Drawing settings
BOUNDARY_COLOR       = (0, 255, 255)
BOX_COLOR_ALERT      = (255, 0, 0)
BOX_COLOR_NORMAL     = (0, 0, 255)
CNTR_COLOR           = (255, 0, 255)
TRAIL_COLOR          = (0, 255, 0)
FONT                 = cv2.FONT_HERSHEY_SIMPLEX

# --- UTILITIES ---
def compute_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    a1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    a2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return inter / float(a1 + a2 - inter + 1e-6)


def create_region_overlays(frame_shape, regions):
    overlays = {}
    for r in regions:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        pts = np.array(r["points"], np.int32)
        cv2.polylines(mask, [pts], True, BOUNDARY_COLOR, 2)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(mask, r["name"], (cx, cy), FONT, 0.6, BOUNDARY_COLOR, 2)
        overlays[r["name"]] = mask
    combined = np.zeros(frame_shape, dtype=np.uint8)
    for m in overlays.values():
        combined = cv2.add(combined, m)
    overlays["combined"] = combined
    return overlays

# --- VIDEO STREAM ---
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader)
        self.thread.start()

    def _reader(self):
        while not self.stopped and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
            except cv2.error as e:
                print(f"VideoStream read error: {e}")
                break
            if not ret:
                time.sleep(0.01)
                continue
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)

# --- MAIN ---
def main(preview=False):
    # Load regions
    regions = load_regions()
    if not regions:
        print("No regions defined. Run with --edit.")
        return

    # Start Telegram worker
    bot = TelegramWorker(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))
    bot.start()

    # Load YOLO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    # Warm up model
    _ = model(torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=torch.float))

    # Video stream
    stream = VideoStream(RTSP_URL)
    if preview:
        cv2.namedWindow("Virtual Boundary Alert", cv2.WINDOW_NORMAL)

    # State
    last_y = {}
    last_alert = {}
    trails = defaultdict(lambda: deque())

    overlays = {}
    last_t, t0 = time.time(), time.time()
    fps_counter, current_fps = 0, 0

    try:
        while True:
            now = time.time()
            sleep = FRAME_INTERVAL - (now - last_t)
            if sleep > 0:
                time.sleep(sleep)
            frame = stream.read()
            last_t = time.time()

            if frame is None:
                continue

            # Prepare overlays once
            if not overlays:
                overlays = create_region_overlays(frame.shape, regions)
            combined_overlay = overlays["combined"]

            # FPS calc
            fps_counter += 1
            if last_t - t0 >= 1.0:
                current_fps = fps_counter / (last_t - t0)
                fps_counter, t0 = 0, last_t

            # Detection + tracking
            results = model.track(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMAGE_SIZE,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[0]
            )

            annotated = frame.copy()
            now_ts = time.time()
            candidates = []

            # First pass: detect entrance crossings
            for box in results[0].boxes:
                if box.id is None:
                    continue
                pid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                for r in regions:
                    if r["name"] != "Entrance":
                        continue
                    inside = cv2.pointPolygonTest(np.array(r["points"], np.int32), (cx, cy), False) >= 0
                    if inside:
                        prev_y = last_y.get(pid)
                        moved_up = prev_y is not None and cy < prev_y - ENTRANCE_SENSITIVITY
                        if moved_up and now_ts - last_alert.get(pid, 0) > ALERT_COOLDOWN:
                            candidates.append((pid, [x1, y1, x2, y2]))
                            last_alert[pid] = now_ts
                        last_y[pid] = cy
                    break

            # De-duplicate via IoU
            triggered = set()
            for i, (pid_i, b_i) in enumerate(candidates):
                suppress = False
                for j, (pid_j, b_j) in enumerate(candidates):
                    if i != j and compute_iou(b_i, b_j) > 0.7 and pid_i > pid_j:
                        suppress = True
                        break
                if not suppress:
                    triggered.add(pid_i)

            # Second pass: draw & alert
            for box in results[0].boxes:
                if box.id is None:
                    continue
                pid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                color = BOX_COLOR_ALERT if pid in triggered else BOX_COLOR_NORMAL
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID:{pid}", (x1, y1-5), FONT, 0.6, color, 2)
                cv2.circle(annotated, (cx, cy), 3, CNTR_COLOR, -1)

                trail = trails[pid]
                trail.append((cx, cy, now_ts))
                while trail and now_ts - trail[0][2] > TRAIL_DURATION:
                    trail.popleft()

                if pid in triggered:
                    for k in range(1, len(trail)):
                        x0, y0, _ = trail[k-1]
                        x1_, y1_, _ = trail[k]
                        cv2.line(annotated, (x0, y0), (x1_, y1_), TRAIL_COLOR, 2)
                    ts_str = time.strftime("%I:%M:%S %p - %d/%m/%Y", time.localtime(now_ts))
                    caption = (f"🚨 ALERT: {next(r['message'] for r in regions if r['name']=='Entrance')}\n"
                               f"Region: Entrance\nPerson ID: {pid}\nTime: {ts_str}")
                    bot.send(text=None, frame=annotated.copy(), caption=caption)

            # Overlay static regions
            annotated = cv2.add(annotated, combined_overlay)

            if preview:
                cv2.putText(annotated, f"FPS:{current_fps:.2f}", (10, 30), FONT, 1, (255,255,255), 2)
                cv2.imshow("Virtual Boundary Alert", annotated)
                if cv2.waitKey(1) == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop()
        bot.stop()
        if preview:
            cv2.destroyAllWindows()
        print("System stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Boundary Alert System")
    parser.add_argument("--edit", action="store_true", help="Launch region editor")
    parser.add_argument("--preview", action="store_true", help="Enable preview window")
    args = parser.parse_args()
    if args.edit:
        from region_editor import interactive_region_editor
        interactive_region_editor(RTSP_URL)
    else:
        main(preview=args.preview)
