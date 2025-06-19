# ZoneFlow Alert System

ZoneFlow Alert System is a Python application that:

- **Monitors** RTSP CCTV streams
- **Tracks** people in real time using YOLOv11
- **Detects** entry/exit or directional flow across custom polygonal zones
- **Sends** annotated snapshots and custom messages to Telegram

## 🚀 Key Features

1. **Interactive Region Editor** (`--edit`)
   - Draw, name, and assign alert messages to polygonal zones.
2. **Directional Flow Detection**
   - Configurable sensitivity to detect movement direction (e.g., entering vs. exiting).
3. **Telegram Integration**
   - Instant push notifications with annotated frames and timestamps.
4. **Performance Optimizations**
   - Frame decimation (target FPS) and buffer management.
   - ByteTrack-based tracking to reuse existing detections.
   - IoU-based duplicate alert suppression.
5. **Benchmark & Diagnostics**
   - `benchmark_yolo11n.py`: measure inference latency on CPU/GPU.
   - `check_cuda.py`: confirm PyTorch CUDA setup.

## 🔧 Installation & Setup

1. **Clone the repo**

   ```bash
   https://github.com/notyashu/ZoneFlow-Alert-System.git
   cd ZoneFlow-Alert-System
   ```

2. **Create & activate virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .\.venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r src/requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Fill in your `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and `RTSP_URL`.

---

### 📱 Telegram Bot Setup

1. **Create a bot**
   - Open Telegram and search **@BotFather**.
   - Send `/newbot` and follow prompts to set a name & username.
   - BotFather will reply with **`TOKEN`**.
2. **Get your `CHAT_ID`**
   - Add your new bot to a group or chat yourself.
   - Send a message to the bot or group.
   - Visit:
     `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Inspect JSON reply; locate `chat.id` under `message` → that’s your `CHAT_ID`.

_For more details, see [Telegram Bot API manual](https://core.telegram.org/bots)._

---

### 📥 Downloading Your YOLO Model

ZoneFlow uses [Ultralytics YOLOv11 models](https://docs.ultralytics.com/tasks/detect/#models). To choose/configure a model:

1. **Install Ultralytics** (already in `requirements.txt`):
   ```bash
   pip install ultralytics
   ```
2. **Download a model**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n.pt')       # or yolo11s.pt, yolo11m.pt, etc.
   model.export(format='onnx')      # optional: export to ONNX
   ```
3. **Point `MODEL_PATH`** in `src/main.py` to your downloaded `.pt` file.

---

## ⚙️ Optimization Techniques Used

- **Target FPS Control**: Sleep intervals to enforce `TARGET_FPS`, reducing redundant inference.
- **Multithreading Techniques**: RTSP video capture and Telegram messaging each run on dedicated background threads using queues, ensuring non-blocking real-time performance and smooth alerting.
- **Video Buffer Management**: `CAP_PROP_BUFFERSIZE=1` + single-slot queue to minimize latency.
- **ByteTrack Integration**: Keeps object identities across frames cheaply, avoiding re-detection overhead.
- **Smart Alert Debounce**: Cooldown timer (`ALERT_COOLDOWN`) per object to prevent spamming.
- **IoU Suppression**: Discards overlapping duplicate triggers (>70% overlap) to keep alerts unique.
- **Pre-rendered Overlays**: Static zone boundaries drawn once and composited every frame for speed.

---

## ▶️ Running the System

- **Edit zones**:
  ```bash
  python src/main.py --edit
  ```
- **Start monitoring** (with preview window):
  ```bash
  python src/main.py --preview
  ```
- **Headless mode**:
  ```bash
  python src/main.py
  ```

---

## 🎓 License

MIT © Suryansh Mehrotra
