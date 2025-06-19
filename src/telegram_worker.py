# telegram_worker.py
import queue
import cv2
import requests

class TelegramWorker:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.queue = queue.Queue()
        self._thread = None

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, frame, caption = item
            try:
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    files = {'photo': ('snapshot.jpg', buf.tobytes(), 'image/jpeg')}
                    data = {'chat_id': self.chat_id, 'caption': caption}
                    requests.post(f"https://api.telegram.org/bot{self.token}/sendPhoto", data=data, files=files, timeout=10)
                else:
                    requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat_id, "text": text}, timeout=10)
            except Exception as e:
                print(f"✗ Telegram error: {e}")
            finally:
                self.queue.task_done()

    def send(self, text=None, frame=None, caption=None):
        self.queue.put((text, frame, caption))

    def stop(self):
        self.queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)
