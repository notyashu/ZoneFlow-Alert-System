import time
import torch
from ultralytics import YOLO

MODEL_PATH   = "yolo11n.pt"
IMAGE_SIZE   = 1280
CONF_THRESHOLD = 0.15

def benchmark(device: str, runs: int = 20):
    print(f"\n--- Benchmarking on {device.upper()} ---")
    # Load model onto device
    model = YOLO(MODEL_PATH).to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        # try half precision
        try:
            model.model.half()
            dtype = torch.half
        except:
            dtype = torch.float
    else:
        dtype = torch.float

    # Warm‑up (allocates GPU memory, optimizes kernels)
    dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=dtype)
    _ = model.track(dummy, conf=CONF_THRESHOLD, imgsz=IMAGE_SIZE,
                    tracker="bytetrack.yaml", persist=True, classes=[0])

    # Measure inference time
    start = time.time()
    for _ in range(runs):
        _ = model.track(dummy, conf=CONF_THRESHOLD, imgsz=IMAGE_SIZE,
                        tracker="bytetrack.yaml", persist=True, classes=[0])
    # On CUDA, wait for all kernels to finish
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start
    print(f"Avg inference time over {runs} runs: {total / runs:.3f} s")

    # Show post‑warmup memory stats (only meaningful on CUDA)
    if device == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_resv  = torch.cuda.memory_reserved()  / 1024**2
        print(f"CUDA memory allocated: {mem_alloc:.1f} MB")
        print(f"CUDA memory reserved : {mem_resv:.1f} MB")

if __name__ == "__main__":
    # Run CPU benchmark
    benchmark("cpu")
    # Run GPU benchmark (if available)
    if torch.cuda.is_available():
        benchmark("cuda")
    else:
        print("\nCUDA not available; skipping GPU benchmark.")
