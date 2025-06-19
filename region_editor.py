# region_editor.py
import cv2
import json
import numpy as np

def load_regions(region_file="boundary.json"):
    try:
        with open(region_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def interactive_region_editor(rtsp_url, region_file="boundary.json"):
    """
    Launches a simple GUI to draw polygonal regions and save them to JSON.
    """
    regions = []
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot grab frame for region editor.")

    canvas, pts = frame.copy(), []
    cv2.namedWindow("Region Editor", cv2.WINDOW_NORMAL)

    def click_event(event, x, y, flags, param):
        nonlocal canvas, pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(canvas, (x, y), 4, (255, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts)], False, (0, 255, 255), 2)
            cv2.imshow("Region Editor", canvas)

    cv2.setMouseCallback("Region Editor", click_event)
    cv2.imshow("Region Editor", canvas)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and pts:
            name = input("Enter region name: ")
            msg = input("Enter alert message: ")
            regions.append({"name": name, "points": pts.copy(), "message": msg})
            pts.clear()
            canvas = frame.copy()
            cv2.imshow("Region Editor", canvas)
        elif key == ord('q'):
            break

    cv2.destroyWindow("Region Editor")
    with open(region_file, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"{len(regions)} regions saved to {region_file}.")