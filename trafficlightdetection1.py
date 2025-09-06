import cv2
import numpy as np
from ultralytics import YOLO
import os

VIDEO_PATH = "tr.mp4"   
MODEL_PATH = "yolo11n.pt"
CONF_THRESHOLD = 0.5

def classify_light_color(crop):
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50]); upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_yellow = np.array([15, 70, 50]); upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([40, 70, 50]); upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    r, y, g = np.count_nonzero(mask_red), np.count_nonzero(mask_yellow), np.count_nonzero(mask_green)

    if max(r, y, g) == 0:
        return "Unknown"
    if r >= y and r >= g:
        return "Red"
    elif y >= r and y >= g:
        return "Yellow"
    else:
        return "Green"

def get_color_code(label):
    
    if label == "Red":
        return (0, 0, 255)      
    elif label == "Yellow":
        return (0, 255, 255)   
    elif label == "Green":
        return (0, 255, 0)      
    else:
        return (200, 200, 200)  

def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()

    out_path = "traffic_lights_colored.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25, (w, h))

    names = model.model.names if hasattr(model, "model") else model.names
    traffic_light_id = [k for k, v in names.items() if v == "traffic light"][0]

    gen = model.predict(source=VIDEO_PATH, conf=CONF_THRESHOLD, stream=True, verbose=False)

    for result in gen:
        frame = result.orig_img
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            for i, c in enumerate(cls):
                if c != traffic_light_id:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                crop = frame[max(0, y1):y2, max(0, x1):x2]

                if crop.size == 0:
                    continue

                color_label = classify_light_color(crop)
                box_color = get_color_code(color_label)
                conf_val = float(conf[i])

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                
                label = f"{color_label} ({conf_val:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)
        screen_w, screen_h = 1280, 720
        scale_w = screen_w / frame.shape[1]
        scale_w = screen_h / frame.shape[0]
        scale = min(scale_w, screen_h)

        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)

        display_frame = cv2.resize(frame, (new_w, new_h))
        cv2.imshow("Traffic Light Detection",display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    print(f"[INFO] Output saved: {out_path}")

if __name__ == "__main__":
    main()
    if 'writer' in locals():
        writer.release()
        cv2.destroyAllWindows()
