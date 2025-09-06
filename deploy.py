import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

def classify_light_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 50]), np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_yellow, upper_yellow = np.array([15, 70, 50]), np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green, upper_green = np.array([40, 70, 50]), np.array([85, 255, 255])
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
    return {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0)}.get(label, (200, 200, 200))

def main():
    st.set_page_config(page_title="🚦 Smart Traffic Light Detector", page_icon="🚦", layout="wide")

    st.markdown("""
        <style>
        .stApp { background-color: #0b0f1a; color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
        .title { text-align: center; font-size: 40px !important; font-weight: 800; color: #38bdf8; margin-bottom: -5px; }
        .subtitle { text-align: center; font-size: 18px; color: #94a3b8; margin-bottom: 25px; }
        .stButton button { background: linear-gradient(90deg, #2563eb, #9333ea); color: white !important; border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600; border: none; transition: 0.3s; }
        .stButton button:hover { opacity: 0.85; transform: scale(1.02); }
        .stDownloadButton button { background: linear-gradient(90deg, #22c55e, #16a34a); color: white !important; border-radius: 10px; padding: 0.7rem 1.3rem; font-weight: 600; border: none; }
        .stProgress .st-bo { background-color: #38bdf8; }
        .css-1d391kg { background-color: #1e293b !important; border: 1px solid #334155 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>🚦 Smart Traffic Light Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a video and let AI highlight <b style='color:#f87171;'>Red</b>, <b style='color:#facc15;'>Yellow</b>, and <b style='color:#4ade80;'>Green</b> traffic signals.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Choose a Traffic Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎬 Original Footage")
            st.video(uploaded_file)

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_file.read())
        temp_input.close()

        if st.button("🚀 Start Detection"):
            with st.spinner("⚡Analyzing your video..."):
                model = YOLO("yolo11n.pt")

                cap = cv2.VideoCapture(temp_input.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w, h = int(cap.get(3)), int(cap.get(4))
                cap.release()

                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = temp_output.name
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25, (w, h))

                names = model.model.names if hasattr(model, "model") else model.names
                traffic_light_id = [k for k, v in names.items() if v == "traffic light"][0]

                gen = model.predict(source=temp_input.name, conf=0.25, stream=True, verbose=False)

                progress = st.progress(0)
                eta_placeholder = st.empty()

                start_time = time.time()
                frame_count = 0

                for result in gen:
                    frame_count += 1
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

                    if total_frames > 0:
                        progress_val = frame_count / total_frames
                        progress.progress(min(1.0, progress_val))

                        elapsed = time.time() - start_time
                        if frame_count > 0:
                            fps_est = frame_count / elapsed
                            remaining_frames = total_frames - frame_count
                            eta_seconds = int(remaining_frames / fps_est) if fps_est > 0 else 0
                            eta_placeholder.markdown(f"⏳ Estimated time left: **{eta_seconds} sec**")

                writer.release()

            with col2:
                st.markdown("### ✅ Processed Output")
                st.video(out_path)
                st.download_button("⬇ Save Video", open(out_path, "rb"), file_name="traffic_lights_detected.mp4")

        else:
            with col2:
                st.info("👆 Click **Start Detection** to process your video")


if __name__ == "__main__":
    main()
