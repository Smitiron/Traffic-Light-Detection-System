# 🚦 Traffic Light Detection with YOLOv11 + OpenCV

This project detects **traffic lights (Red / Yellow / Green)** in videos using **YOLOv11** for object detection and **OpenCV (HSV analysis)** for color classification. Bounding boxes are drawn in the detected color of the light.

Demo Video:
https://drive.google.com/drive/folders/175AalmAAmgjR8pfpx5C48k1ME6kl05IB?usp=sharing
---

## 📂 Project Structure

```
Traffic-Light-Detection-System/
│
├── deploy.py                   # Streamlit app (modern UI version)
├── trafficlightdetection1.py   # Standalone script (runs with OpenCV window)
├── tr.mp4                      # Example input video
├── traffic_lights_output.mp4   # Example output video (after processing)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Installation

### 1. Clone this project

```bash
git clone https://github.com/Smitiron/Traffic-Light-Detection-System.git
cd Traffic-Light-Detection-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Option 1: Run with Streamlit (Web App)

```bash
streamlit run deploy.py
```

* Opens a **web interface** in your browser.
* Upload a video → Click **Start Detection** → View processed video side by side.
* Download the output video once processed.

---

### Option 2: Run Standalone Script

```bash
python trafficlightdetection1.py
```

* Opens a popup window showing **live detection**.
* Bounding boxes are drawn around traffic lights in their respective colors:

  * 🔴 Red → Red Box
  * 🟡 Yellow → Yellow Box
  * 🟢 Green → Green Box
* Press **Q** to stop the video.
* Output is saved as:

```
traffic_lights_output.mp4
```

---

## 🎥 Example

**Input Video**: `tr.mp4`
A normal traffic scene with traffic lights.

**Output Video**: `traffic_lights_output.mp4`
Same video but with bounding boxes around traffic lights showing their active color.

---

## 📦 requirements.txt

```txt
streamlit>=1.38.0
opencv-python==4.11.0.80
opencv-contrib-python==4.11.0.80
numpy>=1.26.0
ultralytics>=8.2.0
```

---

## 🚀 Notes

* Works with **any video** containing traffic lights.
* For the Streamlit app, ensure `yolo11n.pt` (YOLOv11 weights) is in the project folder.
* For the standalone script, update the video path if needed.

---

📌 **GitHub Repo:** [Smitiron/Traffic-Light-Detection-System](https://github.com/Smitiron/Traffic-Light-Detection-System)




