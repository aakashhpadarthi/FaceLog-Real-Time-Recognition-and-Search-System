import streamlit as st
import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time

# ========== Config ==========
CSV_FILE = 'face_log.csv'
IMG_DIR = 'captured_faces'
os.makedirs(IMG_DIR, exist_ok=True)

CAPTURE_INTERVAL = 60  # seconds per person

# ========== Utility ==========
def clean_broken_entries():
    global face_log, known_encodings
    if not face_log.empty:
        original_count = len(face_log)
        face_log = face_log[face_log['image_path'].apply(os.path.exists)]
        face_log.reset_index(drop=True, inplace=True)
        face_log.to_csv(CSV_FILE, index=False)
        known_encodings = [np.array(eval(e)) for e in face_log['encoding']]
        return original_count - len(face_log)
    return 0

# ========== Load or Init ==========
if os.path.exists(CSV_FILE):
    face_log = pd.read_csv(CSV_FILE)
    cleaned = clean_broken_entries()
    if cleaned > 0:
        st.warning(f"🧹 Auto-cleaned {cleaned} broken log entries.")
    known_encodings = [np.array(eval(e)) for e in face_log['encoding']]
else:
    face_log = pd.DataFrame(columns=['timestamp', 'image_path', 'encoding'])
    known_encodings = []

# ========== Streamlit UI ==========
st.set_page_config(page_title="Face Capturer & Finder", layout="wide")
st.title("🧠 Face Capturer & Finder")
section = st.radio("Choose Mode", ["Capture", "Find"])

# Manual clean button
if st.button("🧹 Clean Broken Entries"):
    removed = clean_broken_entries()
    if removed > 0:
        st.success(f"Removed {removed} invalid entries from the log.")
    else:
        st.info("No broken entries found.")

# ========== CAPTURE SECTION ==========
recent_faces = []
recent_timestamps = []

if section == "Capture":
    st.subheader("🎥 Optimized Face Capturer (1 per face per minute)")
    start_cam = st.button("Start Camera")
    stop_cam = st.button("Stop Camera")
    status = st.empty()

    if start_cam and not stop_cam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        status.info("Running... Press 'Stop Camera' to exit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status.error("Failed to access camera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)
            current_time = time.time()

            for encoding, loc in zip(encodings, locations):
                should_capture = True

                for idx, recent_encoding in enumerate(recent_faces):
                    dist = face_recognition.face_distance([recent_encoding], encoding)[0]
                    if dist < 0.45 and current_time - recent_timestamps[idx] < CAPTURE_INTERVAL:
                        should_capture = False
                        break

                if should_capture:
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
                    filename = f"{IMG_DIR}/face_{timestamp}.jpg"

                    y1, x2, y2, x1 = loc
                    face_img = rgb[y1:y2, x1:x2]
                    face_pil = Image.fromarray(face_img)
                    face_pil.save(filename, quality=100)

                    face_log.loc[len(face_log)] = [timestamp, filename, str(encoding.tolist())]
                    face_log.to_csv(CSV_FILE, index=False)
                    known_encodings.append(encoding)
                    st.success(f"✅ Face saved at {timestamp}")

                    recent_faces.append(encoding)
                    recent_timestamps.append(current_time)

                    if len(recent_faces) > 15:
                        recent_faces = recent_faces[-15:]
                        recent_timestamps = recent_timestamps[-15:]

            stframe.image(frame, channels="RGB")

        cap.release()
        status.warning("Camera stopped.")

# ========== FIND SECTION ==========
elif section == "Find":
    st.subheader("🔍 Find a Face")

    mode = st.radio("Choose Input Mode", ["Upload an Image", "Take a Snapshot"])
    uploaded_img = None
    frame = None

    if mode == "Upload an Image":
        uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    elif mode == "Take a Snapshot":
        snap = st.camera_input("Take a Picture")
        if snap:
            img = Image.open(snap)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if frame is not None:
        st.image(frame, caption="Query Image", use_column_width=True)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        query_encoding = face_recognition.face_encodings(rgb)

        if not query_encoding:
            st.error("❌ No face detected in the image.")
        else:
            query_encoding = query_encoding[0]
            match = None
            for i, known in enumerate(known_encodings):
                match_result = face_recognition.compare_faces([known], query_encoding)[0]
                if match_result:
                    match = face_log.iloc[i]
                    break

            if match is not None:
                st.success(f"✅ Match found! Timestamp: {match['timestamp']}")
                if os.path.exists(match['image_path']):
                    st.image(match['image_path'], caption="Matched Image")
                else:
                    st.warning("⚠️ Match found, but image file is missing.")
            else:
                st.warning("😕 No matching face found.")
