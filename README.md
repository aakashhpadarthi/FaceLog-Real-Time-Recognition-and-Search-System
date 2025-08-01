📸 Real-Time Face Recognition and Logging System
A real-time face detection and recognition system built with Python, OpenCV, and Streamlit, designed to capture, log, and search unique human faces using a live webcam feed.

This project is optimized to store only one image per person per minute, reducing redundancy and ensuring clean, high-quality face data logging. It also includes a searchable interface that allows users to upload or capture a photo and find matches from previously captured faces.

 Features
 Real-time face detection using face_recognition

 Saves only unique faces (1 per person per minute)

 CSV-based logging with timestamp, image path, and encoding

 Face search via image upload or live capture

 High-resolution face capture

 Auto-cleaning of broken/missing image references

 Built with an intuitive Streamlit UI

 Tech Stack
Python

OpenCV

face_recognition

Streamlit

pandas

Pillow

 Getting Started
bash
Copy
Edit
pip install -r requirements.txt
streamlit run app.py
 Output
Captured images are saved in the captured_faces/ folder
Logs are stored in face_log.csv with metadata

 Future Enhancements
Add face labeling and tagging

Deploy as a web service or desktop app

Extend to vehicle/license plate recognition system

