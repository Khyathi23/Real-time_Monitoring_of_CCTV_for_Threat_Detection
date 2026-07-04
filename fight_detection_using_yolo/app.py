import cv2
import os
import time
import streamlit as st
from detector import ViolenceDetector

st.set_page_config(
    page_title="Fight Detection System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Real-Time Fight Detection System")
st.caption("AI-powered anomaly detection for surveillance using YOLOv8")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.25,
    max_value=0.95,
    value=0.50,
    step=0.05
)
mode = st.sidebar.radio("Select Input Mode", ["📹 Upload Video", "📷 Live Webcam"])



@st.cache_resource
def load_model():
    return ViolenceDetector()


detector = load_model()

# Initialize session state for incidents
if "incidents" not in st.session_state:
    st.session_state.incidents = []



if mode == "📹 Upload Video":
    st.header("Upload Video for Analysis")
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        temp_path = "temp_upload.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("Video uploaded successfully!")

        if st.button("🔍 Analyze Video"):
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            progress_bar = st.progress(0)
            frame_display = st.empty()
            alert_display = st.empty()

            frame_count = 0
            incident_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress_bar.progress(frame_count / max(total_frames, 1))

                if frame_count % 3 != 0:
                    continue

                annotated_frame, detections = detector.detect(frame, confidence)

                if detections:
                    incident_count += 1
                    max_conf = max(d["confidence"] for d in detections)
                    alert_display.error(
                        f"⚠️ VIOLENCE DETECTED! Confidence: {max_conf:.0%}"
                    )
                    if incident_count % 10 == 1:
                        incident = detector.save_incident(frame, detections)
                        st.session_state.incidents.append(incident)
                else:
                    alert_display.success("✅ Scene is safe - No anomaly detected")

                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_display.image(
                    display_frame, channels="RGB", use_container_width=True
                )
                time.sleep(0.01)

            cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)

            st.balloons()
            st.success(
                f"Analysis complete! Processed {frame_count} frames. "
                f"Detected {incident_count} violent frames."
            )


elif mode == "📷 Live Webcam":
    st.header("Live Webcam Monitoring")
    st.info("Check the box below to start monitoring. Uncheck to stop.")

    # Create a checkbox toggle and placeholder containers
    run = st.checkbox("Start Monitoring")
    frame_display = st.empty()
    alert_display = st.empty()
    metrics_display = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error(
                "Could not access webcam. "
                "Please check your camera connection."
            )
        else:
            incident_count = 0
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                frame_count += 1
                annotated_frame, detections = detector.detect(frame, confidence)

                if detections:
                    incident_count += 1
                    max_conf = max(d["confidence"] for d in detections)
                    alert_display.error(
                        f"⚠️ VIOLENCE DETECTED! Confidence: {max_conf:.0%}"
                    )
                    # Save every 30th violent frame to avoid flooding disk
                    if incident_count % 30 == 1:
                        incident = detector.save_incident(frame, detections)
                        st.session_state.incidents.append(incident)
                else:
                    alert_display.success("✅ Scene is safe - No anomaly detected")

                # Convert BGR to RGB for Streamlit display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_display.image(
                    display_frame, channels="RGB", use_container_width=True
                )
                metrics_display.caption(
                    f"Frames: {frame_count} | Incidents: {incident_count}"
                )

            cap.release()


# Incident Log Section
st.divider()
st.header("📋 Incident Log")

if st.session_state.incidents:
    for incident in reversed(st.session_state.incidents):
        col1, col2 = st.columns([1, 3])
        with col1:
            if os.path.exists(incident["filename"]):
                st.image(incident["filename"], width=200)
        with col2:
            st.write(f"**Time:** {incident['timestamp']}")
            st.write(f"**Max Confidence:** {incident['max_confidence']:.0%}")
            st.write(f"**Detections in frame:** {incident['num_detections']}")
        st.divider()
else:
    st.info("No incidents recorded yet. Upload a video or start webcam monitoring.")