import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

important_body_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mask = img.copy()
            landmarks = results.pose_landmarks.landmark

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * img.shape[1]), int(start.y * img.shape[0])
                x2, y2 = int(end.x * img.shape[1]), int(end.y * img.shape[0])
                cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for idx, l in enumerate(landmarks):
                x, y = int(l.x * img.shape[1]), int(l.y * img.shape[0])
                if idx in important_body_indices:
                    cv2.circle(mask, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(mask, (x, y), 12, (0, 180, 0), 2)
                else:
                    cv2.circle(mask, (x, y), 3, (0, 0, 255), -1)
                    cv2.circle(mask, (x, y), 6, (0, 0, 180), 1)

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0]]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1],
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0]]
            self.left_angle = 180 - calculate_angle(left_elbow, left_shoulder,
                                                    [left_shoulder[0], left_shoulder[1] - 100])

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]]
            self.right_angle = 180 - calculate_angle(right_elbow, right_shoulder,
                                                     [right_shoulder[0], right_shoulder[1] - 100])

            return av.VideoFrame.from_ndarray(mask, format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def draw_angle_meter(angle, label):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    circle = plt.Circle((0, 0), 1, color=(0.2, 0.2, 0.2), fill=True)
    ax.add_artist(circle)

    if angle > 120:
        arc_color = "#4CAF50"
        level = "Excellent"
    elif 60 < angle <= 120:
        arc_color = "#FFC107"
        level = "Moderate"
    else:
        arc_color = "#F44336"
        level = "Needs Work"

    theta = np.linspace(np.pi, np.pi - (np.pi * (angle / 180)), 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=arc_color, linewidth=8)

    ax.text(0, 0, f"{int(angle)}°", ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    ax.text(0, -1.3, label, ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax.text(0, 1.2, level, ha='center', va='center', fontsize=14, color=arc_color, fontweight='bold')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=120)
    buf.seek(0)
    return buf

def main():
    st.set_page_config(layout="wide", page_title="NeuroTrack Pro", page_icon="🧠")
    st.title("🧠 NeuroTrack Pro | Stroke Therapy Monitoring")

    if "left_angles" not in st.session_state:
        st.session_state.left_angles = []
    if "right_angles" not in st.session_state:
        st.session_state.right_angles = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "last_df" not in st.session_state:
        st.session_state.last_df = None
    if "last_left_meter" not in st.session_state:
        st.session_state.last_left_meter = None
    if "last_right_meter" not in st.session_state:
        st.session_state.last_right_meter = None

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.subheader("Live Motion Analysis")
        ctx = webrtc_streamer(
            key="stream",
            video_processor_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.subheader("Left Arm Mobility")
        meter_left = st.empty()

    with col3:
        st.subheader("Right Arm Mobility")
        meter_right = st.empty()

    graph_placeholder = st.empty()
    timer_placeholder = st.empty()

    while ctx.state.playing:
        if ctx.video_processor:
            left_angle = ctx.video_processor.left_angle
            right_angle = ctx.video_processor.right_angle

            st.session_state.left_angles.append(left_angle)
            st.session_state.right_angles.append(right_angle)
            st.session_state.timestamps.append(time.time())

            buf_left = draw_angle_meter(left_angle, "Left Arm")
            buf_right = draw_angle_meter(right_angle, "Right Arm")
            meter_left.image(buf_left)
            meter_right.image(buf_right)

            st.session_state.last_left_meter = buf_left
            st.session_state.last_right_meter = buf_right

            df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "Left Arm Angle": st.session_state.left_angles,
                "Right Arm Angle": st.session_state.right_angles
            })
            df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
            graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

            st.session_state.last_df = df

            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed, 60)
            timer_placeholder.markdown(f"⏳ Session Duration: {mins:02d}:{secs:02d}")

        time.sleep(0.1)

    if not ctx.state.playing:
        if st.session_state.last_df is not None:
            st.success("✅ Session Completed Successfully!")

            if st.session_state.last_left_meter and st.session_state.last_right_meter:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Final Left Arm Reading")
                    st.image(st.session_state.last_left_meter)
                with col2:
                    st.subheader("Final Right Arm Reading")
                    st.image(st.session_state.last_right_meter)

            st.subheader("Session Summary")
            st.line_chart(st.session_state.last_df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

            csv = st.session_state.last_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Session Data (CSV)",
                data=csv,
                file_name="neurotrack_session_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
