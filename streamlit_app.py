import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import threading
import queue

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
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
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0
        self.pose = mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self._lock = threading.Lock()
        
    def get_angles(self):
        """Thread-safe method to get current angles"""
        with self._lock:
            return self.left_angle, self.right_angle
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Create a copy for drawing
            mask = img.copy()
            landmarks = results.pose_landmarks.landmark
            
            # Draw pose connections
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Check if landmarks are visible
                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * img.shape[1]), int(start.y * img.shape[0])
                    x2, y2 = int(end.x * img.shape[1]), int(end.y * img.shape[0])
                    cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw landmarks
            for idx, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    if idx in important_body_indices:
                        cv2.circle(mask, (x, y), 8, (0, 255, 0), -1)
                        cv2.circle(mask, (x, y), 12, (0, 180, 0), 2)
                    else:
                        cv2.circle(mask, (x, y), 3, (0, 0, 255), -1)
                        cv2.circle(mask, (x, y), 6, (0, 0, 180), 1)
            
            # Calculate angles with thread safety
            try:
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                
                # Check if required landmarks are visible
                if (left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and
                    right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5):
                    
                    left_shoulder_pos = [left_shoulder.x * img.shape[1], left_shoulder.y * img.shape[0]]
                    left_elbow_pos = [left_elbow.x * img.shape[1], left_elbow.y * img.shape[0]]
                    
                    right_shoulder_pos = [right_shoulder.x * img.shape[1], right_shoulder.y * img.shape[0]]
                    right_elbow_pos = [right_elbow.x * img.shape[1], right_elbow.y * img.shape[0]]
                    
                    with self._lock:
                        self.left_angle = 180 - calculate_angle(
                            left_elbow_pos, left_shoulder_pos,
                            [left_shoulder_pos[0], left_shoulder_pos[1] - 100]
                        )
                        self.right_angle = 180 - calculate_angle(
                            right_elbow_pos, right_shoulder_pos,
                            [right_shoulder_pos[0], right_shoulder_pos[1] - 100]
                        )
                        
            except Exception as e:
                # If angle calculation fails, keep previous values
                pass
            
            return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_angle_meter(angle, label):
    """Create a circular progress meter for angle visualization"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Background circle
    circle = plt.Circle((0, 0), 1, color=(0.2, 0.2, 0.2), fill=True)
    ax.add_artist(circle)
    
    # Determine color and level based on angle
    if angle > 120:
        arc_color = "#4CAF50"  # Green
        level = "Excellent"
    elif 60 < angle <= 120:
        arc_color = "#FFC107"  # Amber
        level = "Moderate"
    else:
        arc_color = "#F44336"  # Red
        level = "Needs Work"
    
    # Draw arc
    theta = np.linspace(np.pi, np.pi - (np.pi * (min(angle, 180) / 180)), 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=arc_color, linewidth=8)
    
    # Add text
    ax.text(0, 0, f"{int(angle)}°", ha='center', va='center', 
            fontsize=20, color='white', fontweight='bold')
    ax.text(0, -1.3, label, ha='center', va='center', 
            fontsize=14, color='white', fontweight='bold')
    ax.text(0, 1.2, level, ha='center', va='center', 
            fontsize=14, color=arc_color, fontweight='bold')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', 
                transparent=True, dpi=120, facecolor='black')
    buf.seek(0)
    plt.close(fig)
    return buf

def main():
    st.set_page_config(
        layout="wide",
        page_title="NeuroTrack Pro | Stroke Therapy Monitoring",
        page_icon="🧠"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            background-color: #f5f9fc;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
        }
        .header {
            color: #2c3e50;
            padding: 1rem;
            border-radius: 10px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .timer {
            font-size: 1.2rem;
            color: #3498db;
            font-weight: bold;
        }
        .success-box {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:#2c3e50;">🧠 NeuroTrack Pro</h1>
        <p style="margin:0; color:#7f8c8d;">AI-Powered Stroke Rehabilitation Progress Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "angle_data" not in st.session_state:
        st.session_state.angle_data = {
            "left_angles": [],
            "right_angles": [],
            "timestamps": [],
            "start_time": time.time()
        }
    
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    
    # Layout
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Live Motion Analysis</h3>
        """, unsafe_allow_html=True)
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        ctx = webrtc_streamer(
            key="pose-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Left Arm Mobility</h4>
        """, unsafe_allow_html=True)
        left_meter_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Right Arm Mobility</h4>
        """, unsafe_allow_html=True)
        right_meter_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Progress chart
    st.markdown("""
    <div class="card">
        <h3 style="color:#2c3e50; margin-bottom:1rem;">Progress Over Time</h3>
    """, unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Session info
    col_timer, col_controls = st.columns([2, 1])
    
    with col_timer:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Session Details</h3>
        """, unsafe_allow_html=True)
        timer_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_controls:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Controls</h3>
        """, unsafe_allow_html=True)
        
        if st.button("Start Recording" if not st.session_state.is_recording else "Stop Recording"):
            st.session_state.is_recording = not st.session_state.is_recording
            if st.session_state.is_recording:
                st.session_state.angle_data = {
                    "left_angles": [],
                    "right_angles": [],
                    "timestamps": [],
                    "start_time": time.time()
                }
        
        if st.button("Reset Session"):
            st.session_state.angle_data = {
                "left_angles": [],
                "right_angles": [],
                "timestamps": [],
                "start_time": time.time()
            }
            st.session_state.is_recording = False
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data collection and visualization
    if ctx.video_processor and st.session_state.is_recording:
        try:
            left_angle, right_angle = ctx.video_processor.get_angles()
            
            # Update session data
            current_time = time.time()
            st.session_state.angle_data["left_angles"].append(left_angle)
            st.session_state.angle_data["right_angles"].append(right_angle)
            st.session_state.angle_data["timestamps"].append(current_time)
            
            # Update meters
            left_buf = draw_angle_meter(left_angle, "Left Arm")
            right_buf = draw_angle_meter(right_angle, "Right Arm")
            
            left_meter_placeholder.image(left_buf)
            right_meter_placeholder.image(right_buf)
            
            # Update chart
            if len(st.session_state.angle_data["timestamps"]) > 1:
                df = pd.DataFrame({
                    "Time": st.session_state.angle_data["timestamps"],
                    "Left Arm Angle": st.session_state.angle_data["left_angles"],
                    "Right Arm Angle": st.session_state.angle_data["right_angles"]
                })
                df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
                
                chart_placeholder.line_chart(
                    df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]]
                )
            
            # Update timer
            elapsed = int(current_time - st.session_state.angle_data["start_time"])
            mins, secs = divmod(elapsed, 60)
            timer_placeholder.markdown(f"""
            <div class="timer">
                ⏳ Recording: {mins:02d}:{secs:02d} | 
                📊 Data Points: {len(st.session_state.angle_data["timestamps"])}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    elif ctx.video_processor and not st.session_state.is_recording:
        # Still show current angles even when not recording
        try:
            left_angle, right_angle = ctx.video_processor.get_angles()
            
            left_buf = draw_angle_meter(left_angle, "Left Arm")
            right_buf = draw_angle_meter(right_angle, "Right Arm")
            
            left_meter_placeholder.image(left_buf)
            right_meter_placeholder.image(right_buf)
            
            timer_placeholder.markdown("""
            <div class="timer">
                ⏸️ Ready to Record | Click 'Start Recording' to begin
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            pass
    
    # Download functionality
    if (len(st.session_state.angle_data["timestamps"]) > 0 and 
        not st.session_state.is_recording):
        
        st.markdown("""
        <div class="success-box">
            <h3 style="color:white; margin:0;">✅ Session Data Available</h3>
            <p style="color:white; margin:0;">Your rehabilitation data is ready for download.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create DataFrame for download
        df = pd.DataFrame({
            "Timestamp": st.session_state.angle_data["timestamps"],
            "Left Arm Angle": st.session_state.angle_data["left_angles"],
            "Right Arm Angle": st.session_state.angle_data["right_angles"]
        })
        df["Relative Time (seconds)"] = df["Timestamp"] - df["Timestamp"].iloc[0]
        
        # Display final chart
        if len(df) > 1:
            chart_placeholder.line_chart(
                df.set_index("Relative Time (seconds)")[["Left Arm Angle", "Right Arm Angle"]]
            )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Session Data (CSV)",
            data=csv,
            file_name=f"neurotrack_session_{int(time.time())}.csv",
            mime="text/csv",
        )
        
        # Session statistics
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Session Statistics</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{int(df['Relative Time (seconds)'].max())}s")
        with col2:
            st.metric("Data Points", len(df))
        with col3:
            st.metric("Avg Left Angle", f"{df['Left Arm Angle'].mean():.1f}°")
        with col4:
            st.metric("Avg Right Angle", f"{df['Right Arm Angle'].mean():.1f}°")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
