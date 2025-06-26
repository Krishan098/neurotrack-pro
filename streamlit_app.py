import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import threading
import logging
import asyncio
from typing import Optional

# Configure logging to reduce noise
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

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
    """Calculate angle between three points"""
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle
    except:
        return 0

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._lock = threading.Lock()
        self.frame_count = 0
        
    def get_current_angles(self) -> tuple:
        """Thread-safe method to get current angles"""
        with self._lock:
            return self.left_angle, self.right_angle
    
    def recv(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror the image
            h, w = img.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_img)
            
            # Create output image
            output_img = img.copy()
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    output_img, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Highlight important joints
                for idx in important_body_indices:
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        if landmark.visibility > 0.5:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(output_img, (x, y), 8, (0, 255, 0), -1)
                            cv2.circle(output_img, (x, y), 12, (0, 180, 0), 2)
                
                # Calculate angles if landmarks are available
                self._calculate_arm_angles(landmarks, w, h)
            
            # Process every few frames to reduce load
            self.frame_count += 1
            
            return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            # Return original frame if processing fails
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _calculate_arm_angles(self, landmarks, width, height):
        """Calculate arm angles with error handling"""
        try:
            # Get required landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # Check visibility
            if (left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and
                right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5):
                
                # Convert to pixel coordinates
                left_shoulder_pos = [left_shoulder.x * width, left_shoulder.y * height]
                left_elbow_pos = [left_elbow.x * width, left_elbow.y * height]
                right_shoulder_pos = [right_shoulder.x * width, right_shoulder.y * height]
                right_elbow_pos = [right_elbow.x * width, right_elbow.y * height]
                
                # Calculate angles (shoulder elevation)
                left_angle = 180 - calculate_angle(
                    left_elbow_pos, 
                    left_shoulder_pos,
                    [left_shoulder_pos[0], left_shoulder_pos[1] - 100]
                )
                
                right_angle = 180 - calculate_angle(
                    right_elbow_pos, 
                    right_shoulder_pos,
                    [right_shoulder_pos[0], right_shoulder_pos[1] - 100]
                )
                
                # Update angles with thread safety
                with self._lock:
                    self.left_angle = max(0, min(180, left_angle))
                    self.right_angle = max(0, min(180, right_angle))
                    
        except Exception as e:
            # Keep previous angles if calculation fails
            pass

def create_angle_gauge(angle: float, label: str) -> BytesIO:
    """Create a gauge visualization for angle measurement"""
    try:
        # Set up the plot
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # Background circle
        bg_circle = plt.Circle((0, 0), 1.2, color='#f0f0f0', fill=True)
        ax.add_artist(bg_circle)
        
        # Inner circle
        inner_circle = plt.Circle((0, 0), 1, color='white', fill=True)
        ax.add_artist(inner_circle)
        
        # Determine color based on angle
        if angle >= 120:
            color = '#4CAF50'  # Green
            status = 'Excellent'
        elif angle >= 90:
            color = '#FFC107'  # Amber
            status = 'Good'
        elif angle >= 60:
            color = '#FF9800'  # Orange
            status = 'Moderate'
        else:
            color = '#F44336'  # Red
            status = 'Needs Work'
        
        # Draw angle arc
        angle_rad = np.radians(angle)
        theta = np.linspace(0, angle_rad, 100)
        x_arc = 0.9 * np.cos(theta)
        y_arc = 0.9 * np.sin(theta)
        ax.plot(x_arc, y_arc, color=color, linewidth=8, solid_capstyle='round')
        
        # Add angle text
        ax.text(0, 0, f'{int(angle)}°', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='#333')
        
        # Add label
        ax.text(0, -1.4, label, ha='center', va='center', 
                fontsize=16, fontweight='bold', color='#333')
        
        # Add status
        ax.text(0, 1.3, status, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=color)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   dpi=100, facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        # Return empty buffer if visualization fails
        buf = BytesIO()
        return buf

def main():
    # Page configuration
    st.set_page_config(
        page_title="NeuroTrack Pro - Stroke Rehabilitation Monitor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 1rem;
        }
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin: 0.5rem;
            display: inline-block;
        }
        .recording { background: #4CAF50; color: white; }
        .stopped { background: #f44336; color: white; }
        .ready { background: #2196F3; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NeuroTrack Pro</h1>
        <p>AI-Powered Stroke Rehabilitation Progress Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'session_data' not in st.session_state:
        st.session_state.session_data = {
            'angles': {'left': [], 'right': [], 'timestamps': []},
            'is_recording': False,
            'start_time': None
        }
    
    # Create layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Video stream column
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📹 Live Video Analysis")
        
        # WebRTC Configuration with simpler settings
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        })
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="pose-analysis",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=PoseProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Left arm gauge
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("🦾 Left Arm")
        left_gauge_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right arm gauge
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("🦾 Right Arm")
        right_gauge_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Controls and status
    col_controls, col_status = st.columns([1, 2])
    
    with col_controls:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("🎮 Controls")
        
        if st.button("🔴 Start Recording" if not st.session_state.session_data['is_recording'] 
                    else "⏹️ Stop Recording", use_container_width=True):
            if not st.session_state.session_data['is_recording']:
                # Start recording
                st.session_state.session_data['is_recording'] = True
                st.session_state.session_data['start_time'] = time.time()
                st.session_state.session_data['angles'] = {'left': [], 'right': [], 'timestamps': []}
                st.success("Recording started!")
            else:
                # Stop recording
                st.session_state.session_data['is_recording'] = False
                st.success("Recording stopped!")
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.session_data = {
                'angles': {'left': [], 'right': [], 'timestamps': []},
                'is_recording': False,
                'start_time': None
            }
            st.success("Session reset!")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_status:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("📊 Session Status")
        
        # Display status
        if st.session_state.session_data['is_recording']:
            status_class = "recording"
            status_text = "🔴 RECORDING"
            if st.session_state.session_data['start_time']:
                elapsed = int(time.time() - st.session_state.session_data['start_time'])
                mins, secs = divmod(elapsed, 60)
                st.markdown(f"⏱️ Duration: {mins:02d}:{secs:02d}")
        else:
            status_class = "ready" if webrtc_ctx.state.playing else "stopped"
            status_text = "🟢 READY" if webrtc_ctx.state.playing else "⚫ STOPPED"
        
        st.markdown(f'<span class="status-indicator {status_class}">{status_text}</span>', 
                   unsafe_allow_html=True)
        
        # Data points
        data_points = len(st.session_state.session_data['angles']['left'])
        st.markdown(f"📈 Data Points: {data_points}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main processing loop
    if webrtc_ctx.video_processor:
        try:
            # Get current angles
            left_angle, right_angle = webrtc_ctx.video_processor.get_current_angles()
            
            # Update gauges
            left_buf = create_angle_gauge(left_angle, "Left Arm")
            right_buf = create_angle_gauge(right_angle, "Right Arm")
            
            left_gauge_placeholder.image(left_buf, use_column_width=True)
            right_gauge_placeholder.image(right_buf, use_column_width=True)
            
            # Record data if recording is active
            if st.session_state.session_data['is_recording']:
                current_time = time.time()
                st.session_state.session_data['angles']['left'].append(left_angle)
                st.session_state.session_data['angles']['right'].append(right_angle)
                st.session_state.session_data['angles']['timestamps'].append(current_time)
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    
    # Chart and download section
    if len(st.session_state.session_data['angles']['left']) > 0:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Progress Chart")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time': st.session_state.session_data['angles']['timestamps'],
            'Left Arm': st.session_state.session_data['angles']['left'],
            'Right Arm': st.session_state.session_data['angles']['right']
        })
        
        if st.session_state.session_data['start_time']:
            df['Relative Time'] = df['Time'] - st.session_state.session_data['start_time']
            chart_df = df.set_index('Relative Time')[['Left Arm', 'Right Arm']]
            st.line_chart(chart_df)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download section
        if not st.session_state.session_data['is_recording'] and len(df) > 10:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("💾 Download Session Data")
            
            # Session summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{df['Relative Time'].max():.1f}s")
            with col2:
                st.metric("Data Points", len(df))
            with col3:
                st.metric("Avg Left Angle", f"{df['Left Arm'].mean():.1f}°")
            with col4:
                st.metric("Avg Right Angle", f"{df['Right Arm'].mean():.1f}°")
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Session Data (CSV)",
                data=csv_data,
                file_name=f"neurotrack_session_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
