import streamlit as st
import cv2
import tempfile
import torch
import time
import threading
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor

# Custom CSS styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(45deg, #2a0a3a, #1a0829);
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4a1f5a 0%, #321245 100%) !important;
        border-right: 3px solid #7a4f8a;
        padding: 20px !important;
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 1.8rem !important;
        color: #ffffff !important;
        background: linear-gradient(45deg, #e6b0ff, #c38fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 15px 0 !important;
        padding: 10px 0;
        border-bottom: 2px solid #6a3a7a;
    }
    
    /* Radio Buttons in Sidebar */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.2rem !important;
        color: #fff !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        background: rgba(90, 50, 110, 0.7) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 2px solid #8a5a9a;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(170, 100, 200, 0.2) !important;
        transform: translateX(10px);
    }
    
    /* File Uploader in Sidebar */
    [data-testid="stSidebar"] .stFileUploader {
        background: rgba(90, 50, 110, 0.7) !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border: 2px solid #8a5a9a;
    }
    
    [data-testid="stSidebar"] .stFileUploader label {
        font-size: 1.2rem !important;
        color: #fff !important;
    }
    
    /* Buttons in Sidebar */
    [data-testid="stSidebar"] .stButton button {
        font-size: 1.3rem !important;
        padding: 15px 30px !important;
        border-radius: 10px !important;
        background: linear-gradient(45deg, #6a3a7a, #8a5a9a);
    }
    
    /* Headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #c38fff !important;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #ffffff !important;
        font-weight: bold;
    }
    .stRadio [role="radiogroup"] {
        background: #4a2a5a;
        padding: 15px;
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #4a2a5a;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Statistics display */
    .stAlert {
        background: rgba(90, 50, 110, 0.9) !important;
        border: 2px solid #a87fc1;
        border-radius: 15px;
    }
    
    /* Buttons */
    .stButton button {
        background: #6a3a7a;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: #8a5a9a;
        transform: scale(1.05);
    }
    
    /* Video frame */
    .stImage {
        border: 3px solid #6a3a7a;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(170, 100, 200, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize models and clients
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="clkjEzWr0PTQR8J0SgjV"
)

# Main app
st.title("A1 Future Technologies")
app_mode = st.sidebar.radio("Select Application Mode:", 
                           ["Person Counter", "Inventory Management"])

# ...existing code...

if app_mode == "Person Counter":
    st.header("👥 Person Counting")

    detections = []
    num_people = 0
    frame_count = 0
    process_every_n_frames = 5
    
    # Person Counter specific settings
    st.sidebar.header("Person Counter Settings")
    source = st.sidebar.radio("Input Source:", ["Webcam", "Upload Video"])


    # Thread-safe inference function
    def run_inference(frame):
        global detections, num_people
        cv2.imwrite("temp_frame.jpg", frame)  # Save frame temporarily
        result = CLIENT.infer("temp_frame.jpg", model_id="crowd-density-ou3ne/1")
        detections = result.get('predictions', [])
        num_people = len(detections)


    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    # Streamlit Video Display
    frame_placeholder = st.empty()
    
    # Initialize session state
    if 'frame_placeholder' not in st.session_state:
        st.session_state.frame_placeholder = st.empty()
    
    # Person detection function
    def detect_persons(frame):
        try:
            # Convert and save frame
            cv2.imwrite("temp_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Run inference
            result = CLIENT.infer(
                "temp_frame.jpg",
                model_id="person-detection-s1sxr/1",  # Updated model ID
                overlap=0.45
            )
            
            detections = result.get('predictions', [])
            return detections
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return []

    def process_video(video_source):
        global future, frame_count

        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        resize_factor = 0.5
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)

        if not cap.isOpened():
            st.error("🚨 Error: Could not open video source!")
            return
        
        process_every_n_frames = 10

        frame_buffer = []
        buffer_size = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends

            frame_count += 1

            frame = cv2.resize(frame, (new_width, new_height))

            # Start inference on every Nth frame in a separate thread
            if frame_count % process_every_n_frames == 0:
                if future is None or future.done():  # Ensure only one inference at a time
                    inference_frame = cv2.resize(frame.copy(), (640, 480))
                    future = executor.submit(run_inference, inference_frame)
                    
            # Draw bounding boxes from the last completed inference
            for detection in detections:
            # Scale coordinates back to resized frame
                x = int(detection['x'] * resize_factor)
                y = int(detection['y'] * resize_factor)
                w = int(detection['width'] * resize_factor)
                h = int(detection['height'] * resize_factor)
                
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 100, 200), 2)
                
                # Only show confidence for higher confidence detections
                if detection['confidence'] > 0.5:
                    label = f"{detection['class']} {detection['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 100, 200), 2)

        # Display count on frame
            cv2.putText(frame, f"👥 Total People: {num_people}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Reduced font size

            # Convert BGR to RGB for Streamlit
            frame_buffer.append(frame)
            if len(frame_buffer) >= buffer_size:
                display_frame = frame_buffer.pop(0)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        cap.release()

    # Handle video source selection
    if source == "Webcam":
        st.sidebar.warning("🔴 Press Stop to end the stream")
        process_video(0)
        
    elif source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "📤 Upload Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            if st.sidebar.button("Start Processing"):
                process_video(tfile.name)
                tfile.close()



elif app_mode == "Inventory Management":

    CONF_THRESHOLD = 0.4

    st.header("📦 Inventory Management")
    
    # Inventory Management specific settings
    st.sidebar.header("Inventory Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Video", 
                                            type=["mp4", "avi", "mov"])

    # Load YOLO model
    yolo_model = load_yolo_model()

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.sidebar.button("Start Processing"):
            tracker = DeepSort(
                embedder="mobilenet",
                embedder_gpu=torch.cuda.is_available()
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                st.stop()

            counted_objects = set()
            frame_placeholder = st.empty()
            status_text = st.sidebar.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = yolo_model(frame)
                detections = []
                for result in results:
                    for box in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = box

                        # Apply confidence threshold
                        if score > CONF_THRESHOLD:
                            detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

                tracks = tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    if track_id not in counted_objects:
                        counted_objects.add(track_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                cv2.putText(frame, f"Total Objects: {len(counted_objects)}", 
                          (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                status_text.info(f"Processing... Current count: {len(counted_objects)}")

            cap.release()
            st.sidebar.success(f"Final count: {len(counted_objects)} objects")
            st.balloons()
