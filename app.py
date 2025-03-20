import os
import cv2
import tempfile
import time
import numpy as np
import streamlit as st
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor
import gc
from queue import Queue
import threading


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

@st.cache_resource
def configure_torch():
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = False
        # Prevent JIT compilation issues
        torch.jit.disable_jit()
    return True

# Initialize torch configuration
_ = configure_torch()


# Add near the top of your file
frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)

def process_frames_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        try:
            # Process frame with model
            results = yolo_model(frame)
            result_queue.put(results)
        except Exception as e:
            result_queue.put(None)
        finally:
            frame_queue.task_done()

# Configure Streamlit
st.set_page_config(
    page_title="Utility Counter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Optimize memory usage
@st.cache_resource
def init_settings():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

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
def load_models():
    """Load and cache models"""
    try:
        # Set model path
        model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        
        # Load model with error handling
        if not os.path.exists(model_path):
            st.error("Model file not found!")
            return None
            
        model = YOLO(model_path, task='detect')
        
        # Optimize for inference
        model.to('cpu' if not torch.cuda.is_available() else 'cuda')
        model.conf = 0.4
        model.iou = 0.45
        
        if torch.cuda.is_available():
            model.fuse()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model with error handling
yolo_model = load_models()
if yolo_model is None:
    st.error("Failed to initialize model!")
    st.stop()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="clkjEzWr0PTQR8J0SgjV"
)

# Add the new function:
def get_video_capture(source):
    """Try different video capture methods"""
    if source == "Webcam":
        # Try different camera backends
        backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]
        for backend in backends:
            cv2.setNumThreads(1)  # Reduce threading issues
            for index in [0, 1, 2, -1]:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        # Set buffer size to reduce lag
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                except Exception:
                    continue
                
        # Fallback message
        st.error("⚠️ Could not access webcam. Please upload a video instead.")
        return None
    else:
        return cv2.VideoCapture(source)

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
        global frame_count
        
        try:
            cap = get_video_capture(video_source)
            if cap is None:
                return
    
            # Start processing worker
            worker = threading.Thread(target=process_frames_worker, daemon=True)
            worker.start()
    
            # Initialize display components
            frame_placeholder = st.empty()
            stats_placeholder = st.sidebar.empty()
            progress_bar = st.progress(0)
    
            frame_count = 0
            skip_frames = 3  # Process every nth frame
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
    
                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue
    
                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))
                
                # Add frame to processing queue
                if not frame_queue.full():
                    frame_queue.put(frame)
                
                # Get results if available
                if not result_queue.empty():
                    results = result_queue.get()
                    if results:
                        # Draw detections
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                b = box.xyxy[0]
                                cv2.rectangle(frame, 
                                            (int(b[0]), int(b[1])), 
                                            (int(b[2]), int(b[3])), 
                                            (0, 255, 0), 2)
    
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", 
                                     use_column_width=True)
    
                # Update progress
                if frame_count % 30 == 0:
                    gc.collect()
                    progress_bar.progress(frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            # Cleanup
            frame_queue.put(None)  # Signal worker to stop
            worker.join()
            cap.release()
            cv2.destroyAllWindows()

    # Handle video source selection
    # Handle video source selection
    if source == "Webcam":
        st.sidebar.warning("🔴 Press Stop to end the stream")
        # Try webcam with index 0 first
        process_video("Webcam")  # Changed from process_video(0)
        
    elif source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "📤 Upload Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
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
