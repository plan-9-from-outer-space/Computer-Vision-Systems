# -----------------------------
# Standard Python libraries
# -----------------------------
import os                  # For file/folder handling
import time                # For timestamps and delays
import urllib.request      # For downloading model file
    
# -----------------------------
# External libraries
# -----------------------------
import cv2                 # OpenCV (camera + image processing)
import numpy as np         # Numerical arrays
import mediapipe as mp     # MediaPipe core
    
# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
    
# Drawing utilities
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# -----------------------------
# Settings
# -----------------------------

# Webcam index (0 = default camera, or video pathname)
# CAMERA_INDEX = 0
CAMERA_INDEX = '../videos/Screen Recording 2026-01-28 145406.mp4' 

# How often to print coordinates (seconds)
PRINT_EVERY_SECONDS = 0.25
    
# Folder for AI model
MODEL_DIR = "models"
    
# Name of model file
MODEL_FILENAME = "face_landmarker_v2_with_blendshapes.task"
    
# Full path to model file
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
# Official MediaPipe model download URL
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
    
# -----------------------------
# Main landmark indices
# (Face Mesh points we want)
# -----------------------------
MAIN_POINTS = {
    
    # Nose
    "nose_tip": 1,
    
    # Eyes
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    
    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    
    # Chin
    "chin": 152,
}
    
# -----------------------------
# Helper Functions
# -----------------------------
    
def ensure_model_file():
    """
    Check if the AI model exists.
    If not, download it automatically.
    """
    
    # Create models folder if missing
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # If file already exists → do nothing
    if os.path.exists(MODEL_PATH):
        return
    
    # Inform user
    print(f"Model not found at {MODEL_PATH}")
    print("Downloading model (one-time)...")
    
    # Download model from Google servers
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    except Exception as e:
        # Stop program if download fails
        raise RuntimeError(
            "Failed to download the model.\n"
            "Check internet connection.\n"
            f"Error: {e}"
        )
    
    print("Download complete.")
    
def lm_to_xy(lm, w, h):
    """
    Convert MediaPipe landmark to:
    
    - normalized (0..1)
    - pixel (screen coords)
    """
    
    # Normalized values (0-1)
    xn = float(lm.x)
    yn = float(lm.y)
    
    # Convert to pixels
    xp = int(xn * w)
    yp = int(yn * h)
    
    return (xn, yn), (xp, yp)
    
def print_points(points_dict):
    """
    Print all face points in one line.
    """
    
    parts = []
    
    # Loop over each point
    for name, d in points_dict.items():
    
        # Normalized coords
        xn, yn = d["norm"]
    
        # Pixel coords
        xp, yp = d["px"]
    
        # Format output
        parts.append(
            f"{name}: norm=({xn:.3f},{yn:.3f}) px=({xp},{yp})"
        )
    
    # Print single line
    print(" | ".join(parts))
    
def draw_keypoints(img_rgb, points_dict):
    """
    Draw green dots and names on face parts.
    """
    
    for name, d in points_dict.items():
    
        # Pixel position
        x, y = d["px"]
    
        # Draw circle
        cv2.circle(img_rgb, (x, y), 4, (0, 255, 0), -1)
    
        # Draw text label
        cv2.putText(
            img_rgb,
            name,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
# -----------------------------
# Main Program
# -----------------------------
def main():
    
    # Make sure model exists
    ensure_model_file()
    
    # -------------------------
    # Create FaceLandmarker
    # -------------------------
    
    # Load model settings
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )
    
    # Configure detector
    options = vision.FaceLandmarkerOptions(
    
        base_options=base_options,
    
        # VIDEO mode = continuous frames
        running_mode=vision.RunningMode.VIDEO,
    
        # Only detect one face
        num_faces=1,
    
        # Disable extra outputs
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    
        # Confidence thresholds
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    # Create detector object
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # -------------------------
    # Open webcam
    # -------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam {CAMERA_INDEX}"
        )
    
    # Timer for printing
    last_print = 0.0
    
    # -------------------------
    # Main loop
    # -------------------------
    while True:
    
        # Read frame from webcam
        ok, frame_bgr = cap.read()
    
        if not ok:
            break
    
        # Mirror image (like mirror)
        frame_bgr = cv2.flip(frame_bgr, 1)
    
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )
    
        # Get image size
        h, w, _ = frame_rgb.shape
    
        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(frame_rgb)
        )
    
        # Current time in milliseconds
        ts_ms = int(time.time() * 1000)
    
        # Run face detection
        result = detector.detect_for_video(
            mp_image,
            ts_ms
        )
    
        # Copy frame for drawing
        annotated = frame_rgb.copy()
    
        # -------------------------
        # If face detected
        # -------------------------
        if result.face_landmarks:
    
            # Get first detected face
            face = result.face_landmarks[0]
    
            # Draw face mesh (triangles)
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )
    
            # Draw contours (edges)
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
    
            # Draw iris (eyes)
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
    
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
    
            # -------------------------
            # Collect main points
            # -------------------------
            points = {}
    
            for name, idx in MAIN_POINTS.items():
    
                # Check index valid
                if idx < len(face):
    
                    lm = face[idx]
    
                    # Convert coords
                    (xn, yn), (xp, yp) = lm_to_xy(lm, w, h)
    
                    points[name] = {
                        "norm": (xn, yn),
                        "px": (xp, yp),
                    }
    
            # Draw dots and labels
            draw_keypoints(annotated, points)
    
            # Print periodically
            now = time.time()
    
            if points and (now - last_print) >= PRINT_EVERY_SECONDS:
                # print_points(points)
                last_print = now
    
        # -------------------------
        # Show window
        # -------------------------
        out_bgr = cv2.cvtColor(
            annotated,
            cv2.COLOR_RGB2BGR
        )
    
        cv2.imshow(
            "MediaPipe Face Landmarker (ESC to quit)",
            out_bgr
        )
    
        # Exit when ESC pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    cv2.destroyAllWindows()
    
# Run program
if __name__ == "__main__":
    main()

