
# Note: This Python script is designed to run with the latest version of MediaPipe (0.10.32 or later).

# Imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import numpy as np
import time

# The MediaPipe Face Landmarker requires a trained model that is compatible with this task.
model_path = '../models/face_landmarker_v2_with_blendshapes.task'

# Change to your video file path or use 0 for a webcam.
video_path = '../videos/Screen Recording 2026-01-28 145406.mp4' 

# Variables to support head movement detection.
nose_tip_positions = []
start_time = time.time()
movement_window_seconds = 2  # Time window to consider for head movement detection.

###################################################################
#### Function to draw the detected landmarks on a frame/image. ####
###################################################################

def draw_landmarks_on_image (rgb_image, detection_results):
    """Draw the detected face landmarks on the image or video frame."""

    # This allows modification of the global variable inside a function.
    global nose_tip_positions 

    face_landmarks_list = detection_results.face_landmarks
    annotated_image = np.copy (rgb_image)

    # Loop through the detected face landmarks.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks on the annotated image frame.
        drawing_utils.draw_landmarks (
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks (
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks (
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks (
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

        # The Face Landmark Model provides 468 3D landmarks.
        nose_tip = face_landmarks[1]
        left_eye = face_landmarks[473] # 33
        right_eye = face_landmarks[468] # 263
        mouth_center = face_landmarks[14]
        
        # Get coordinates of key points.
        # Example: NormalizedLandmark(x=0.49070340394973755, y=0.5004143714904785, z=-0.061409562826156616, ...)
        image_height, image_width, _ = annotated_image.shape
        nose_tip_coords = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
        left_eye_coords = (int(left_eye.x * image_width), int(left_eye.y * image_height))
        right_eye_coords = (int(right_eye.x * image_width), int(right_eye.y * image_height))
        mouth_center_coords = (int(mouth_center.x * image_width), int(mouth_center.y * image_height))

        # Draw circles at the selected key points.
        cv2.circle (annotated_image, nose_tip_coords, 5, (0, 255, 0), -1)     # Green for nose tip
        cv2.circle (annotated_image, left_eye_coords, 5, (255, 0, 0), -1)     # Blue for left eye
        cv2.circle (annotated_image, right_eye_coords, 5, (255, 0, 255), -1)  # Yellow for right eye
        cv2.circle (annotated_image, mouth_center_coords, 5, (0, 0, 255), -1) # Red for mouth center

        ###############################################
        # Head movement code.
        ###############################################
        # 
        # Store nose tip positions (for the last N seconds).
        nose_tip_positions.append((nose_tip_coords[0], nose_tip_coords[1], time.time()))
        nose_tip_positions = [pos for pos in nose_tip_positions if time.time() - pos[2] <= movement_window_seconds]
        # 
        # Detect head movements ('No' and 'Yes').
        horizontal_movements, vertical_movements = detect_head_movement (nose_tip_positions)
        # 
        # Detect 'No' head gesture
        if len(horizontal_movements) >= 2:
            if horizontal_movements[-1] < 0 < horizontal_movements[-2] or horizontal_movements[-1] > 0 > horizontal_movements[-2]:
                cv2.putText (annotated_image, "NO", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 
        # Detect 'Yes' head gesture
        if len(vertical_movements) >= 2:
            if vertical_movements[-1] < 0 < vertical_movements[-2] or vertical_movements[-1] > 0 > vertical_movements[-2]:
                cv2.putText (annotated_image, "YES", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_image

###############################################
# Function to detect head movement.
# Note: Adjust the threshold to make it more or less sensitive to your video.
###############################################

def detect_head_movement (nose_positions, threshold=10):
    """Detect head movement based on nose tip position movement over time."""
    horizontal_movements = []
    vertical_movements = []
    for i in range(1, len(nose_positions)):
        prev_x, prev_y, _ = nose_positions[i-1]
        curr_x, curr_y, _ = nose_positions[i]
        delta_x = curr_x - prev_x
        delta_y = curr_y - prev_y
        if abs (delta_x) > threshold:
            horizontal_movements.append(delta_x)
        if abs (delta_y) > threshold:
            vertical_movements.append(delta_y)
    return horizontal_movements, vertical_movements

##########################################
# Main Python script code starts here.
##########################################

# Mediapipe task construction for Face Landmarker.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the options for the face landmarker instance with the video mode.
options = FaceLandmarkerOptions (
    base_options = BaseOptions (model_asset_path=model_path),
    running_mode = VisionRunningMode.VIDEO,
    # Additional parameters (with default values shown)
    num_faces = 1,
    min_face_detection_confidence = 0.5,
    min_face_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_face_blendshapes = False,
    output_facial_transformation_matrixes = False
)

# options = FaceLandmarkerOptions (
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

with FaceLandmarker.create_from_options (options) as landmarker:
    # The landmarker is now initialized.

    # Use OpenCV to load the input video.
    cap = cv2.VideoCapture (video_path)
 
    # Get the frame rate of the video using OpenCV (not used).
    # frame_rate = cap.get (cv2.CAP_PROP_FPS)

    # Loop through each frame in the video.
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Flip the frame horizontally, if using a webcam.
        if (video_path == 0):
            frame = cv2.flip (frame, 1)

        # Get the current frame timestamp in milliseconds.
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Convert the frame color from BGR to RGB as MediaPipe uses RGB format.
        rgb_frame = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a numpy array.
        numpy_frame_from_opencv = np.array (rgb_frame)

        # Convert the numpy array to a MediaPipe Image object.
        mp_image = mp.Image (
           image_format = mp.ImageFormat.SRGB, 
           data = numpy_frame_from_opencv)
    
        # Call the AI model: Perform face landmarking on the provided video frame.
        face_landmarker_results = landmarker.detect_for_video (mp_image, frame_timestamp_ms)

        # The Face Landmarker returns a FaceLandmarkerResult object for each detection run.
        # The result object contains a face mesh for each detected face, with coordinates
        # for each face landmark. Optionally, the result object can also contain blendshapes,
        # which denote facial expressions, and a facial transformation matrix to apply 
        # face effects on the detected landmarks.

        # Draw the face landmarks on the image.
        annotated_image = draw_landmarks_on_image (numpy_frame_from_opencv, face_landmarker_results)

        # Display the annotated image in a window.
        cv2.imshow ("Annotated Video Frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Hit the Escape (Esc) key to stop the program.
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

