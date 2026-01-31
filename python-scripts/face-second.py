
# Imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import numpy as np
# import matplotlib.pyplot as plt # Only needed for blendshapes visualization

# The MediaPipe Face Landmarker task requires a trained model that is compatible with this task.
model_path = '../models/face_landmarker_v2_with_blendshapes.task'

###################################################################
#### Function to draw the detected landmarks on a frame/image. ####
###################################################################

def draw_landmarks_on_image (rgb_image, detection_results):
    """Draw the detected face landmarks on the image or video frame."""

    face_landmarks_list = detection_results.face_landmarks
    annotated_image = np.copy (rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        # print(face_landmarks)

        # Draw the face landmarks on the frame.
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

        # Face Landmark Model: Provides 468 3D landmarks.
        # nose_tip = face_landmarks[vision.FaceLandmark.NOSE_TIP]
        # left_eye = face_landmarks[vision.FaceLandmark.LEFT_EYE]
        # right_eye = face_landmarks[vision.FaceLandmark.RIGHT_EYE]
        # mouth_center = face_landmarks[vision.FaceLandmark.MOUTH_CENTER]
        nose_tip = face_landmarks[1]
        left_eye = face_landmarks[473] # 33
        right_eye = face_landmarks[468] # 263
        mouth_center = face_landmarks[13]
        # print(nose_tip, left_eye, right_eye, mouth_center)

        # Example:
        # NormalizedLandmark(x=0.49070340394973755, y=0.5004143714904785, z=-0.061409562826156616, visibility=None, presence=None, name=None) 
        # NormalizedLandmark(x=0.4201613962650299, y=0.40931710600852966, z=0.03650977835059166, visibility=None, presence=None, name=None) 
        # NormalizedLandmark(x=0.5868799686431885, y=0.3977035582065582, z=0.006838127039372921, visibility=None, presence=None, name=None) 
        # NormalizedLandmark(x=0.4980510473251343, y=0.5629154443740845, z=-0.026454409584403038, visibility=None, presence=None, name=None)
        
        # Get coordinates of key points.
        image_height, image_width, _ = annotated_image.shape
        nose_tip_coords = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
        left_eye_coords = (int(left_eye.x * image_width), int(left_eye.y * image_height))
        right_eye_coords = (int(right_eye.x * image_width), int(right_eye.y * image_height))
        mouth_center_coords = (int(mouth_center.x * image_width), int(mouth_center.y * image_height))

        # Draw circles at key points.
        cv2.circle (annotated_image, nose_tip_coords, 3, (0, 255, 0), -1)     # Green for nose tip
        cv2.circle (annotated_image, left_eye_coords, 3, (255, 0, 0), -1)     # Blue for left eye
        cv2.circle (annotated_image, right_eye_coords, 3, (255, 0, 255), -1)  # Yellow for right eye
        cv2.circle (annotated_image, mouth_center_coords, 3, (0, 0, 255), -1) # Red for mouth center

    return annotated_image

# Task construction for video files.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode.
options = FaceLandmarkerOptions (
    base_options = BaseOptions (model_asset_path=model_path),
    running_mode = VisionRunningMode.VIDEO, # IMAGE, VIDEO, LIVE_STREAM
    num_faces = 1,
    # optional parameters (with default values shown)
    min_face_detection_confidence = 0.5,
    min_face_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_face_blendshapes = False,
    output_facial_transformation_matrixes = False
)

with FaceLandmarker.create_from_options (options) as landmarker:
    # The landmarker is now initialized.

    # Use OpenCV to load the input video.
    video_path = '../videos/Screen Recording 2026-01-28 110158.mp4'
    cap = cv2.VideoCapture (video_path)
 
    # Load the frame rate of the video using OpenCV.
    # It is needed to calculate the timestamp for each frame.
    frame_rate = cap.get (cv2.CAP_PROP_FPS)
  
    # Loop through each frame in the video.
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Flip the frame if we are using a web cam
        if (video_path == 0):
            frame = cv2.flip (frame, 1)

        # Get the current frame timestamp in milliseconds.
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Convert the frame color from BGR to RGB as MediaPipe uses RGB format.
        rgb_frame = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a numpy array.
        numpy_frame_from_opencv = np.array (rgb_frame)

        # Convert the frame received from OpenCV to a MediaPipe Image object.
        mp_image = mp.Image (
           image_format = mp.ImageFormat.SRGB, 
           data = numpy_frame_from_opencv)
    
        # Call the model: Perform face landmarking on the provided frame.
        # The face landmarker must be created with the video mode.
        face_landmarker_results = landmarker.detect_for_video (mp_image, frame_timestamp_ms)
        # print(face_landmarker_results)

        # Handle and display results:
        # The Face Landmarker returns a FaceLandmarkerResult object for each detection run.
        # The result object contains a face mesh for each detected face, with coordinates
        # for each face landmark. Optionally, the result object can also contain blendshapes,
        # which denote facial expressions, and a facial transformation matrix to apply 
        # face effects on the detected landmarks.

        # Draw the face landmarks on the image.
        annotated_image = draw_landmarks_on_image (numpy_frame_from_opencv, face_landmarker_results)

        # Display the annotated image in a window.
        cv2.imshow("Annotated Video Frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Original Video Frame", frame)

        # Hit the Escape (Esc) key to stop the program.
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

