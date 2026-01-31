
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# Define a face landmark function
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

# Load a video
video_path = '../videos/people.mp4'
cap = cv2.VideoCapture(video_path)

# STEP: Create an FaceLandmarker object.
base_options = python.BaseOptions(
   model_asset_path='../face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
   base_options=base_options,
   output_face_blendshapes=True,
   output_facial_transformation_matrixes=True,
   num_faces=1
) 
detector = vision.FaceLandmarker.create_from_options(options)

# STEP: Load the input image.
success, frame = cap.read()

# Change frame color space from BGR to RGB as OpenCV uses BGR by default.
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = np.array(frame)

# STEP: Detect face landmarks from the input image.
detection_result = detector.detect(frame)

# STEP: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image (frame, detection_result)
cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.imshow(annotated_image)

# if cv2.waitKey(5) & 0xFF == 27:
#     break

cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()
exit(0)




# # Loop over the video frames
# with mp_face_mesh.FaceMesh(
#         # static_image_mode = False,
#         # max_num_faces = 1,
#         # refine_landmarks = True,
#         min_detection_confidence = 0.5,
#         min_tracking_confidence = 0.5
#     ) as face_mesh:

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success: break

#         # Flip the image horizontally for a later selfie-view display
#         frame = cv2.flip(frame, 1) 

#         # Convert the BGR image to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the image and find face landmarks
#         results = face_mesh.process(frame_rgb) 

#         # Draw the face mesh annotations on the image
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks (
#                     image=frame,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     # landmark_drawing_spec=None,
#                     # connection_drawing_spec=mp_drawing_styles
#                     # .get_default_face_mesh_tesselation_style()
#                 ) 
#                 # mp_drawing.draw_landmarks (
#                 #     image=frame,
#                 #     landmark_list=face_landmarks,
#                 #     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 #     landmark_drawing_spec=None,
#                 #     connection_drawing_spec=mp_drawing_styles
#                 #     .get_default_face_mesh_contours_style()
#                 # )
#                 # mp_drawing.draw_landmarks (
#                 #     image=frame,
#                 #     landmark_list=face_landmarks,
#                 #     connections=mp_face_mesh.FACEMESH_IRISES,
#                 #     landmark_drawing_spec=None,
#                 #     connection_drawing_spec=mp_drawing_styles
#                 #     .get_default_face_mesh_iris_connections_style()
#                 # )

#         # Display the resulting frame
#         cv2.imshow('MediaPipe Face Mesh', frame)

#         if cv2.waitKey(5) & 0xFF == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()

