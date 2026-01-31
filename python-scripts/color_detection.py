
import cv2
import numpy as np

# Load a video file
cap = cv2.VideoCapture('videos/video6.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the lower and upper bounds for each color in HSV space
color_bounds = {
    # part 1: shade of color
    # part 2: saturation
    # part 3: brightness
    'red': ((0, 120, 70), (10, 255, 255)),
    'green': ((36, 100, 100), (86, 255, 255)),
    'blue': ((94, 80, 2), (126, 255, 255))
}

# Assign colors for bounding boxes
box_colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}

# Loop over the video frames
while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert the frame to HSV color space (helps to detect colors better)
    hsv_frame = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)

    # Loop over each color to detect
    for color, (lower, upper) in color_bounds.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # Detect the colors in the frame
        # Create a mask for the specific current color
        mask = cv2.inRange (hsv_frame, lower_bound, upper_bound)

        # Find contours in the mask (outlines of objects)
        contours, _ = cv2.findContours (mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected contours
        for contour in contours:
            area = cv2.contourArea (contour)
            if area > 500:  # Filter out small areas to reduce noise
                x, y, w, h = cv2.boundingRect (contour)
                cv2.rectangle (frame, (x, y), (x + w, y + h), box_colors[color], 2)
                cv2.putText (frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_colors[color], 2)

    # Display the resulting frame
    cv2.imshow('Color Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
