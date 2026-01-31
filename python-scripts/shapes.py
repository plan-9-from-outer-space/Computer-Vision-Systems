
import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/shapes_2.png') 

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Apply Gaussian blur (for noise reduction)
blurred = cv2.GaussianBlur(gray, (5, 5), 0) 

# Edge detection using Canny (using edge thresholds)
edges = cv2.Canny(blurred, 50, 150) 

# Find contours (shape boudaries, continuous lines)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a function to detect shapes
def detect_shape (contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    
    return shape

# Loop through contours and identify shapes
for contour in contours: 
    area = cv2.contourArea(contour)
    if area < 400: continue # ignore small contours
    shape = detect_shape(contour)
    shape_name = shape.capitalize()
    M = cv2.moments(contour)
    # Find the center of the shape to place the label
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Draw the contour and label the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape_name, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display the result
cv2.imshow('Shapes', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

