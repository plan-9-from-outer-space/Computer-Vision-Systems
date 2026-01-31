
import cv2
import numpy as np

# Detect corners in an image

# Load an image
image = cv2.imread('images\\interior-residence-nice-10.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect corners using the Shi-Tomasi method
corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
# Convert results to integer
corners = np.intp(corners)

cv2.imshow('Original Image', image)

# Draw corners on the image
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Display the image with corners
cv2.imshow('Shi-Tomasi Corners Detected', image)

# Harris corner detection
gray_float = np.float32(gray)
harris_corners = cv2.cornerHarris(gray_float, blockSize=5, ksize=3, k=0.04)
# Highlight the corners
image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
cv2.imshow('Harris Corners', image)

# Dilated Harris corners for better visibility
dilated_corners = cv2.dilate(harris_corners, None)
image[dilated_corners > 0.01 * dilated_corners.max()] = [0, 255, 0]
cv2.imshow('Dilated Harris Corners', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

