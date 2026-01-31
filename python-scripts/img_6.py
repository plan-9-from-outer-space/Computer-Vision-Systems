
import cv2
import numpy as np

# Load image as grayscale, then convert to a binary image
image = cv2.imread('images/interior-residence-nice-8.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
binary_image = cv2.bitwise_not(binary_image)
print(cv2.THRESH_BINARY)

# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Apply a closing operation
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
# Invert the image back
inverted_image = cv2.bitwise_not(closed_image)

# Display the results
cv2.imshow('Original Binary Image', binary_image)
cv2.imshow('Closed Image', closed_image)
cv2.imshow('Inverted Closed Image', inverted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

