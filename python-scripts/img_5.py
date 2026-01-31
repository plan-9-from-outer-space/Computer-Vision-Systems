
import cv2
import numpy as np

# Load an image as grayscale
image = cv2.imread('images/interior-residence-nice-11.jpg', cv2.IMREAD_GRAYSCALE)

# Apply OTSU binarization
# Creates a histogram of the brightness values and finds an 
#   optimal threshold value to separate foreground and background.
ret, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)  # Print the optimal threshold value found by OTSU (0-255)

# Display the original and binary images
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

