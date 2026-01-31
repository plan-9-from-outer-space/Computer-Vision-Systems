import cv2
import numpy as np

# Load two images
image1 = cv2.imread('images/interior-residence-nice-7.jpg')
image2 = cv2.imread('images/interior-residence-nice-9.jpg')

# Ensure both images are of the same size
image1 = cv2.resize(image1, (500, 500))
image2 = cv2.resize(image2, (500, 500))

# Perform the XOR operation
xor_image = cv2.bitwise_xor(image1, image2)

# Display the original images and the result
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('XOR Image', xor_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

