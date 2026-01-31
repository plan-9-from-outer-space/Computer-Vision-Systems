
import cv2
import numpy as np

# Load an image in grayscale
image = cv2.imread('images\\Red green blue balls.png', cv2.IMREAD_GRAYSCALE)

# Use Canny edge dete3ction (it is a series of steps) to find edges in the image
edges = cv2.Canny(image, 100, 200)

cv2.imshow('Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

