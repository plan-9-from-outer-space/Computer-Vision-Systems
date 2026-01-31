import cv2

# Load an image
# image = cv2.imread('images/interior-residence-nice-10.jpg')
image = cv2.imread('images/Red green blue balls.png')

# create grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)

# OpenCV stores images in B-G-R format
# Split the color channels
blue, green, red = cv2.split(image)

# Each channel will be in grayscale (single chanel)
cv2.imshow('Blue Channel', blue)
cv2.imshow('Green Channel', green)
cv2.imshow('Red Channel', red)

cv2.waitKey(0)
cv2.destroyAllWindows()
