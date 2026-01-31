import cv2

# Load an image
image = cv2.imread('images/interior-residence-nice-10.jpg')
if image is None:
    print("Error: Could not load image.")
    exit()

# Draw a rectangle
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 3)
# Draw a circle
cv2.circle(image, (300, 300), 50, (255, 0, 0), -1)
# Draw a line
cv2.line(image, (400, 400), (500, 500), (0, 0, 255), 5)
# Put text
cv2.putText(image, 'Hello from OpenCV', (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show the modified image
cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

