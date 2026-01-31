
import cv2
import numpy as np

# Finding template matches in an image

# Load the main image and template
image = cv2.imread('images/Playing-Cards.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_copy = image.copy()

template = cv2.imread('images/jack-of-diamonds-template.jpg', 0)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template.shape[::-1]

# Perform template matching
result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.425
locations = np.where(result >= threshold)
print(locations[::-1])
# exit()

# Draw bounding boxes around matched regions
for pt in zip(*locations[::-1]):
    cv2.rectangle(image_copy, pt, (pt[0] + template_w, pt[1] + template_h), (255, 0, 0), 1)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Template Matches', image_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

