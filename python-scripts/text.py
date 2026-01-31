
import cv2
import pytesseract
import numpy as np

# Set path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract\\tesseract.exe'

# Loan an image
image_path = '../images/How-to-Make-Text-Stand-Out-And-More-Readable.jpg'
image = cv2.imread(image_path)
cv2.imshow ('Original Image', image) 

# Convert image to grayscale (to improve OCR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow ('Gray Image', gray_image) 

# Apply thresholding to improve text detection (binary, black & white)
_, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow ('Threshold Image', thresh_image)

# Apply dilation to connect text regions
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(thresh_image, kernel, iterations=1)
cv2.imshow ('Dilated Image', dilated_image)

# Use Tesseract to perform OCR on the image
text = pytesseract.image_to_string (dilated_image)
print("Extracted Text: ")
print(text)

# Draw bounding box for each detected word
h, w, _ = image.shape 
boxes = pytesseract.image_to_boxes (dilated_image)
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle (image, (x, h - y2), (x2, h - y), (0, 255, 0), 2)

# Display the image with extracted text overlay 
cv2.putText (image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow ('Image with Extracted Text', image) 

cv2.waitKey(0)
cv2.destroyAllWindows()

