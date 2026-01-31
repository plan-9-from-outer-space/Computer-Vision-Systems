
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
image = cv2.imread('images/IM-0030-0001.jpeg', 0) # grayscale

# Calculate histogram (channel [0] for grayscale image)
image_histogram = cv2.calcHist(
    images = [image], 
    channels = [0], 
    mask = None, 
    histSize = [256], # bins
    ranges = [0, 256])

# Normalize the histogram
# histogram = cv2.normalize(histogram, histogram).flatten()

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image) # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# Calculate histogram of equalized image 
equalized_histogram = cv2.calcHist(
    images = [equalized_image], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])
# equalized_histogram = cv2.normalize(equalized_histogram, equalized_histogram).flatten()

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE (clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply (image)

# Calculate histogram of CLAHE image
clahe_histogram = cv2.calcHist(
    images = [clahe_image], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])

# Display the results
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.plot(image_histogram)
plt.title('Original Histogram'), plt.xlim([0, 256])

plt.subplot(232), plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.plot(equalized_histogram)
plt.title('Equalized Histogram'), plt.xlim([0, 256])

plt.subplot(233), plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Image'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.plot(clahe_histogram)
plt.title('CLAHE Histogram'), plt.xlim([0, 256])

# plt.tight_layout()
plt.show()

# Save the plot (doesn't work well)
# Expand the plot and save manually instead
# plt.savefig('outputs/histogram_comparison.png')
