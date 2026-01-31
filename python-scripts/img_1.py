
import cv2

# Load an image
image = cv2.imread ('images/interior-residence-nice-10.jpg')
if image is not None:
    # Display the image in a window
    cv2.imshow ('Interior Residence', image)
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not load image.")
