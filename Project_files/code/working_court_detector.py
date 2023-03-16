import cv2
import numpy as np

ROOT_DIR = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/Dataset/train/background_subtraction_test'

# Step 1: Collect a set of image frames that contain the badminton court from various angles.
# filepath = f'{ROOT_DIR}/video_label_1_119_jpg.rf.fb6487e3c2d8212062cc1bb07b960784.jpg'
# filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/UntrackedFiles/out/rgb_mask.png'
# filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/code/lucas-kanade/test/court_edges.png'
filepath = '/Users/bryanmbanefo/Desktop/GIT_REPOS/CS231A/Project_files/image_frames_ours/frame_0001.png'

# Load image
img = cv2.imread(filepath)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Apply adaptive thresholding to isolate the court from the background
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
cv2.imwrite("./tutor_results/thresh.png", thresh)

# Apply morphological opening to remove small holes in the court
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(opening, 50, 150, apertureSize=3)
cv2.imwrite("./tutor_results/edges.png", edges)

# Find contours in the image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

new_img = img.copy()
# Iterate through contours and find the one with four corners
rect_contours = []
for contour in contours:
   # Approximate the contour to reduce the number of points
   approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
   
   # If the contour has four corners, it's likely the badminton court
   if len(approx) == 4:
       # Verify that the contour is roughly rectangular in shape
       if cv2.isContourConvex(approx):
           rect_contours.append(approx)
           new_img = cv2.drawContours(new_img, [approx], 0, (0, 0, 255), 2)
           # break
           
areas = [cv2.contourArea(c) for c in rect_contours]
max_index = np.argmax(areas)
largest_contour = rect_contours[max_index]
approx = cv2.approxPolyDP(largest_contour, 0.01*cv2.arcLength(largest_contour, True), True)

# Draw the contour on the original image
cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)

# Draw circles on the corners of the court
for corner in approx:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# Show the original image with the detected corners
cv2.imwrite("./tutor_results/new_img.png", new_img)
cv2.imshow('Badminton Court', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 


# Explanation:
# The code follows the following steps:

# Load the image and convert it to grayscale.
# Apply Gaussian blur to remove noise.
# Apply adaptive thresholding to isolate the court from the background.
# Apply morphological opening to remove small holes in the court.
# Apply Canny edge detection to detect edges.
# Find contours in the image.
# Iterate through the contours and find the one with four corners.
# Verify that the contour is roughly rectangular in shape.
# Draw the contour on the original image and circles on the corners of the court.
# Show the original image with the detected corners.
# The main steps for detecting the badminton court are the use of adaptive thresholding and finding contours with four corners. 
# The code approximates the contours to reduce the number of points, 
# and it verifies that the contour is roughly rectangular in shape. 
# Finally, it draws the contour on the original image and circles on the corners of the court to 
# highlight the detected area.

# This code can be used as a starting point for similar image processing tasks, where a specific object or feature needs to be detected in an image.

